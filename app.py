"""
StudyAgent AI - Flask Backend (AgentCore Runtime Client)
Main application entry point for local development.


Architecture:
- Flask frontend serves the chat UI
- Chat requests are forwarded to the deployed AgentCore Runtime agent
- Documents uploaded to S3
- AgentCore handles memory, orchestration, verification, and observability
"""
import os
import io
import csv
import json
import random
import tempfile
import logging

from flask import Flask, request, jsonify, Response, render_template, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

from src.config import (
    AWS_PROFILE,
    AWS_REGION,
    APP_NAME,
    DEBUG_MODE,
    MAX_UPLOAD_SIZE_MB,
    MAX_MESSAGE_LENGTH,
    SUPPORTED_FILE_TYPES,
    get_boto3_session,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# =============================================================================
# Application Setup
# =============================================================================
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# =============================================================================
# AgentCore Runtime Client
# =============================================================================
AGENT_RUNTIME_ARN = os.environ.get("AGENTCORE_RUNTIME_ARN", "")
if not AGENT_RUNTIME_ARN:
    raise RuntimeError(
        "AGENTCORE_RUNTIME_ARN env var is required. "
        "Set it to the ARN of your deployed AgentCore Runtime agent."
    )

_agentcore_client = None


def get_agentcore_client():
    """Get cached AgentCore client."""
    global _agentcore_client
    if _agentcore_client is None:
        session = get_boto3_session()
        _agentcore_client = session.client("bedrock-agentcore", region_name=AWS_REGION)
    return _agentcore_client


# Track AgentCore session IDs per chat session for conversation continuity
_runtime_sessions = {}


def invoke_agentcore(message: str, session_id: str = "default", module: str = None):
    """
    Invoke the deployed AgentCore agent and return its response.

    Uses runtimeSessionId for conversation continuity across messages.
    """
    client = get_agentcore_client()

    payload = {"prompt": message}
    if module:
        payload["module"] = module
    payload["actor_id"] = "default_student"
    payload["session_id"] = session_id

    invoke_kwargs = {
        "agentRuntimeArn": AGENT_RUNTIME_ARN,
        "contentType": "application/json",
        "accept": "application/json",
        "payload": json.dumps(payload).encode("utf-8"),
    }

    # Reuse runtime session for conversation continuity
    if session_id in _runtime_sessions:
        invoke_kwargs["runtimeSessionId"] = _runtime_sessions[session_id]

    logger.info(f"Invoking AgentCore: {message[:80]}...")

    response = client.invoke_agent_runtime(**invoke_kwargs)

    # Store runtime session ID for future calls
    runtime_session_id = response.get("runtimeSessionId")
    if runtime_session_id:
        _runtime_sessions[session_id] = runtime_session_id

    # Read response body (streaming body object)
    response_body = response["response"].read().decode("utf-8")

    try:
        result = json.loads(response_body)
    except json.JSONDecodeError:
        result = {"result": response_body}

    logger.info(f"AgentCore response: {str(result)[:200]}...")
    return result


def get_aws_resources():
    """Lazy import of AWS resources manager."""
    from src.utils.aws_resources import get_resource_manager
    return get_resource_manager()


# =============================================================================
# API Routes
# =============================================================================
@app.route("/")
def index():
    """Serve the main chat interface."""
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "app": APP_NAME,
        "aws_profile": AWS_PROFILE,
        "aws_region": AWS_REGION,
        "backend": "agentcore-runtime",
        "agent_arn": AGENT_RUNTIME_ARN,
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    """Non-streaming chat endpoint - forwards to AgentCore Runtime."""
    data = request.get_json()
    message = data.get("message", "")
    session_id = data.get("session_id", "default")
    module = data.get("module")

    if not message:
        return jsonify({"error": "Message is required"}), 400

    if len(message) > MAX_MESSAGE_LENGTH:
        return jsonify({"error": f"Message too long ({len(message)} chars). Maximum is {MAX_MESSAGE_LENGTH} characters."}), 400

    try:
        result = invoke_agentcore(message, session_id, module)
        return jsonify({"response": result.get("result", str(result)), "session_id": session_id})
    except Exception as e:
        logger.error(f"AgentCore invocation error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    """
    Streaming chat endpoint using Server-Sent Events (SSE).
    Invokes AgentCore and streams the response back to the frontend.
    AgentCore returns a complete response, so we emit it as a single text event
    followed by validation and done events.
    """
    data = request.get_json()
    message = data.get("message", "")
    session_id = data.get("session_id", "default")
    module = data.get("module")

    if not message:
        return jsonify({"error": "Message is required"}), 400

    if len(message) > MAX_MESSAGE_LENGTH:
        return jsonify({"error": f"Message too long ({len(message)} chars). Maximum is {MAX_MESSAGE_LENGTH} characters."}), 400

    def generate():
        """Generator for SSE - invokes AgentCore and emits response events."""
        try:
            result = invoke_agentcore(message, session_id, module)

            # Emit the main response text
            response_text = result.get("result", str(result))
            yield f"data: {json.dumps({'type': 'text', 'content': response_text})}\n\n"

            # Emit validation if present
            validation = result.get("validation")
            if validation:
                yield f"data: {json.dumps({'type': 'validation', 'grounded': validation.get('grounded', 'N/A'), 'confidence': validation.get('confidence', 'low'), 'notes': validation.get('notes', ''), 'reasoning': validation.get('reasoning', '')})}\n\n"

            # Emit flashcard data if present
            fc_data = result.get("flashcard_data")
            if fc_data:
                yield f"data: {json.dumps({'type': 'flashcard_data', 'format': fc_data.get('format', 'chat'), 'cards': fc_data.get('cards', []), 'deck_name': fc_data.get('deck_name', '')})}\n\n"

            # Emit analytics
            from src.utils.analytics import log_interaction
            interaction_id = log_interaction(
                session_id=session_id,
                module=module or "General",
                question=message,
                tools_used=[],
                response_time_ms=0,
                verification_grounded=validation.get("grounded", "N/A") if validation else "N/A",
                verification_confidence=validation.get("confidence", "low") if validation else "low",
            )
            yield f"data: {json.dumps({'type': 'analytics', 'interaction_id': interaction_id})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    """Record thumbs up/down feedback for an interaction."""
    data = request.get_json()
    interaction_id = data.get("interaction_id")
    feedback = data.get("feedback")  # "up" or "down"

    if not interaction_id or feedback not in ("up", "down"):
        return jsonify({"error": "interaction_id and feedback ('up'/'down') required"}), 400

    from src.utils.analytics import record_feedback
    success = record_feedback(interaction_id, feedback)
    if success:
        return jsonify({"success": True})
    return jsonify({"error": "Failed to record feedback"}), 500


@app.route("/api/analytics", methods=["GET"])
def analytics_summary():
    """Return aggregate pilot metrics from DynamoDB analytics table."""
    try:
        from src.utils.analytics import get_analytics_summary
        return jsonify(get_analytics_summary())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload", methods=["POST"])
def upload_document():
    """Upload a document to S3."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    module = request.form.get("module", "General")

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_FILE_TYPES:
        return jsonify({
            "error": f"Unsupported file type: {ext}. Supported: {SUPPORTED_FILE_TYPES}"
        }), 400

    filepath = None
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        resources = get_aws_resources()
        s3_uri = resources.upload_document_to_s3(filepath, module)

        os.remove(filepath)

        return jsonify({
            "success": True,
            "filename": filename,
            "module": module,
            "s3_uri": s3_uri,
        })

    except Exception as e:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": str(e)}), 500


@app.route("/api/documents", methods=["GET"])
def list_documents():
    """List all documents in S3."""
    module = request.args.get("module")

    try:
        resources = get_aws_resources()
        docs = resources.list_documents(module=module)
        return jsonify({"documents": docs})
    except Exception as e:
        return jsonify({"error": str(e), "documents": []}), 500


@app.route("/api/documents/<path:s3_key>", methods=["DELETE"])
def delete_document(s3_key: str):
    """Delete a document from S3."""
    try:
        resources = get_aws_resources()
        resources.delete_document(s3_key)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/modules", methods=["GET"])
def list_modules():
    """List all available modules (from S3 folder structure)."""
    try:
        resources = get_aws_resources()
        docs = resources.list_documents()

        modules = list(set(doc.get("module", "General") for doc in docs))
        if not modules:
            modules = ["General"]

        return jsonify({"modules": modules})
    except Exception:
        return jsonify({"modules": ["General"]})


@app.route("/api/session/<session_id>", methods=["DELETE"])
def clear_session(session_id: str):
    """Clear a chat session by removing the runtime session mapping."""
    if session_id in _runtime_sessions:
        del _runtime_sessions[session_id]
        return jsonify({"success": True})
    return jsonify({"success": True})  # No-op if no session exists


@app.route("/api/session/<session_id>/history", methods=["GET"])
def get_session_history(session_id: str):
    """Get chat history - not available when using AgentCore Runtime."""
    return jsonify({"history": []})


@app.route("/api/session/<session_id>/module", methods=["PUT"])
def set_session_module(session_id: str):
    """Set the active module for a session."""
    data = request.get_json()
    module = data.get("module", "General")
    return jsonify({"success": True, "module": module})


@app.route("/api/setup", methods=["POST"])
def setup_aws_resources():
    """Set up all required AWS resources (S3 bucket)."""
    try:
        resources = get_aws_resources()
        result = resources.setup_all_resources()
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =============================================================================
# Flashcard Export Endpoints
# =============================================================================
@app.route("/api/export/anki", methods=["POST"])
def export_anki():
    """Generate and download an Anki .apkg file from flashcard data."""
    try:
        import genanki

        data = request.get_json()
        deck_name = data.get("deck_name", "StudyAgent Flashcards")
        cards = data.get("cards", [])

        if not cards:
            return jsonify({"error": "No flashcard data provided"}), 400

        model_id = random.randrange(1 << 30, 1 << 31)
        deck_id = random.randrange(1 << 30, 1 << 31)

        model = genanki.Model(
            model_id,
            "StudyAgent Model",
            fields=[{"name": "Front"}, {"name": "Back"}],
            templates=[{
                "name": "Card 1",
                "qfmt": "{{Front}}",
                "afmt": '{{FrontSide}}<hr id="answer">{{Back}}',
            }],
            css=".card { font-family: arial; font-size: 18px; text-align: center; }",
        )

        deck = genanki.Deck(deck_id, deck_name)

        for card in cards:
            front = card.get("front", "").strip()
            back = card.get("back", "").strip()
            if front and back:
                note = genanki.Note(model=model, fields=[front, back])
                deck.add_note(note)

        tmp = tempfile.NamedTemporaryFile(suffix=".apkg", delete=False)
        try:
            genanki.Package(deck).write_to_file(tmp.name)
            return send_file(
                tmp.name,
                as_attachment=True,
                download_name=f"{deck_name.replace(' ', '_')}.apkg",
                mimetype="application/octet-stream",
            )
        finally:
            import atexit
            atexit.register(lambda: os.unlink(tmp.name) if os.path.exists(tmp.name) else None)

    except ImportError:
        return jsonify({"error": "genanki not installed. Run: pip install genanki"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/export/csv", methods=["POST"])
def export_csv():
    """Generate and download a CSV file from flashcard data."""
    data = request.get_json()
    deck_name = data.get("deck_name", "StudyAgent Flashcards")
    cards = data.get("cards", [])

    if not cards:
        return jsonify({"error": "No flashcard data provided"}), 400

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Front", "Back"])
    for card in cards:
        front = card.get("front", "").strip()
        back = card.get("back", "").strip()
        if front and back:
            writer.writerow([front, back])

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{deck_name.replace(" ", "_")}.csv"'
        },
    )


# =============================================================================
# Startup
# =============================================================================
if __name__ == "__main__":
    print(
        f"""
========================================================
  {APP_NAME}

  AWS Profile: {AWS_PROFILE}
  AWS Region:  {AWS_REGION}
  Backend:     AgentCore Runtime (deployed)
  Agent ARN:   {AGENT_RUNTIME_ARN}

  Document Storage: S3
  Memory: AgentCore Memory (STM + LTM)
  Observability: OpenTelemetry + X-Ray (cloud)

  Starting server on http://localhost:5001
========================================================
"""
    )
    app.run(host="0.0.0.0", port=5001, debug=DEBUG_MODE)
