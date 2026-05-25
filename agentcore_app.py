"""
StudyAgent AI - AgentCore Runtime Application (Multi-Agent)
Main entry point for AgentCore deployment.

Architecture:
- Orchestrator agent delegates to specialist agents via @tool functions:
  - generate_quiz: Quiz generation specialist
  - generate_flashcards: Flashcard generation specialist
  - generate_summary: Summarization specialist
  - web_search: Web search specialist (Tavily-powered)
- Verification agent runs post-response to check grounding (with search-augmented verification)
- Documents stored in S3, loaded as context
- AgentCore Memory for conversation persistence (STM + LTM)

Region: eu-west-1 (only EU region with AgentCore)
This file is the entry point for `agentcore launch`.
In cloud, IAM roles are used (no AWS_PROFILE needed).
"""
import os
import re
import io
import json
import time
import logging

from strands import Agent
from strands.agent.conversation_manager import SummarizingConversationManager
from strands.models.bedrock import BedrockModel
from strands_tools import calculator

# AgentCore imports
from bedrock_agentcore.runtime import BedrockAgentCoreApp

from src.config import (
    AWS_REGION,
    ORCHESTRATOR_MODEL_ID,
    DEFAULT_TEMPERATURE,
    MAX_MESSAGE_LENGTH,
    MEMORY_ID,
    S3_BUCKET_NAME,
    SYSTEM_PROMPT_TEMPLATE,
    TAVILY_SECRET_NAME,
)
from src.tools.quiz import generate_quiz
from src.tools.flashcard import generate_flashcards
from src.tools.summarizer import generate_summary
from src.tools.search import web_search
from src.tools.read_url import read_url
from src.tools.verification import run_verification
from src.tools._context import set_context, get_and_clear_last_flashcard_output

# =============================================================================
# Configuration
# =============================================================================
os.environ["AWS_DEFAULT_REGION"] = AWS_REGION
os.environ["BYPASS_TOOL_CONSENT"] = "true"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("studyagent")


# =============================================================================
# Tavily API Key (from Secrets Manager)
# =============================================================================
def _load_tavily_api_key():
    """Load Tavily API key from AWS Secrets Manager and set as env var."""
    if os.environ.get("TAVILY_API_KEY"):
        return  # Already set (e.g. local dev)
    try:
        import boto3

        client = boto3.client("secretsmanager", region_name=AWS_REGION)
        response = client.get_secret_value(SecretId=TAVILY_SECRET_NAME)
        secret = json.loads(response["SecretString"])
        os.environ["TAVILY_API_KEY"] = secret["TAVILY_API_KEY"]
        logger.info("Tavily API key loaded from Secrets Manager")
    except Exception as e:
        logger.warning(f"Could not load Tavily API key: {e}")


_load_tavily_api_key()

# =============================================================================
# Module-Level Cached Resources
# =============================================================================
app = BedrockAgentCoreApp()
model = BedrockModel(
    model_id=ORCHESTRATOR_MODEL_ID,
    region_name=AWS_REGION,
    temperature=DEFAULT_TEMPERATURE,
)

# Document context cache (refreshed every 5 minutes)
_doc_cache = {"content": "", "module": None, "timestamp": 0}
_DOC_CACHE_TTL = 300  # 5 minutes


# =============================================================================
# S3 Document Context (with caching)
# =============================================================================
def get_document_context(module: str = None, max_chars: int = 150000) -> str:
    """Load documents from S3 as context, with caching."""
    global _doc_cache

    now = time.time()
    if (
        _doc_cache["content"]
        and _doc_cache["module"] == module
        and (now - _doc_cache["timestamp"]) < _DOC_CACHE_TTL
    ):
        return _doc_cache["content"]

    try:
        import boto3

        s3 = boto3.client("s3", region_name=AWS_REGION)
        prefix = "documents/"
        if module:
            prefix = f"documents/{module.replace(' ', '_').lower()}/"

        response = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=prefix)

        context_parts = []
        total_chars = 0

        for obj in response.get("Contents", []):
            if total_chars >= max_chars:
                break
            if obj["Key"].endswith("/"):
                continue

            try:
                doc_response = s3.get_object(Bucket=S3_BUCKET_NAME, Key=obj["Key"])
                content = doc_response["Body"].read()
                ext = obj["Key"].lower().split(".")[-1]

                text = None
                if ext == "pdf":
                    try:
                        from pypdf import PdfReader

                        pdf_reader = PdfReader(io.BytesIO(content))
                        text = "\n".join(
                            page.extract_text()
                            for page in pdf_reader.pages
                            if page.extract_text()
                        )
                    except Exception as e:
                        logger.warning(f"Could not parse PDF {obj['Key']}: {e}")
                else:
                    try:
                        text = content.decode("utf-8")
                    except UnicodeDecodeError:
                        continue

                if text:
                    remaining = max_chars - total_chars
                    if len(text) > remaining:
                        text = text[:remaining] + "...[truncated]"

                    filename = obj["Key"].split("/")[-1]
                    module_name = obj["Key"].split("/")[1].replace("_", " ").title() if "/" in obj["Key"] else "unknown"
                    doc_header = f"\n\n{'='*60}\n[START OF {filename} | Module: {module_name} | {len(text)} characters of content follow]\n{'='*60}\n"
                    doc_footer = f"\n[END OF {filename}]\n"
                    context_parts.append(doc_header + text + doc_footer)
                    total_chars += len(doc_header) + len(text) + len(doc_footer)
            except Exception as e:
                logger.warning(f"Could not read document {obj['Key']}: {e}")

        result = "".join(context_parts) if context_parts else ""
        _doc_cache = {"content": result, "module": module, "timestamp": now}
        logger.info(f"Loaded {len(result)} chars of document context from S3")
        return result
    except Exception as e:
        logger.warning(f"Could not load document context: {e}")
        return _doc_cache.get("content", "")


# =============================================================================
# Memory Session Manager (with RetrievalConfig for LTM)
# =============================================================================
def create_session_manager(session_id: str, actor_id: str):
    """
    Create AgentCore Memory session manager with LTM retrieval.

    RetrievalConfig ensures LTM memories (facts, preferences, summaries)
    are actually retrieved and injected into the agent's context.
    Without this, LTM extraction happens but memories are never used.
    """
    try:
        from bedrock_agentcore.memory.integrations.strands.config import (
            AgentCoreMemoryConfig,
            RetrievalConfig,
        )
        from bedrock_agentcore.memory.integrations.strands.session_manager import (
            AgentCoreMemorySessionManager,
        )

        config = AgentCoreMemoryConfig(
            memory_id=MEMORY_ID,
            session_id=session_id,
            actor_id=actor_id,
            retrieval_config={
                "facts/{actorId}/": RetrievalConfig(
                    top_k=5, relevance_score=0.3
                ),
                "preferences/{actorId}/": RetrievalConfig(
                    top_k=3, relevance_score=0.5
                ),
                "summaries/{actorId}/{sessionId}/": RetrievalConfig(
                    top_k=2, relevance_score=0.4
                ),
            },
        )

        session_manager = AgentCoreMemorySessionManager(
            agentcore_memory_config=config,
            region_name=AWS_REGION,
        )
        logger.info(
            f"Memory session created - session: {session_id}, actor: {actor_id}"
        )
        return session_manager
    except Exception as e:
        logger.warning(f"Could not create memory session: {e}")
        return None


# =============================================================================
# File Inventory Builder
# =============================================================================
def _build_file_inventory() -> str:
    """Build file inventory from S3 listing (not from loaded content).

    This ensures the agent always knows about ALL modules and files,
    even if some document content is truncated due to context limits.
    """
    try:
        import boto3

        s3 = boto3.client("s3", region_name=AWS_REGION)
        response = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix="documents/")
        modules = {}
        for obj in response.get("Contents", []):
            if obj["Key"].endswith("/"):
                continue
            parts = obj["Key"].split("/")
            if len(parts) >= 3:
                module_name = parts[1].replace("_", " ").title()
                filename = parts[-1]
                modules.setdefault(module_name, []).append(filename)
        if not modules:
            return "[No documents uploaded yet. Upload study materials to get started!]"
        lines = []
        for module, files in sorted(modules.items()):
            lines.append(f"- **{module}**: {', '.join(files)}")
        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"Could not build file inventory from S3: {e}")
        return "[Could not load file inventory]"


# =============================================================================
# Validation Skip Logic (reused from orchestrator for latency optimization)
# =============================================================================
_SKIP_VALIDATION_PATTERNS = re.compile(
    r"(?i)^(hi|hello|hey|thanks|thank you|bye|goodbye|good morning|good evening|"
    r"what can you do|who are you|how are you)",
)

_SKIP_VALIDATION_RESPONSE_PATTERNS = re.compile(
    r"(?i)(FRONT:|BACK:|Question \d|^\d+\.\s.*\?\s*$|"
    r"here are.*flashcard|here.*quiz|practice question)",
    re.MULTILINE,
)


def _should_skip_validation(question: str, response: str) -> bool:
    """Skip validation for greetings, quizzes, and flashcards to halve latency."""
    if _SKIP_VALIDATION_PATTERNS.match(question.strip()):
        return True
    if _SKIP_VALIDATION_RESPONSE_PATTERNS.search(response[:500]):
        return True
    return False


# =============================================================================
# AgentCore Entry Point
# =============================================================================
@app.entrypoint
def invoke(payload, context=None):
    """
    Main entry point for AgentCore Runtime invocations.

    Args:
        payload: Contains 'prompt', optionally 'module', 'session_id', 'actor_id'
        context: AgentCore context with session_id

    Returns:
        Dict with result, session_id, and validation
    """
    try:
        user_message = payload.get("prompt", "Hello!")
        module = payload.get("module", None)
        actor_id = payload.get("actor_id", "default_student")

        # Input guardrail
        if len(user_message) > MAX_MESSAGE_LENGTH:
            return {
                "result": f"Message too long ({len(user_message)} chars). Maximum is {MAX_MESSAGE_LENGTH} characters.",
                "error": "message_too_long",
            }

        # Session ID: prefer context (AgentCore provides it), fall back to payload
        session_id = None
        if context:
            session_id = getattr(context, "session_id", None)
        if not session_id:
            session_id = payload.get("session_id")
        if not session_id:
            from datetime import datetime

            session_id = f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        start_time = time.monotonic()

        logger.info(f"Received: {user_message[:100]}...")
        logger.info(f"Session: {session_id}, Actor: {actor_id}, Module: {module}")

        # Load document context (cached) - "General" means all documents
        effective_module = None if module in (None, "General") else module
        document_context = get_document_context(module=effective_module)

        # Set shared context for specialist agent tools
        set_context(document_context, module or "General")

        # Build file inventory from S3 listing (independent of loaded content)
        file_inventory = _build_file_inventory()

        # Build system prompt
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            document_context=document_context
            if document_context
            else "[No documents uploaded yet. Upload study materials to get started!]",
            file_inventory=file_inventory,
        )

        # Create session manager with LTM retrieval
        session_manager = create_session_manager(session_id, actor_id)

        # Create orchestrator agent with specialist tools
        agent_kwargs = {
            "model": model,
            "system_prompt": system_prompt,
            "tools": [
                calculator,
                generate_quiz,
                generate_flashcards,
                generate_summary,
                web_search,
                read_url,
            ],
            "conversation_manager": SummarizingConversationManager(
                summary_ratio=0.4,
                preserve_recent_messages=6,
            ),
        }
        if session_manager:
            agent_kwargs["session_manager"] = session_manager

        agent = Agent(**agent_kwargs)
        result = agent(user_message)
        response_text = str(result)

        # Capture structured flashcard data if tool stored any
        fc_data = get_and_clear_last_flashcard_output()

        # Run verification agent (with reasoning capture)
        if _should_skip_validation(user_message, response_text):
            validation = {
                "grounded": "yes",
                "confidence": "high",
                "notes": "Non-factual response (greeting, quiz, or flashcard)",
                "reasoning": "This response type does not require factual verification.",
            }
        else:
            validation = run_verification(
                user_message, response_text, document_context
            )
        logger.info(
            f"Verification: grounded={validation['grounded']}, confidence={validation['confidence']}"
        )

        # Log analytics to DynamoDB
        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        tools_used = []
        if result and hasattr(result, "metrics"):
            metrics = result.metrics
            if hasattr(metrics, "tool_use_count") and metrics.tool_use_count:
                tools_used = list(metrics.tool_use_count.keys()) if hasattr(metrics.tool_use_count, "keys") else []
        try:
            from src.utils.analytics import log_interaction
            log_interaction(
                session_id=session_id,
                module=module or "General",
                question=user_message,
                tools_used=tools_used,
                response_time_ms=elapsed_ms,
                verification_grounded=validation.get("grounded", "N/A"),
                verification_confidence=validation.get("confidence", "low"),
            )
        except Exception as e:
            logger.warning(f"Analytics logging failed: {e}")

        response = {
            "result": response_text,
            "session_id": session_id,
            "validation": validation,
        }
        if fc_data:
            response["flashcard_data"] = fc_data

        return response
    except Exception as e:
        logger.error(f"Agent invocation error: {e}", exc_info=True)
        return {
            "result": "I'm sorry, an error occurred while processing your request. Please try again.",
            "error": str(e)[:200],
        }


# =============================================================================
# Local Development Entry Point
# =============================================================================
if __name__ == "__main__":
    print(
        f"""
========================================================
  StudyAgent AI - AgentCore Runtime (Local Mode)
  Region:    {AWS_REGION}
  S3 Bucket: {S3_BUCKET_NAME}
  Memory ID: {MEMORY_ID}
  Model:     Claude Haiku 4.5
========================================================
"""
    )
    app.run()
