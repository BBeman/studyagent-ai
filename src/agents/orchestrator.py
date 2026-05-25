"""
Orchestrator Agent - Multi-agent entry point for StudyAgent AI (local Flask app)

Architecture:
- Orchestrator agent delegates to specialist agents via @tool functions:
  - generate_quiz: Quiz generation specialist
  - generate_flashcards: Flashcard generation specialist
  - generate_summary: Summarization specialist
  - web_search: Web search specialist (Tavily-powered)
- Verification agent runs post-response to check grounding (with search-augmented verification)
- Documents stored in S3, loaded as context
- AgentCore Memory for conversation persistence (STM + LTM)
"""
import os
import re
import queue
import threading
import asyncio
import logging

os.environ.setdefault("BYPASS_TOOL_CONSENT", "true")

# Load Tavily API key from Secrets Manager for local Flask app
if not os.environ.get("TAVILY_API_KEY"):
    try:
        import json
        from src.config import AWS_REGION, TAVILY_SECRET_NAME, get_boto3_session

        _sm = get_boto3_session().client("secretsmanager", region_name=AWS_REGION)
        _secret = json.loads(
            _sm.get_secret_value(SecretId=TAVILY_SECRET_NAME)["SecretString"]
        )
        os.environ["TAVILY_API_KEY"] = _secret["TAVILY_API_KEY"]
        logging.getLogger("orchestrator").info("Tavily API key loaded from Secrets Manager")
    except Exception as _e:
        logging.getLogger("orchestrator").warning(f"Could not load Tavily API key: {_e}")

from strands import Agent
from strands.agent.conversation_manager import SummarizingConversationManager
from strands.models.bedrock import BedrockModel
from strands_tools import calculator
from typing import Generator

from src.config import (
    AWS_REGION,
    ORCHESTRATOR_MODEL_ID,
    DEFAULT_TEMPERATURE,
    MEMORY_ID,
    SYSTEM_PROMPT_TEMPLATE,
)
from src.tools.quiz import generate_quiz
from src.tools.flashcard import generate_flashcards
from src.tools.summarizer import generate_summary
from src.tools.search import web_search
from src.tools.read_url import read_url
from src.tools.verification import run_verification
from src.tools._context import set_context, get_and_clear_last_flashcard_output

logger = logging.getLogger("orchestrator")


_cached_model = None


def _get_bedrock_model():
    """Get cached Bedrock model instance. AWS_PROFILE is set via src.config env vars."""
    global _cached_model
    if _cached_model is None:
        _cached_model = BedrockModel(
            model_id=ORCHESTRATOR_MODEL_ID,
            region_name=AWS_REGION,
            temperature=DEFAULT_TEMPERATURE,
        )
    return _cached_model


def _create_session_manager(session_id: str, actor_id: str):
    """
    Create AgentCore Memory session manager with RetrievalConfig for LTM.

    Without RetrievalConfig, LTM strategies extract memories but they are
    never retrieved - cross-session memory would be broken.
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
        logger.info(f"Memory session created - session: {session_id}, actor: {actor_id}")
        return session_manager
    except Exception as e:
        logger.warning(f"Could not initialize AgentCore Memory: {e}")
        return None


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


class StudyAgentOrchestrator:
    """
    Multi-agent orchestrator for StudyAgent AI (local Flask app).

    Architecture:
    - Orchestrator agent with specialist agent tools (quiz, flashcard, summarizer)
    - Calculator and web_search (Tavily) tools for math and web search
    - Verification agent runs post-response for hallucination detection
    - AgentCore Memory for cross-session persistence
    """

    def __init__(
        self,
        module_context: str = "No module selected",
        session_id: str = None,
        actor_id: str = None,
    ):
        from datetime import datetime

        self.module_context = module_context
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.actor_id = actor_id or "default_user"
        self._document_context = ""
        self.agent = None
        self._initialize_agent()

    def _get_document_context(self) -> str:
        """Load documents from S3 as context."""
        if self._document_context:
            return self._document_context

        try:
            from src.utils.aws_resources import get_document_context

            self._document_context = get_document_context(
                module=None
                if self.module_context in ("No module selected", "General")
                else self.module_context,
                max_chars=150000,
            )
            if self._document_context:
                logger.info(f"Loaded {len(self._document_context)} chars of document context")
        except Exception as e:
            logger.warning(f"Could not load document context: {e}")
            self._document_context = ""

        return self._document_context

    @staticmethod
    def _build_file_inventory_from_s3() -> str:
        """Build file inventory from S3 listing (not from loaded content).

        This ensures the agent always knows about ALL modules and files,
        even if some document content is truncated due to context limits.
        """
        try:
            from src.utils.aws_resources import list_all_files

            docs = list_all_files()
            if not docs:
                return "[No documents uploaded yet. Upload study materials to get started!]"
            modules = {}
            for doc in docs:
                module_name = doc.get("module", "unknown").replace("_", " ").title()
                modules.setdefault(module_name, []).append(doc["filename"])
            lines = []
            for module, files in sorted(modules.items()):
                lines.append(f"- **{module}**: {', '.join(files)}")
            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"Could not build file inventory from S3: {e}")
            return "[Could not load file inventory]"

    def _initialize_agent(self):
        """Initialize the orchestrator agent with specialist tools, memory, and system prompt."""
        doc_context = self._get_document_context()

        # Set shared context for specialist agent tools
        set_context(doc_context, self.module_context)

        # Build file inventory from S3 listing (independent of loaded content)
        file_inventory = self._build_file_inventory_from_s3()

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            document_context=doc_context
            if doc_context
            else "[No documents uploaded yet. Upload study materials to get started!]",
            file_inventory=file_inventory,
        )

        session_manager = _create_session_manager(self.session_id, self.actor_id)

        agent_kwargs = {
            "model": _get_bedrock_model(),
            "system_prompt": system_prompt,
            "tools": [
                calculator,
                generate_quiz,
                generate_flashcards,
                generate_summary,
                web_search,
                read_url,
            ],
            "callback_handler": None,  # Suppress stdout printing; we stream via SSE
            "conversation_manager": SummarizingConversationManager(
                summary_ratio=0.4,
                preserve_recent_messages=6,
            ),
        }
        if session_manager:
            agent_kwargs["session_manager"] = session_manager

        self.agent = Agent(**agent_kwargs)

    def update_module_context(self, module_context: str):
        """Update the current module context and reinitialize agent."""
        self.module_context = module_context
        self._document_context = ""
        self._initialize_agent()

    def refresh_document_context(self):
        """Refresh document context from S3 (call after uploads)."""
        self._document_context = ""
        self._initialize_agent()

    def chat(self, user_message: str) -> str:
        """Send a message and get a response (non-streaming)."""
        response = self.agent(user_message)
        return str(response)

    def chat_stream_sync(self, user_message: str) -> Generator[dict, None, None]:
        """
        Synchronous streaming - wraps async stream_async for Flask SSE.
        Includes verification agent after the main response completes.
        Emits an 'analytics' event with tools_used, response_time_ms, and verification scores.
        """
        import time

        event_queue = queue.Queue()
        full_response = ""

        async def run_stream():
            nonlocal full_response
            tools_used = []
            start_time = time.monotonic()
            validation = {}
            try:
                async for event in self.agent.stream_async(user_message):
                    if "data" in event:
                        # Use delta text (not accumulated "data") to avoid quadratic growth
                        delta_text = event.get("delta", {}).get("text", "")
                        if delta_text:
                            full_response += delta_text
                            event_queue.put({"type": "text", "content": delta_text})
                    elif "current_tool_use" in event:
                        tool_info = event["current_tool_use"]
                        tool_name = tool_info.get("name", "unknown")
                        if tool_name not in tools_used:
                            tools_used.append(tool_name)
                        event_queue.put({
                            "type": "tool_call",
                            "tool": tool_name,
                            "status": "calling",
                        })

                # Emit structured flashcard data if the tool stored any
                fc_data = get_and_clear_last_flashcard_output()
                if fc_data:
                    event_queue.put({"type": "flashcard_data", **fc_data})

                # Run verification agent after main response completes
                try:
                    if _should_skip_validation(user_message, full_response):
                        validation = {
                            "grounded": "yes",
                            "confidence": "high",
                            "notes": "Non-factual response (greeting, quiz, or flashcard)",
                            "reasoning": "This response type does not require factual verification.",
                        }
                    else:
                        validation = run_verification(
                            user_message, full_response, self._document_context
                        )
                    event_queue.put({"type": "validation", **validation})
                except Exception as e:
                    logger.warning(f"Verification error in stream: {e}")

                elapsed_ms = int((time.monotonic() - start_time) * 1000)

                # Emit analytics metadata for the app layer to log
                event_queue.put({
                    "type": "analytics",
                    "tools_used": tools_used,
                    "response_time_ms": elapsed_ms,
                    "verification_grounded": validation.get("grounded", "N/A"),
                    "verification_confidence": validation.get("confidence", "low"),
                })

                event_queue.put({"type": "done", "content": full_response})
            except Exception as e:
                logger.error(f"Error in async stream: {e}")
                event_queue.put({"type": "error", "content": str(e)})
            finally:
                event_queue.put(None)  # Signal completion

        def run_async_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(run_stream())
            finally:
                loop.close()

        thread = threading.Thread(target=run_async_in_thread)
        thread.start()

        try:
            while True:
                event = event_queue.get(timeout=120)
                if event is None:
                    break
                yield event
        except queue.Empty:
            yield {"type": "error", "content": "Stream timeout"}
        finally:
            thread.join(timeout=5)

    def clear_history(self):
        """Clear conversation by reinitializing the agent."""
        self._initialize_agent()

    def get_history(self) -> list:
        """Get conversation history from the agent's messages."""
        if self.agent and hasattr(self.agent, "messages"):
            return self.agent.messages.copy()
        return []


# =============================================================================
# Convenience function
# =============================================================================
def create_study_agent(
    module: str = "General",
    session_id: str = None,
    actor_id: str = None,
) -> StudyAgentOrchestrator:
    """Create a new StudyAgent orchestrator instance."""
    return StudyAgentOrchestrator(
        module_context=module,
        session_id=session_id,
        actor_id=actor_id,
    )
