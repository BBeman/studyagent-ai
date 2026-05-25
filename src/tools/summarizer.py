"""
Summarizer Agent Tool - Specialist agent for creating summaries.
Delegates summarization to a focused sub-agent with its own prompt.
"""
import logging

from strands import Agent, tool
from strands.models.bedrock import BedrockModel

from src.config import SPECIALIST_MODEL_ID, DEFAULT_TEMPERATURE, AWS_REGION, SUMMARIZER_AGENT_PROMPT
from src.tools._context import get_document_context

logger = logging.getLogger("summarizer-agent")

_cached_model = None


def _get_model():
    global _cached_model
    if _cached_model is None:
        _cached_model = BedrockModel(model_id=SPECIALIST_MODEL_ID, region_name=AWS_REGION, temperature=DEFAULT_TEMPERATURE)
    return _cached_model


@tool
def generate_summary(topic: str, style: str = "bullet_points") -> str:
    """Create a concise summary of course material on a given topic.
    Use this tool whenever a student asks for a summary, overview, key points, or revision notes.

    Args:
        topic: The topic or section to summarize.
        style: Summary style - 'bullet_points', 'paragraph', or 'outline' (default bullet_points).

    Returns:
        A structured summary with source citations.
    """
    doc_context = get_document_context()
    materials = doc_context[:50000] if doc_context else "[No course materials available]"

    agent = Agent(
        model=_get_model(),
        system_prompt=SUMMARIZER_AGENT_PROMPT,
        callback_handler=None,
    )

    prompt = (
        f"Summarize the following topic: {topic}\n"
        f"Style: {style}\n\n"
        f"Source materials:\n{materials}"
    )

    logger.info(f"Summarizer agent: '{topic}' ({style})")
    return str(agent(prompt))
