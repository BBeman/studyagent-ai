"""
URL Reader Agent Tool - Extracts and summarises content from web pages.
Uses Tavily Extract for clean content extraction, then an inner agent summarises it.

Use case: Student pastes a URL (documentation, article, lecture slides online)
and wants the agent to read and explain the content.
"""
import logging

from strands import Agent, tool
from strands.models.bedrock import BedrockModel
from strands_tools import tavily

from src.config import SPECIALIST_MODEL_ID, DEFAULT_TEMPERATURE, AWS_REGION

logger = logging.getLogger("read-url-agent")

_cached_model = None

URL_READER_PROMPT = """You are a specialist content reader for an educational AI system.
You have been given a URL to read and summarise for a university student.

Rules:
- Use tavily_extract to pull the content from the URL in markdown format
- Summarise the key points in a clear, structured, student-friendly format
- Preserve important details: definitions, formulas, code examples, key arguments
- Use headers and bullet points for readability
- Cite the URL as the source: "(Source: [URL])"
- If extraction fails, explain what went wrong honestly
- Do NOT fabricate content - only report what was actually extracted from the page
"""


def _get_model():
    global _cached_model
    if _cached_model is None:
        _cached_model = BedrockModel(
            model_id=SPECIALIST_MODEL_ID,
            region_name=AWS_REGION,
            temperature=DEFAULT_TEMPERATURE,
        )
    return _cached_model


@tool
def read_url(url: str) -> str:
    """Read and extract content from a webpage URL, then summarise it for studying.
    Use this tool when:
    - The student provides a URL and wants you to read or summarise it
    - The student shares a link to documentation, an article, lecture slides, or notes
    - The student says "read this", "summarise this page", or "what does this link say"

    Args:
        url: The webpage URL to read and extract content from.

    Returns:
        A structured summary of the webpage content with the source URL cited.
    """
    agent = Agent(
        model=_get_model(),
        system_prompt=URL_READER_PROMPT,
        tools=[tavily],
        callback_handler=None,
    )

    prompt = (
        f"Extract the content from this URL and summarise it for a university student:\n"
        f"{url}\n\n"
        f"Use tavily_extract with format='markdown' to get the page content, "
        f"then present a clear structured summary."
    )

    logger.info(f"URL reader agent: '{url}'")
    return str(agent(prompt))
