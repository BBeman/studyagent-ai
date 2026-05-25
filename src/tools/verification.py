"""
Verification Agent - Validates responses against source materials.
Runs as a post-processing step with full reasoning capture.

This is a dedicated agent (not a @tool) because it needs to inspect the main
agent's complete output. Running it post-response keeps the student-facing
response clean while providing transparent verification.

Search-augmented verification (three triggers):
1. Claims grounded in source materials → verified against documents only (no search)
2. Claims marked as "general knowledge" → verified via quick Tavily web search
3. Responses containing web URLs (from search/deep research) → verified via quick Tavily search
4. Greetings, quizzes, flashcards → skipped entirely (no verification needed)
"""
import logging
import re

from strands import Agent
from strands.models.bedrock import BedrockModel
from strands_tools import tavily

from src.config import VERIFICATION_MODEL_ID, AWS_REGION, VALIDATION_PROMPT

logger = logging.getLogger("verification")

_cached_model = None

# Patterns that indicate general knowledge claims needing web verification
_GENERAL_KNOWLEDGE_PATTERN = re.compile(
    r"(?i)(general knowledge|not from course materials|"
    r"not in the provided materials|beyond the course materials|"
    r"from my training|commonly known|widely accepted)",
)

# Patterns that indicate the response came from web search / deep research
_WEB_SOURCE_PATTERN = re.compile(
    r"(https?://[^\s\)]+|"
    r"\(Source:\s*http|"
    r"Further Reading|"
    r"web search results|"
    r"according to .{0,30}(arxiv|ieee|acm|springer|nature|wiley))",
    re.IGNORECASE,
)

SEARCH_VALIDATION_PROMPT = """You are a fact-checking agent with web search capability.
Your job is to verify factual claims from an AI response by cross-referencing with web sources.

Rules:
- Identify the 1-3 most important factual claims in the response
- Use ONE focused tavily_search query (search_depth="basic") to verify those claims
- Target academic and authoritative sources for verification
- Report whether each key claim is SUPPORTED, PARTIALLY SUPPORTED, or CONTRADICTED
- Cite the verification source URL
- Be concise - brief verification, not a full report

Respond with EXACTLY this format:
WEB_VERIFIED: [SUPPORTED/PARTIALLY SUPPORTED/CONTRADICTED]
CLAIMS_CHECKED: [Number of claims verified]
SOURCE: [Primary verification URL]
DETAIL: [One or two sentence summary of verification findings]
"""


def _get_model():
    global _cached_model
    if _cached_model is None:
        _cached_model = BedrockModel(
            model_id=VERIFICATION_MODEL_ID,
            region_name=AWS_REGION,
            temperature=0.1,
        )
    return _cached_model


def _parse_result(result: str) -> dict:
    """Parse verification agent output into structured dict."""
    grounded = "N/A"
    confidence = "low"
    notes = ""

    for line in result.split("\n"):
        line = line.strip()
        if line.upper().startswith("GROUNDED:"):
            val = line.split(":", 1)[1].strip().upper()
            if val in ("YES", "PARTIAL", "NO", "N/A"):
                grounded = val.lower() if val != "N/A" else "N/A"
        elif line.upper().startswith("CONFIDENCE:"):
            val = line.split(":", 1)[1].strip().upper()
            if val in ("HIGH", "MEDIUM", "LOW"):
                confidence = val.lower()
        elif line.upper().startswith("NOTES:"):
            notes = line.split(":", 1)[1].strip()

    return {"grounded": grounded, "confidence": confidence, "notes": notes}


def _needs_web_verification(response: str) -> bool:
    """Check if the response contains claims that need web verification.

    Triggers on:
    - General knowledge claims (explicitly stated)
    - Web-sourced content (URLs present - from search/deep research tools)
    """
    if _GENERAL_KNOWLEDGE_PATTERN.search(response):
        return True
    if _WEB_SOURCE_PATTERN.search(response):
        return True
    return False


def _run_web_verification(response: str) -> str:
    """Run quick web search verification on claims from general knowledge or web search."""
    try:
        agent = Agent(
            model=_get_model(),
            system_prompt=SEARCH_VALIDATION_PROMPT,
            tools=[tavily],
            callback_handler=None,
        )

        prompt = (
            f"Verify the key factual claims in the following response. "
            f"Use a single quick tavily_search (search_depth='basic') to cross-check "
            f"the most important claims against authoritative sources.\n\n"
            f"Response to verify:\n{response[:5000]}"
        )

        result = str(agent(prompt))
        logger.info(f"Web verification result: {result[:200]}")
        return result
    except Exception as e:
        logger.warning(f"Web verification failed: {e}")
        return ""


def run_verification(question: str, response: str, document_context: str) -> dict:
    """
    Run verification agent and capture its full reasoning.
    Uses web search to verify claims from general knowledge OR web-sourced responses.

    Returns dict with grounded, confidence, notes, reasoning, and optionally web_verification.
    """
    if not document_context or not document_context.strip():
        return {
            "grounded": "N/A",
            "confidence": "low",
            "notes": "No source materials to validate against",
            "reasoning": "No source materials were provided, so verification cannot be performed.",
        }

    try:
        validator = Agent(
            model=_get_model(),
            system_prompt=VALIDATION_PROMPT,
            callback_handler=None,
        )

        check_prompt = (
            f"Question: {question}\n\n"
            f"Response to validate:\n{response}\n\n"
            f"Source materials:\n{document_context[:30000]}"
        )

        result = str(validator(check_prompt))
        parsed = _parse_result(result)
        parsed["reasoning"] = result.strip()

        # If response contains general knowledge claims OR web-sourced content,
        # run a quick web verification search to cross-check key claims
        if _needs_web_verification(response):
            trigger = (
                "web-sourced content"
                if _WEB_SOURCE_PATTERN.search(response)
                else "general knowledge claims"
            )
            logger.info(f"Detected {trigger} - running web verification")
            web_result = _run_web_verification(response)
            if web_result:
                parsed["web_verification"] = web_result.strip()

        logger.info(f"Verification: {parsed['grounded']} ({parsed['confidence']})")
        return parsed

    except Exception as e:
        logger.warning(f"Verification failed: {e}")
        return {
            "grounded": "N/A",
            "confidence": "low",
            "notes": f"Verification error: {str(e)[:100]}",
            "reasoning": f"Verification agent encountered an error: {str(e)[:200]}",
        }
