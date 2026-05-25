"""
Flashcard Agent Tool - Specialist agent for generating flashcards.
Delegates flashcard creation to a focused sub-agent with its own prompt.

Stores structured output in _context so the streaming layer can emit it
as a dedicated SSE event, bypassing orchestrator reformatting.
"""
import json
import logging
import re

from strands import Agent, tool
from strands.models.bedrock import BedrockModel

from src.config import SPECIALIST_MODEL_ID, DEFAULT_TEMPERATURE, AWS_REGION, FLASHCARD_AGENT_PROMPT
from src.tools._context import get_document_context, set_last_flashcard_output

logger = logging.getLogger("flashcard-agent")

_cached_model = None


def _get_model():
    global _cached_model
    if _cached_model is None:
        _cached_model = BedrockModel(model_id=SPECIALIST_MODEL_ID, region_name=AWS_REGION, temperature=DEFAULT_TEMPERATURE)
    return _cached_model


def _parse_flashcard_output(output: str, export_format: str) -> dict | None:
    """Parse specialist output into structured card data.

    Tries multiple formats: FRONT/BACK, JSON export, and common fallback patterns.
    """
    if export_format in ("anki", "csv"):
        # Try JSON export format
        try:
            start = output.find('"flashcard_export"')
            if start != -1:
                brace_start = output.rfind("{", 0, start)
                if brace_start != -1:
                    depth = 0
                    for i in range(brace_start, len(output)):
                        if output[i] == "{":
                            depth += 1
                        elif output[i] == "}":
                            depth -= 1
                            if depth == 0:
                                json_str = output[brace_start : i + 1]
                                parsed = json.loads(json_str)
                                export = parsed.get("flashcard_export", {})
                                if export.get("cards"):
                                    return {
                                        "format": export.get("format", export_format),
                                        "deck_name": export.get("deck_name", "Flashcards"),
                                        "cards": export["cards"],
                                    }
                                break
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    # Try FRONT/BACK chat format (primary)
    cards = []
    for m in re.finditer(
        r"FRONT:\s*([\s\S]*?)\nBACK:\s*([\s\S]*?)(?=\nFRONT:|\s*$)",
        output,
        re.IGNORECASE,
    ):
        cards.append({"front": m.group(1).strip(), "back": m.group(2).strip()})

    if cards:
        return {"format": export_format, "cards": cards}

    # Fallback: Q:/A: or Question:/Answer: patterns
    for m in re.finditer(
        r"(?:^|\n)\s*(?:Q|Question)\s*[:]\s*([\s\S]*?)\n\s*(?:A|Answer)\s*[:]\s*([\s\S]*?)(?=\n\s*(?:Q|Question)\s*[:]|\s*$)",
        output,
        re.IGNORECASE,
    ):
        cards.append({"front": m.group(1).strip(), "back": m.group(2).strip()})

    if cards:
        return {"format": export_format, "cards": cards}

    # Fallback: Numbered "N. **term** - definition" or "N. term\ndefinition"
    for m in re.finditer(
        r"(?:^|\n)\s*\d+\.\s*\*{0,2}(.+?)\*{0,2}\s*[-–-:]\s*([\s\S]*?)(?=\n\s*\d+\.|\s*$)",
        output,
    ):
        front = m.group(1).strip()
        back = m.group(2).strip()
        if front and back and len(front) > 3 and len(back) > 3:
            cards.append({"front": front, "back": back})

    if cards:
        return {"format": export_format, "cards": cards}

    return None


@tool
def generate_flashcards(topic: str, num_cards: int = 10, export_format: str = "chat") -> str:
    """Generate study flashcards from the student's course materials.
    Use this tool whenever a student asks for flashcards, study cards, or revision cards.

    Args:
        topic: The topic or subject area to create flashcards about.
        num_cards: Number of flashcards to generate (default 10).
        export_format: Output format - 'chat' for inline display, 'anki' for Anki deck JSON, 'csv' for CSV JSON (default chat).

    Returns:
        Flashcards in the requested format.
    """
    doc_context = get_document_context()
    materials = doc_context[:50000] if doc_context else "[No course materials available]"

    agent = Agent(
        model=_get_model(),
        system_prompt=FLASHCARD_AGENT_PROMPT,
        callback_handler=None,
    )

    prompt = (
        f"Create {num_cards} flashcards about: {topic}\n"
        f"Output format: {export_format}\n\n"
        f"Source materials:\n{materials}"
    )

    logger.info(f"Flashcard agent: {num_cards} cards on '{topic}' ({export_format})")
    result = str(agent(prompt))

    # Parse and store structured output for the streaming layer
    parsed = _parse_flashcard_output(result, export_format)
    if parsed:
        set_last_flashcard_output(parsed)
        logger.info(f"Stored {len(parsed['cards'])} structured flashcards ({parsed['format']})")

        # Reconstruct as clean FRONT/BACK text for the orchestrator pass-through
        if export_format == "chat":
            result = "\n\n".join(
                f"FRONT: {c['front']}\nBACK: {c['back']}" for c in parsed["cards"]
            )
    else:
        logger.warning("Could not parse structured flashcard output from specialist")

    return result
