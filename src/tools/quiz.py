"""
Quiz Agent Tool - Specialist agent for generating quizzes.
Delegates quiz generation to a focused sub-agent with its own prompt.
"""
import logging

from strands import Agent, tool
from strands.models.bedrock import BedrockModel

from src.config import SPECIALIST_MODEL_ID, DEFAULT_TEMPERATURE, AWS_REGION, QUIZ_AGENT_PROMPT
from src.tools._context import get_document_context

logger = logging.getLogger("quiz-agent")

_cached_model = None


def _get_model():
    global _cached_model
    if _cached_model is None:
        _cached_model = BedrockModel(model_id=SPECIALIST_MODEL_ID, region_name=AWS_REGION, temperature=DEFAULT_TEMPERATURE)
    return _cached_model


@tool
def generate_quiz(topic: str, num_questions: int = 5, difficulty: str = "medium") -> str:
    """Generate a quiz with varied question types based on the student's course materials.
    Use this tool whenever a student asks for a quiz, test, practice questions, or exam prep.

    Args:
        topic: The topic or subject area to create quiz questions about.
        num_questions: Number of questions to generate (default 5).
        difficulty: Difficulty level - easy, medium, or hard (default medium).

    Returns:
        A formatted quiz with varied question types and an answer key.
    """
    doc_context = get_document_context()
    materials = doc_context[:50000] if doc_context else "[No course materials available]"

    agent = Agent(
        model=_get_model(),
        system_prompt=QUIZ_AGENT_PROMPT,
        callback_handler=None,
    )

    prompt = (
        f"Create a {difficulty} difficulty quiz with {num_questions} questions about: {topic}\n\n"
        f"Source materials:\n{materials}"
    )

    logger.info(f"Quiz agent: {num_questions} questions on '{topic}' ({difficulty})")
    return str(agent(prompt))
