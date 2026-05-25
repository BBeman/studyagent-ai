"""
StudyAgent AI - Multi-Agent Tools Module

Specialist agents exposed as Strands @tool functions:
- generate_quiz: Quiz generation agent
- generate_flashcards: Flashcard generation agent
- generate_summary: Summarization agent
- run_verification: Post-response verification agent
"""
from src.tools.quiz import generate_quiz
from src.tools.flashcard import generate_flashcards
from src.tools.summarizer import generate_summary
from src.tools.verification import run_verification
from src.tools._context import set_context, get_document_context
