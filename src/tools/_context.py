"""
Shared context for specialist agent tools.
Set by the orchestrator before each agent call so tools can access document context.
Also stores last flashcard output so the streaming layer can emit it as structured data.
"""

_context = {"document_context": "", "module": "General"}
_last_flashcard_output = None


def set_context(document_context: str, module: str = "General"):
    """Set shared document context for specialist tools."""
    _context["document_context"] = document_context
    _context["module"] = module


def get_document_context() -> str:
    """Get current document context."""
    return _context["document_context"]


def get_module() -> str:
    """Get current module name."""
    return _context["module"]


def set_last_flashcard_output(data: dict):
    """Store structured flashcard output from the specialist tool."""
    global _last_flashcard_output
    _last_flashcard_output = data


def get_and_clear_last_flashcard_output() -> dict | None:
    """Get and clear stored flashcard output. Returns None if no output stored."""
    global _last_flashcard_output
    result = _last_flashcard_output
    _last_flashcard_output = None
    return result
