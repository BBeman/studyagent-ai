"""
Web Search Agent Tool - Specialist agent for searching beyond course materials.
Uses Tavily search via Strands built-in tool to find supplementary study material.

Two modes:
- Quick search (default): Single tavily_search with academic domain preference
- Deep research: Multi-round search → extract → synthesise for comprehensive study notes

Academic targeting: Prioritises .edu, .ac.uk, arxiv.org, IEEE, ACM, Springer, etc.
"""
import logging

from strands import Agent, tool
from strands.models.bedrock import BedrockModel
from strands_tools import tavily

from src.config import SPECIALIST_MODEL_ID, AWS_REGION

logger = logging.getLogger("search-agent")

_cached_model = None

# Trusted academic and educational domains for targeted search
ACADEMIC_DOMAINS = [
    "arxiv.org",
    "scholar.google.com",
    "semanticscholar.org",
    "ieee.org",
    "acm.org",
    "springer.com",
    "sciencedirect.com",
    "nature.com",
    "wiley.com",
    "jstor.org",
    "pubmed.ncbi.nlm.nih.gov",
    "researchgate.net",
    "dl.acm.org",
    "proceedings.neurips.cc",
    "openreview.net",
    "mit.edu",
    "stanford.edu",
    "cam.ac.uk",
    "ox.ac.uk",
    "cs.cmu.edu",
    "geeksforgeeks.org",
    "tutorialspoint.com",
    "w3schools.com",
    "docs.python.org",
    "pytorch.org",
    "tensorflow.org",
    "huggingface.co",
]

SEARCH_AGENT_PROMPT = """You are a specialist web search agent for an educational AI system.
Your job is to find high-quality supplementary material for university students.

Rules:
- Search for academic, educational, and authoritative sources
- Summarise findings clearly with source URLs cited
- Focus on explanations, examples, and current information
- Prioritise these source types (in order):
  1. Academic papers (arxiv, IEEE, ACM, Springer, Nature)
  2. University course materials (.edu, .ac.uk)
  3. Official documentation (Python docs, PyTorch, TensorFlow)
  4. High-quality tutorials (GeeksforGeeks, TutorialsPoint)
  5. Educational blogs and articles
- Always cite the URL source for every piece of information: "(Source: [URL])"
- Present information in a structured, student-friendly format
- If search results are poor or irrelevant, say so honestly
- Do NOT fabricate information - only report what the search returns

When using tavily_search:
- Use search_depth="advanced" for better quality results when instructed
- Use include_answer="advanced" to get a pre-synthesised answer alongside results
"""

DEEP_RESEARCH_PROMPT = """You are a deep research agent for an educational AI system.
Your job is to conduct thorough, multi-step research on a topic for a university student.

Research process:
1. Start with a broad tavily_search (search_depth="advanced") to discover key sources
2. Identify the 3-5 most relevant and authoritative URLs from the results
3. Use tavily_extract on those URLs to get full page content in markdown
4. Synthesise all extracted content into a comprehensive research report

Report format:
- Start with a brief overview / executive summary
- Organise findings under clear topic headers
- Include key definitions, formulas, and examples
- Cite every source with its URL: "(Source: [URL])"
- End with a "Further Reading" section listing all source URLs
- Flag any conflicting information between sources

Source quality rules:
- Prioritise academic papers, university sites, and official documentation
- Clearly label the type of each source (paper, tutorial, documentation, blog, etc.)
- If a claim appears in only one source, note it as "single-source claim"
- Do NOT fabricate information - only report what was actually found
"""


def _get_model():
    global _cached_model
    if _cached_model is None:
        _cached_model = BedrockModel(
            model_id=SPECIALIST_MODEL_ID,
            region_name=AWS_REGION,
            temperature=0.2,
        )
    return _cached_model


@tool
def web_search(query: str, num_results: int = 5, deep_research: bool = False) -> str:
    """Search the web for supplementary study material beyond the course documents.
    Use this tool when:
    - The student asks about a topic NOT covered in their course materials
    - The student wants current or updated information beyond their lectures
    - Additional examples, explanations, or perspectives would help the student learn
    - The student explicitly asks to search the web or look something up

    Do NOT use this tool when:
    - The answer is already in the course materials
    - The student is asking about basic concepts well-covered in their lectures
    - The student asks for a quiz, flashcard, or summary (use the specialist tools instead)

    Args:
        query: The search query - be specific and academic (e.g. "backpropagation algorithm step by step explanation")
        num_results: Number of search results to retrieve (default 5, max 10).
        deep_research: Set to true for comprehensive multi-step research that searches, extracts full page content from top sources, and synthesises a detailed report. Use for complex topics or when the student asks for thorough/in-depth research. Default false for quick searches.

    Returns:
        A summarised, cited response with web sources. Deep research returns a comprehensive report.
    """
    num_results = min(max(num_results, 1), 10)

    if deep_research:
        agent = Agent(
            model=_get_model(),
            system_prompt=DEEP_RESEARCH_PROMPT,
            tools=[tavily],
            callback_handler=None,
        )

        prompt = (
            f"Conduct deep research on: {query}\n\n"
            f"Step 1: Search with tavily_search using search_depth='advanced' and "
            f"include_answer='advanced' for up to {num_results} results.\n"
            f"Step 2: Extract full content from the top 3-5 most relevant URLs "
            f"using tavily_extract with format='markdown'.\n"
            f"Step 3: Synthesise everything into a comprehensive research report "
            f"with proper citations.\n\n"
            f"Prioritise academic sources: {', '.join(ACADEMIC_DOMAINS[:10])}"
        )

        logger.info(f"Deep research agent: '{query}' (max {num_results} results)")
    else:
        agent = Agent(
            model=_get_model(),
            system_prompt=SEARCH_AGENT_PROMPT,
            tools=[tavily],
            callback_handler=None,
        )

        prompt = (
            f"Search the web for educational material about: {query}\n"
            f"Use tavily_search with search_depth='advanced', include_answer='advanced', "
            f"and max_results={num_results}.\n"
            f"Summarise the key findings in a clear, structured format with source URLs.\n\n"
            f"Prefer academic sources: {', '.join(ACADEMIC_DOMAINS[:10])}"
        )

        logger.info(f"Search agent: '{query}' (max {num_results} results)")

    try:
        result = str(agent(prompt))
        logger.info(f"Search completed for: '{query}' ({len(result)} chars)")
        return result
    except Exception as e:
        logger.error(f"Web search failed for '{query}': {type(e).__name__}: {e}", exc_info=True)
        return f"Web search failed: {e}. Please try again or rephrase your query."
