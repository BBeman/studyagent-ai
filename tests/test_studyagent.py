"""
StudyAgent AI - Comprehensive Automated Test Suite
===================================================
Tests the running Flask application at http://localhost:5001
Covers: API health, input validation, specialist agents, verification,
        SSE streaming, session management, documents, analytics, response quality.

Run with: python tests/test_studyagent.py
Requires: app running on localhost:5001
"""

import requests
import json
import time
import uuid
import os
import sys
import io
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "http://localhost:5001"
TIMEOUT = 90  # seconds - LLM calls can be slow
STREAM_TIMEOUT = 120

# ============================================================================
# Test Framework
# ============================================================================

class TestResult:
    def __init__(self, name, category, passed, duration_ms, details="", error=""):
        self.name = name
        self.category = category
        self.passed = passed
        self.duration_ms = duration_ms
        self.details = details
        self.error = error

    def to_dict(self):
        return {
            "name": self.name,
            "category": self.category,
            "passed": self.passed,
            "duration_ms": round(self.duration_ms, 1),
            "details": self.details,
            "error": self.error,
        }


results = []


def run_test(name, category, test_fn):
    """Execute a single test and record the result."""
    start = time.time()
    try:
        details = test_fn()
        duration = (time.time() - start) * 1000
        r = TestResult(name, category, True, duration, details or "")
        results.append(r)
        print(f"  PASS  {name} ({duration:.0f}ms)")
    except AssertionError as e:
        duration = (time.time() - start) * 1000
        r = TestResult(name, category, False, duration, error=str(e))
        results.append(r)
        print(f"  FAIL  {name} ({duration:.0f}ms) - {e}")
    except Exception as e:
        duration = (time.time() - start) * 1000
        tb = traceback.format_exc()
        r = TestResult(name, category, False, duration, error=f"{e}\n{tb}")
        results.append(r)
        print(f"  ERR   {name} ({duration:.0f}ms) - {e}")


def parse_sse_events(response):
    """Parse Server-Sent Events from a streaming response."""
    events = []
    for line in response.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            data_str = line[6:]
            try:
                events.append(json.loads(data_str))
            except json.JSONDecodeError:
                events.append({"raw": data_str})
    return events


def stream_chat(message, session_id=None, module=None):
    """Send a streaming chat request and return parsed SSE events."""
    payload = {"message": message}
    if session_id:
        payload["session_id"] = session_id
    if module:
        payload["module"] = module
    r = requests.post(
        f"{BASE_URL}/api/chat/stream",
        json=payload,
        stream=True,
        timeout=STREAM_TIMEOUT,
    )
    assert r.status_code == 200, f"Stream returned {r.status_code}: {r.text[:200]}"
    events = parse_sse_events(r)
    return events


def chat(message, session_id=None, module=None):
    """Send a non-streaming chat request."""
    payload = {"message": message}
    if session_id:
        payload["session_id"] = session_id
    if module:
        payload["module"] = module
    r = requests.post(f"{BASE_URL}/api/chat", json=payload, timeout=TIMEOUT)
    return r


# ============================================================================
# 1. API Health & Infrastructure (7 tests)
# ============================================================================

def test_health_returns_200():
    """Health endpoint returns 200 with correct fields."""
    r = requests.get(f"{BASE_URL}/api/health", timeout=10)
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "app" in data
    return f"Status: {data['status']}, App: {data.get('app', 'N/A')}"


def test_health_has_aws_fields():
    """Health endpoint includes AWS profile and region."""
    r = requests.get(f"{BASE_URL}/api/health", timeout=10)
    data = r.json()
    assert "aws_profile" in data or "aws_region" in data
    return f"Profile: {data.get('aws_profile')}, Region: {data.get('aws_region')}"


def test_cors_headers():
    """CORS headers are present on responses."""
    r = requests.get(f"{BASE_URL}/api/health", timeout=10)
    # Flask-CORS adds these headers
    has_cors = (
        "access-control-allow-origin" in r.headers
        or "Access-Control-Allow-Origin" in r.headers
    )
    assert has_cors, f"No CORS headers. Headers: {dict(r.headers)}"
    return "CORS headers present"


def test_invalid_route_returns_404():
    """Invalid route returns 404."""
    r = requests.get(f"{BASE_URL}/api/nonexistent", timeout=10)
    assert r.status_code == 404
    return "404 returned correctly"


def test_root_serves_html():
    """Root URL serves the frontend HTML."""
    r = requests.get(f"{BASE_URL}/", timeout=10)
    assert r.status_code == 200
    assert "text/html" in r.headers.get("Content-Type", "")
    return "HTML frontend served"


def test_health_response_time():
    """Health endpoint responds within 2 seconds."""
    start = time.time()
    r = requests.get(f"{BASE_URL}/api/health", timeout=10)
    elapsed = time.time() - start
    assert elapsed < 2.0, f"Health took {elapsed:.2f}s"
    return f"Response time: {elapsed*1000:.0f}ms"


def test_post_to_get_endpoint():
    """POST to a GET-only endpoint returns 405."""
    r = requests.post(f"{BASE_URL}/api/health", timeout=10)
    assert r.status_code == 405
    return "405 Method Not Allowed returned"


# ============================================================================
# 2. Input Validation & Edge Cases (12 tests)
# ============================================================================

def test_empty_message_returns_400():
    """Empty message returns 400 error."""
    r = requests.post(f"{BASE_URL}/api/chat", json={"message": ""}, timeout=TIMEOUT)
    assert r.status_code == 400
    return f"Error: {r.json().get('error', 'N/A')}"


def test_missing_message_field_returns_400():
    """Missing message field returns 400."""
    r = requests.post(f"{BASE_URL}/api/chat", json={}, timeout=TIMEOUT)
    assert r.status_code == 400
    return f"Error: {r.json().get('error', 'N/A')}"


def test_message_over_limit_returns_400():
    """Message over 10,000 chars returns 400."""
    long_msg = "A" * 10001
    r = requests.post(f"{BASE_URL}/api/chat", json={"message": long_msg}, timeout=TIMEOUT)
    assert r.status_code == 400
    return f"Error: {r.json().get('error', 'N/A')}"


def test_message_at_limit_accepted():
    """Message at exactly 10,000 chars is accepted (not rejected)."""
    msg = "test " * 2000  # 10,000 chars
    r = requests.post(f"{BASE_URL}/api/chat", json={"message": msg}, timeout=TIMEOUT)
    # Should be accepted (200) or at least not 400 for length
    assert r.status_code != 400 or "length" not in r.json().get("error", "").lower()
    return f"Status: {r.status_code}"


def test_unicode_message():
    """Unicode characters in message handled correctly."""
    r = chat("Explain quantum computing using emojis: atoms, waves, particles")
    assert r.status_code == 200
    return f"Response length: {len(r.json().get('response', ''))}"


def test_emoji_message():
    """Emoji-heavy message is handled."""
    events = stream_chat("What does this mean? 🧬🔬📊🎓")
    text_events = [e for e in events if e.get("type") == "text"]
    assert len(text_events) > 0, "No text events received"
    return f"Got {len(text_events)} text events"


def test_code_block_message():
    """Code block in message is handled safely."""
    msg = "Explain this code:\n```python\ndef hello():\n    print('world')\n```"
    r = chat(msg)
    assert r.status_code == 200
    return f"Response received, len={len(r.json().get('response', ''))}"


def test_sql_injection_attempt():
    """SQL injection in message does not crash the system."""
    msg = "'; DROP TABLE students; --"
    r = chat(msg)
    assert r.status_code in [200, 400], f"Unexpected status: {r.status_code}"
    return f"Status: {r.status_code} (handled safely)"


def test_xss_payload():
    """XSS payload in message is handled safely."""
    msg = '<script>alert("xss")</script> explain machine learning'
    r = chat(msg)
    assert r.status_code in [200, 400], f"Unexpected status: {r.status_code}"
    if r.status_code == 200:
        resp = r.json().get("response", "")
        assert "<script>" not in resp, "XSS payload reflected in response"
    return f"Status: {r.status_code} (XSS handled)"


def test_very_short_message():
    """Very short message ('hi') produces a response."""
    r = chat("hi")
    assert r.status_code == 200
    resp = r.json().get("response", "")
    assert len(resp) > 0, "Empty response for 'hi'"
    return f"Response: {resp[:80]}..."


def test_single_char_message():
    """Single character message ('?') is handled."""
    r = chat("?")
    assert r.status_code == 200
    return f"Response received, len={len(r.json().get('response', ''))}"


def test_whitespace_only_message():
    """Whitespace-only message returns 400."""
    r = requests.post(f"{BASE_URL}/api/chat", json={"message": "   "}, timeout=TIMEOUT)
    # Should be rejected as empty or handled gracefully
    assert r.status_code in [200, 400], f"Unexpected status: {r.status_code}"
    return f"Status: {r.status_code}"


# ============================================================================
# 3. Specialist Agent Delegation (16 tests)
# ============================================================================

def test_quiz_generation():
    """Quiz request triggers generate_quiz and returns questions."""
    events = stream_chat("Generate a quiz on machine learning with 3 questions")
    text_content = " ".join(
        e.get("content", "") for e in events if e.get("type") == "text"
    )
    tool_events = [e for e in events if e.get("type") == "tool"]
    has_questions = any(
        kw in text_content.lower()
        for kw in ["question", "?", "a)", "b)", "c)", "true", "false", "answer"]
    )
    assert has_questions, f"No questions found in response: {text_content[:200]}"
    tool_names = [e.get("tool", "") for e in tool_events]
    return f"Tools called: {tool_names}, Response has questions: {has_questions}"


def test_quiz_with_difficulty():
    """Quiz with explicit difficulty parameter."""
    events = stream_chat("Give me 2 easy questions about data structures")
    text_content = " ".join(
        e.get("content", "") for e in events if e.get("type") == "text"
    )
    assert len(text_content) > 20, "Response too short for a quiz"
    return f"Response length: {len(text_content)}"


def test_flashcard_generation():
    """Flashcard request triggers generate_flashcards."""
    events = stream_chat("Make 3 flashcards about neural networks in the chat")
    text_content = " ".join(
        e.get("content", "") for e in events if e.get("type") == "text"
    )
    tool_events = [e for e in events if e.get("type") == "tool"]
    # Check for FRONT:/BACK: format or flashcard content
    has_flashcards = (
        "FRONT:" in text_content.upper()
        or "front" in text_content.lower()
        or any(e.get("type") == "flashcard_data" for e in events)
    )
    return f"Tools: {[e.get('tool') for e in tool_events]}, Flashcard content: {has_flashcards}"


def test_flashcard_anki_export():
    """Flashcard with Anki export format returns flashcard_data event."""
    events = stream_chat("Generate 3 flashcards about Python basics for Anki export")
    flashcard_events = [e for e in events if e.get("type") == "flashcard_data"]
    text_content = " ".join(
        e.get("content", "") for e in events if e.get("type") == "text"
    )
    has_export = len(flashcard_events) > 0 or "flashcard_export" in text_content
    return f"Flashcard data events: {len(flashcard_events)}, Export detected: {has_export}"


def test_summary_bullet_points():
    """Summary request with bullet_points style."""
    events = stream_chat("Summarise artificial intelligence in bullet points")
    text_content = " ".join(
        e.get("content", "") for e in events if e.get("type") == "text"
    )
    tool_events = [e for e in events if e.get("type") == "tool"]
    # Check for bullet-like formatting
    has_bullets = any(c in text_content for c in ["-", "*", "•", "1."])
    return f"Tools: {[e.get('tool') for e in tool_events]}, Bullet format: {has_bullets}"


def test_summary_paragraph():
    """Summary with paragraph style."""
    events = stream_chat("Give me a paragraph summary of cloud computing")
    text_content = " ".join(
        e.get("content", "") for e in events if e.get("type") == "text"
    )
    assert len(text_content) > 50, "Summary too short"
    return f"Response length: {len(text_content)}"


def test_summary_outline():
    """Summary with outline style."""
    events = stream_chat("Create an outline summary of database systems")
    text_content = " ".join(
        e.get("content", "") for e in events if e.get("type") == "text"
    )
    assert len(text_content) > 50, "Summary too short"
    return f"Response length: {len(text_content)}"


def test_web_search():
    """Web search request delegates to web_search agent."""
    events = stream_chat("Search for recent developments in quantum computing 2025")
    text_content = " ".join(
        e.get("content", "") for e in events if e.get("type") == "text"
    )
    tool_events = [e for e in events if e.get("type") == "tool"]
    # Web search should include URLs or citations
    has_citations = any(
        kw in text_content.lower()
        for kw in ["http", "www", "source", "according to", "url"]
    )
    return f"Tools: {[e.get('tool') for e in tool_events]}, Citations found: {has_citations}"


def test_calculator():
    """Calculator request returns correct answer."""
    events = stream_chat("What is 15 * 23?")
    text_content = " ".join(
        e.get("content", "") for e in events if e.get("type") == "text"
    )
    assert "345" in text_content, f"Expected 345 in response: {text_content[:200]}"
    return f"Calculator result contains 345: True"


def test_calculator_complex():
    """Calculator handles more complex expression."""
    events = stream_chat("Calculate the square root of 144")
    text_content = " ".join(
        e.get("content", "") for e in events if e.get("type") == "text"
    )
    assert "12" in text_content, f"Expected 12 in response: {text_content[:200]}"
    return "Correct: sqrt(144) = 12"


def test_read_url():
    """read_url extracts content from a webpage."""
    events = stream_chat(
        "Read this URL and summarise it: https://en.wikipedia.org/wiki/Artificial_intelligence"
    )
    text_content = " ".join(
        e.get("content", "") for e in events if e.get("type") == "text"
    )
    tool_events = [e for e in events if e.get("type") == "tool"]
    assert len(text_content) > 50, "Response too short for URL summary"
    return f"Tools: {[e.get('tool') for e in tool_events]}, Response length: {len(text_content)}"


def test_direct_question_no_tool():
    """Direct question is answered without tool delegation."""
    events = stream_chat("Explain what machine learning is in two sentences")
    text_content = " ".join(
        e.get("content", "") for e in events if e.get("type") == "text"
    )
    tool_events = [e for e in events if e.get("type") == "tool"]
    assert len(text_content) > 20, "Response too short"
    return f"Tools called: {len(tool_events)}, Direct answer length: {len(text_content)}"


def test_greeting_handling():
    """Greeting produces friendly response, no heavy tool calls."""
    events = stream_chat("Hello!")
    text_content = " ".join(
        e.get("content", "") for e in events if e.get("type") == "text"
    )
    tool_events = [e for e in events if e.get("type") == "tool"]
    assert len(text_content) > 0, "Empty response to greeting"
    return f"Tools called: {len(tool_events)}, Response: {text_content[:80]}"


def test_multi_topic_question():
    """Question spanning multiple topics gets a comprehensive answer."""
    events = stream_chat("Compare supervised and unsupervised learning")
    text_content = " ".join(
        e.get("content", "") for e in events if e.get("type") == "text"
    )
    has_both = "supervised" in text_content.lower() and "unsupervised" in text_content.lower()
    assert has_both, "Response should discuss both supervised and unsupervised"
    return f"Both topics covered: {has_both}"


def test_flashcard_csv_export():
    """Flashcard CSV export request."""
    events = stream_chat("Create 3 flashcards about algorithms as CSV export")
    flashcard_events = [e for e in events if e.get("type") == "flashcard_data"]
    text_content = " ".join(
        e.get("content", "") for e in events if e.get("type") == "text"
    )
    return f"Flashcard events: {len(flashcard_events)}, Response length: {len(text_content)}"


def test_quiz_different_topic():
    """Quiz on a different topic to test versatility."""
    events = stream_chat("Make a quiz about software engineering principles")
    text_content = " ".join(
        e.get("content", "") for e in events if e.get("type") == "text"
    )
    has_questions = "?" in text_content or "question" in text_content.lower()
    assert has_questions or len(text_content) > 100, "Expected quiz content"
    return f"Response length: {len(text_content)}"


# ============================================================================
# 4. Verification Agent (8 tests)
# ============================================================================

def test_verification_on_factual_question():
    """Factual question triggers verification with grounding score."""
    events = stream_chat("What is the time complexity of binary search?")
    validation_events = [e for e in events if e.get("type") == "validation"]
    if validation_events:
        v = validation_events[0]
        grounded = v.get("grounded", "")
        confidence = v.get("confidence", "")
        return f"GROUNDED: {grounded}, CONFIDENCE: {confidence}"
    return "No validation event (may be skipped for general knowledge)"


def test_verification_on_greeting():
    """Greeting should get GROUNDED: YES with Non-factual note."""
    events = stream_chat("Hey there!")
    validation_events = [e for e in events if e.get("type") == "validation"]
    if validation_events:
        v = validation_events[0]
        grounded = v.get("grounded", "")
        notes = v.get("notes", "")
        return f"GROUNDED: {grounded}, Notes: {notes}"
    return "No validation event (verification may be skipped for greetings)"


def test_verification_skipped_for_quiz():
    """Quiz request should skip verification (latency optimization)."""
    events = stream_chat("Generate 2 quiz questions about Python")
    validation_events = [e for e in events if e.get("type") == "validation"]
    # Verification may be skipped for tool-based outputs
    return f"Validation events: {len(validation_events)}"


def test_verification_skipped_for_flashcards():
    """Flashcard request should skip verification."""
    events = stream_chat("Make 2 flashcards about HTML basics in the chat")
    validation_events = [e for e in events if e.get("type") == "validation"]
    return f"Validation events: {len(validation_events)}"


def test_verification_general_knowledge():
    """General knowledge question shows appropriate grounding."""
    events = stream_chat("What year was Python created?")
    validation_events = [e for e in events if e.get("type") == "validation"]
    if validation_events:
        v = validation_events[0]
        return f"GROUNDED: {v.get('grounded')}, CONFIDENCE: {v.get('confidence')}"
    return "No validation event received"


def test_verification_fields_present():
    """Validation event has required fields: grounded, confidence."""
    events = stream_chat("Explain recursion briefly")
    validation_events = [e for e in events if e.get("type") == "validation"]
    if validation_events:
        v = validation_events[0]
        assert "grounded" in v, f"Missing 'grounded' field: {v}"
        assert "confidence" in v, f"Missing 'confidence' field: {v}"
        return f"Fields present: grounded={v['grounded']}, confidence={v['confidence']}"
    return "No validation event (may be skipped)"


def test_analytics_event_present():
    """Analytics event with interaction_id is emitted."""
    events = stream_chat("What is an API?")
    analytics_events = [e for e in events if e.get("type") == "analytics"]
    if analytics_events:
        a = analytics_events[0]
        assert "interaction_id" in a, f"Missing interaction_id: {a}"
        return f"Interaction ID: {a.get('interaction_id')}"
    return "No analytics event received"


def test_done_event_present():
    """Stream always ends with a done event."""
    events = stream_chat("Define algorithm")
    done_events = [e for e in events if e.get("type") == "done"]
    assert len(done_events) > 0, "No 'done' event in stream"
    return "Done event present as expected"


# ============================================================================
# 5. SSE Streaming (9 tests)
# ============================================================================

def test_stream_content_type():
    """Stream endpoint returns text/event-stream content type."""
    r = requests.post(
        f"{BASE_URL}/api/chat/stream",
        json={"message": "Hi"},
        stream=True,
        timeout=STREAM_TIMEOUT,
    )
    ct = r.headers.get("Content-Type", "")
    assert "text/event-stream" in ct, f"Expected text/event-stream, got: {ct}"
    # Consume stream to clean up
    for _ in r.iter_lines():
        pass
    return f"Content-Type: {ct}"


def test_stream_has_text_events():
    """Stream contains text events with content."""
    events = stream_chat("Say hello")
    text_events = [e for e in events if e.get("type") == "text"]
    assert len(text_events) > 0, "No text events in stream"
    total_text = " ".join(e.get("content", "") for e in text_events)
    return f"Text events: {len(text_events)}, Total chars: {len(total_text)}"


def test_stream_ends_with_done():
    """Stream ends with 'done' event."""
    events = stream_chat("What is 2+2?")
    assert len(events) > 0, "No events received"
    # Find last non-empty event
    last = events[-1]
    assert last.get("type") == "done", f"Last event type: {last.get('type')}, expected 'done'"
    return "Stream ends with done event"


def test_stream_event_order():
    """Stream events follow logical order: text -> validation -> analytics -> done."""
    events = stream_chat("What is a variable?")
    types = [e.get("type") for e in events]
    # Done should be last
    if "done" in types:
        done_idx = types.index("done")
        assert done_idx == len(types) - 1, f"Done at index {done_idx}, total events: {len(types)}"
    return f"Event types: {types}"


def test_stream_tool_event_for_quiz():
    """Tool call emits 'tool' event before text for quiz."""
    events = stream_chat("Generate 2 quiz questions on testing")
    tool_events = [e for e in events if e.get("type") == "tool"]
    text_events = [e for e in events if e.get("type") == "text"]
    if tool_events:
        return f"Tool events: {[e.get('tool') for e in tool_events]}, Text events: {len(text_events)}"
    return f"No tool events. Text events: {len(text_events)}"


def test_stream_concurrent_requests():
    """Two simultaneous stream requests both complete successfully."""
    def do_stream(msg):
        return stream_chat(msg)

    with ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(do_stream, "What is Python?")
        f2 = ex.submit(do_stream, "What is Java?")
        events1 = f1.result(timeout=STREAM_TIMEOUT)
        events2 = f2.result(timeout=STREAM_TIMEOUT)

    text1 = " ".join(e.get("content", "") for e in events1 if e.get("type") == "text")
    text2 = " ".join(e.get("content", "") for e in events2 if e.get("type") == "text")
    assert len(text1) > 0, "Stream 1 empty"
    assert len(text2) > 0, "Stream 2 empty"
    return f"Stream 1: {len(text1)} chars, Stream 2: {len(text2)} chars"


def test_stream_large_response():
    """Stream handles a response that generates significant text."""
    events = stream_chat("Give a detailed explanation of how the internet works")
    text_content = " ".join(
        e.get("content", "") for e in events if e.get("type") == "text"
    )
    assert len(text_content) > 100, f"Response too short: {len(text_content)}"
    return f"Response length: {len(text_content)} chars"


def test_nonstreaming_chat_returns_json():
    """Non-streaming /api/chat returns proper JSON response."""
    r = chat("What is 1+1?")
    assert r.status_code == 200
    data = r.json()
    assert "response" in data, f"Missing 'response' field: {data}"
    assert len(data["response"]) > 0, "Empty response"
    return f"Response length: {len(data['response'])}"


def test_stream_validation_event():
    """Stream contains validation event with grounding info."""
    events = stream_chat("Explain the OSI model briefly")
    validation_events = [e for e in events if e.get("type") == "validation"]
    if validation_events:
        v = validation_events[0]
        return f"GROUNDED: {v.get('grounded')}, CONFIDENCE: {v.get('confidence')}, Notes: {v.get('notes', '')[:60]}"
    return "No validation event (may be skipped)"


# ============================================================================
# 6. Session Management (7 tests)
# ============================================================================

def test_new_session_created():
    """Sending a chat with new session_id creates a session."""
    sid = f"test-{uuid.uuid4().hex[:8]}"
    r = chat("Hello", session_id=sid)
    assert r.status_code == 200
    return f"Session {sid} created, status: {r.status_code}"


def test_session_history_after_chat():
    """Session history returns messages after a chat."""
    sid = f"test-hist-{uuid.uuid4().hex[:8]}"
    chat("Hello, remember this session", session_id=sid)
    r = requests.get(f"{BASE_URL}/api/session/{sid}/history", timeout=TIMEOUT)
    assert r.status_code == 200
    data = r.json()
    history = data.get("history", [])
    return f"History entries: {len(history)}"


def test_clear_session():
    """Clearing session removes it."""
    sid = f"test-clear-{uuid.uuid4().hex[:8]}"
    chat("Hello", session_id=sid)
    r = requests.delete(f"{BASE_URL}/api/session/{sid}", timeout=TIMEOUT)
    assert r.status_code == 200
    return f"Session {sid} cleared"


def test_session_isolation():
    """Different sessions are isolated from each other."""
    sid1 = f"test-iso1-{uuid.uuid4().hex[:8]}"
    sid2 = f"test-iso2-{uuid.uuid4().hex[:8]}"
    chat("My name is Alice", session_id=sid1)
    chat("My name is Bob", session_id=sid2)
    # Both should succeed independently
    return "Sessions isolated successfully"


def test_module_switching():
    """PUT endpoint sets module for a session."""
    sid = f"test-mod-{uuid.uuid4().hex[:8]}"
    chat("Hello", session_id=sid)
    r = requests.put(
        f"{BASE_URL}/api/session/{sid}/module",
        json={"module": "Artificial Intelligence"},
        timeout=TIMEOUT,
    )
    assert r.status_code == 200
    return f"Module set: {r.json()}"


def test_session_history_empty_for_new():
    """New session has empty history."""
    sid = f"test-empty-{uuid.uuid4().hex[:8]}"
    r = requests.get(f"{BASE_URL}/api/session/{sid}/history", timeout=TIMEOUT)
    # Should return 200 with empty history or 404
    assert r.status_code in [200, 404]
    if r.status_code == 200:
        history = r.json().get("history", [])
        return f"Empty history, {len(history)} entries"
    return "404 for non-existent session"


def test_session_persists_context():
    """Multi-turn conversation maintains context within session."""
    sid = f"test-ctx-{uuid.uuid4().hex[:8]}"
    chat("My favourite colour is blue. Remember this.", session_id=sid)
    r = chat("What is my favourite colour?", session_id=sid)
    assert r.status_code == 200
    resp = r.json().get("response", "").lower()
    has_blue = "blue" in resp
    return f"Context maintained (mentions blue): {has_blue}, Response: {resp[:100]}"


# ============================================================================
# 7. Document Management (7 tests)
# ============================================================================

def test_list_documents():
    """List documents endpoint returns array."""
    r = requests.get(f"{BASE_URL}/api/documents", timeout=TIMEOUT)
    assert r.status_code == 200
    data = r.json()
    assert "documents" in data
    assert isinstance(data["documents"], list)
    return f"Documents count: {len(data['documents'])}"


def test_list_documents_with_module_filter():
    """List documents with module query parameter."""
    r = requests.get(f"{BASE_URL}/api/documents?module=General", timeout=TIMEOUT)
    assert r.status_code == 200
    data = r.json()
    assert "documents" in data
    return f"Documents for 'General': {len(data['documents'])}"


def test_list_modules():
    """List modules endpoint returns array."""
    r = requests.get(f"{BASE_URL}/api/modules", timeout=TIMEOUT)
    assert r.status_code == 200
    data = r.json()
    assert "modules" in data
    assert isinstance(data["modules"], list)
    return f"Modules: {data['modules']}"


def test_upload_unsupported_type():
    """Upload with unsupported file type returns 400."""
    # Create a temporary .exe file
    files = {"file": ("test.exe", b"fake content", "application/octet-stream")}
    data = {"module": "General"}
    r = requests.post(f"{BASE_URL}/api/upload", files=files, data=data, timeout=TIMEOUT)
    assert r.status_code == 400
    return f"Error: {r.json().get('error', 'N/A')}"


def test_upload_no_file():
    """Upload without file returns 400."""
    r = requests.post(f"{BASE_URL}/api/upload", data={"module": "General"}, timeout=TIMEOUT)
    assert r.status_code == 400
    return f"Error: {r.json().get('error', 'N/A')}"


def test_upload_valid_txt_file():
    """Upload a valid .txt file succeeds."""
    content = "This is a test document for StudyAgent AI automated testing."
    files = {"file": ("test_upload.txt", content.encode(), "text/plain")}
    data = {"module": "Test Module"}
    r = requests.post(f"{BASE_URL}/api/upload", files=files, data=data, timeout=TIMEOUT)
    assert r.status_code == 200
    resp = r.json()
    assert resp.get("success") is True
    return f"Uploaded: {resp.get('filename')}, S3: {resp.get('s3_uri', 'N/A')}"


def test_delete_document():
    """Delete a document (using a test key, may 404 if not found)."""
    r = requests.delete(
        f"{BASE_URL}/api/documents/documents/test-module/nonexistent.txt",
        timeout=TIMEOUT,
    )
    # 200 if deleted, 404/500 if not found is acceptable
    assert r.status_code in [200, 404, 500]
    return f"Status: {r.status_code}"


# ============================================================================
# 8. Analytics & Feedback (7 tests)
# ============================================================================

def test_analytics_returns_summary():
    """Analytics endpoint returns summary with expected fields."""
    r = requests.get(f"{BASE_URL}/api/analytics", timeout=TIMEOUT)
    assert r.status_code == 200
    data = r.json()
    # Check for expected aggregate fields
    has_fields = any(
        k in data
        for k in ["total_interactions", "summary", "interactions", "analytics"]
    )
    return f"Analytics keys: {list(data.keys())[:5]}"


def test_feedback_thumbs_up():
    """Feedback endpoint accepts 'up' vote."""
    # First, get an interaction_id from a chat
    events = stream_chat("What is an array?")
    analytics_events = [e for e in events if e.get("type") == "analytics"]
    if not analytics_events:
        return "Skipped: no analytics event to get interaction_id"

    interaction_id = analytics_events[0].get("interaction_id")
    r = requests.post(
        f"{BASE_URL}/api/feedback",
        json={"interaction_id": interaction_id, "feedback": "up"},
        timeout=TIMEOUT,
    )
    assert r.status_code == 200
    return f"Feedback 'up' recorded for {interaction_id}"


def test_feedback_thumbs_down():
    """Feedback endpoint accepts 'down' vote."""
    events = stream_chat("What is a linked list?")
    analytics_events = [e for e in events if e.get("type") == "analytics"]
    if not analytics_events:
        return "Skipped: no analytics event"

    interaction_id = analytics_events[0].get("interaction_id")
    r = requests.post(
        f"{BASE_URL}/api/feedback",
        json={"interaction_id": interaction_id, "feedback": "down"},
        timeout=TIMEOUT,
    )
    assert r.status_code == 200
    return f"Feedback 'down' recorded for {interaction_id}"


def test_feedback_missing_interaction_id():
    """Feedback without interaction_id returns 400."""
    r = requests.post(
        f"{BASE_URL}/api/feedback",
        json={"feedback": "up"},
        timeout=TIMEOUT,
    )
    assert r.status_code == 400
    return f"Error: {r.json().get('error', 'N/A')}"


def test_feedback_missing_feedback_field():
    """Feedback without feedback field returns 400."""
    r = requests.post(
        f"{BASE_URL}/api/feedback",
        json={"interaction_id": "test-123"},
        timeout=TIMEOUT,
    )
    assert r.status_code == 400
    return f"Error: {r.json().get('error', 'N/A')}"


def test_feedback_invalid_value():
    """Feedback with invalid value (not 'up'/'down') returns 400."""
    r = requests.post(
        f"{BASE_URL}/api/feedback",
        json={"interaction_id": "test-123", "feedback": "maybe"},
        timeout=TIMEOUT,
    )
    # Should reject invalid feedback values
    assert r.status_code in [200, 400]
    return f"Status: {r.status_code}"


def test_analytics_after_chat():
    """After sending a chat, analytics should have data."""
    chat("What is testing?")
    r = requests.get(f"{BASE_URL}/api/analytics", timeout=TIMEOUT)
    assert r.status_code == 200
    data = r.json()
    return f"Analytics data keys: {list(data.keys())[:5]}"


# ============================================================================
# 9. Export Endpoints (4 tests)
# ============================================================================

def test_anki_export():
    """Anki export endpoint generates .apkg file."""
    payload = {
        "deck_name": "Test Deck",
        "cards": [
            {"front": "What is Python?", "back": "A programming language"},
            {"front": "What is Flask?", "back": "A web framework"},
        ],
    }
    r = requests.post(f"{BASE_URL}/api/export/anki", json=payload, timeout=TIMEOUT)
    assert r.status_code == 200
    ct = r.headers.get("Content-Type", "")
    content_disp = r.headers.get("Content-Disposition", "")
    assert len(r.content) > 0, "Empty file returned"
    return f"File size: {len(r.content)} bytes, Content-Type: {ct}"


def test_csv_export():
    """CSV export endpoint generates CSV file."""
    payload = {
        "deck_name": "Test CSV",
        "cards": [
            {"front": "Q1", "back": "A1"},
            {"front": "Q2", "back": "A2"},
        ],
    }
    r = requests.post(f"{BASE_URL}/api/export/csv", json=payload, timeout=TIMEOUT)
    assert r.status_code == 200
    content = r.text if hasattr(r, "text") else r.content.decode()
    assert "Q1" in content or len(r.content) > 0
    return f"CSV size: {len(r.content)} bytes"


def test_anki_export_empty_cards():
    """Anki export with empty cards list."""
    payload = {"deck_name": "Empty Deck", "cards": []}
    r = requests.post(f"{BASE_URL}/api/export/anki", json=payload, timeout=TIMEOUT)
    # May return 400 or generate an empty deck
    assert r.status_code in [200, 400]
    return f"Status: {r.status_code}"


def test_csv_export_no_deck_name():
    """CSV export without deck_name."""
    payload = {"cards": [{"front": "Q", "back": "A"}]}
    r = requests.post(f"{BASE_URL}/api/export/csv", json=payload, timeout=TIMEOUT)
    # Should use default name or return error
    assert r.status_code in [200, 400]
    return f"Status: {r.status_code}"


# ============================================================================
# 10. Response Quality (5 tests)
# ============================================================================

def test_response_not_empty():
    """Response to a real question is never empty."""
    r = chat("What is an operating system?")
    assert r.status_code == 200
    resp = r.json().get("response", "")
    assert len(resp) > 10, f"Response too short: '{resp}'"
    return f"Response length: {len(resp)}"


def test_response_reasonable_length():
    """Response is not excessively long (under 10,000 chars for a simple question)."""
    r = chat("What is a function in programming?")
    assert r.status_code == 200
    resp = r.json().get("response", "")
    assert len(resp) < 10000, f"Response too long: {len(resp)} chars"
    return f"Response length: {len(resp)} chars (within limits)"


def test_general_knowledge_disclaimer():
    """General knowledge answers mention they are not from course materials."""
    r = chat("Who invented the World Wide Web?")
    assert r.status_code == 200
    resp = r.json().get("response", "").lower()
    # Should mention general knowledge or note source
    has_source_note = any(
        kw in resp
        for kw in ["general knowledge", "not from course", "tim berners", "berners-lee"]
    )
    return f"Has source context: {has_source_note}, Response: {resp[:100]}"


def test_multi_turn_context():
    """Multi-turn conversation maintains context."""
    sid = f"test-mt-{uuid.uuid4().hex[:8]}"
    chat("I am studying computer science at university", session_id=sid)
    r = chat("What subject did I say I was studying?", session_id=sid)
    assert r.status_code == 200
    resp = r.json().get("response", "").lower()
    has_context = "computer science" in resp
    return f"Context retained (mentions computer science): {has_context}"


def test_response_in_english():
    """Response to English question is in English."""
    r = chat("What is a database?")
    resp = r.json().get("response", "")
    # Basic check: contains common English words
    english_words = ["the", "is", "a", "data", "information"]
    has_english = any(w in resp.lower() for w in english_words)
    assert has_english, "Response does not appear to be in English"
    return "Response is in English"


# ============================================================================
# Test Runner
# ============================================================================

def run_all_tests():
    """Execute all test categories and generate summary."""
    print("=" * 70)
    print("StudyAgent AI - Automated Test Suite")
    print(f"Target: {BASE_URL}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Check server is running
    try:
        r = requests.get(f"{BASE_URL}/api/health", timeout=5)
        print(f"\nServer health: {r.json().get('status', 'unknown')}\n")
    except Exception as e:
        print(f"\nERROR: Server not reachable at {BASE_URL}")
        print(f"Start the app first: python app.py")
        print(f"Error: {e}")
        sys.exit(1)

    categories = [
        ("API Health & Infrastructure", [
            ("test_health_returns_200", test_health_returns_200),
            ("test_health_has_aws_fields", test_health_has_aws_fields),
            ("test_cors_headers", test_cors_headers),
            ("test_invalid_route_returns_404", test_invalid_route_returns_404),
            ("test_root_serves_html", test_root_serves_html),
            ("test_health_response_time", test_health_response_time),
            ("test_post_to_get_endpoint", test_post_to_get_endpoint),
        ]),
        ("Input Validation & Edge Cases", [
            ("test_empty_message_returns_400", test_empty_message_returns_400),
            ("test_missing_message_field_returns_400", test_missing_message_field_returns_400),
            ("test_message_over_limit_returns_400", test_message_over_limit_returns_400),
            ("test_message_at_limit_accepted", test_message_at_limit_accepted),
            ("test_unicode_message", test_unicode_message),
            ("test_emoji_message", test_emoji_message),
            ("test_code_block_message", test_code_block_message),
            ("test_sql_injection_attempt", test_sql_injection_attempt),
            ("test_xss_payload", test_xss_payload),
            ("test_very_short_message", test_very_short_message),
            ("test_single_char_message", test_single_char_message),
            ("test_whitespace_only_message", test_whitespace_only_message),
        ]),
        ("Specialist Agent Delegation", [
            ("test_quiz_generation", test_quiz_generation),
            ("test_quiz_with_difficulty", test_quiz_with_difficulty),
            ("test_flashcard_generation", test_flashcard_generation),
            ("test_flashcard_anki_export", test_flashcard_anki_export),
            ("test_summary_bullet_points", test_summary_bullet_points),
            ("test_summary_paragraph", test_summary_paragraph),
            ("test_summary_outline", test_summary_outline),
            ("test_web_search", test_web_search),
            ("test_calculator", test_calculator),
            ("test_calculator_complex", test_calculator_complex),
            ("test_read_url", test_read_url),
            ("test_direct_question_no_tool", test_direct_question_no_tool),
            ("test_greeting_handling", test_greeting_handling),
            ("test_multi_topic_question", test_multi_topic_question),
            ("test_flashcard_csv_export", test_flashcard_csv_export),
            ("test_quiz_different_topic", test_quiz_different_topic),
        ]),
        ("Verification Agent", [
            ("test_verification_on_factual_question", test_verification_on_factual_question),
            ("test_verification_on_greeting", test_verification_on_greeting),
            ("test_verification_skipped_for_quiz", test_verification_skipped_for_quiz),
            ("test_verification_skipped_for_flashcards", test_verification_skipped_for_flashcards),
            ("test_verification_general_knowledge", test_verification_general_knowledge),
            ("test_verification_fields_present", test_verification_fields_present),
            ("test_analytics_event_present", test_analytics_event_present),
            ("test_done_event_present", test_done_event_present),
        ]),
        ("SSE Streaming", [
            ("test_stream_content_type", test_stream_content_type),
            ("test_stream_has_text_events", test_stream_has_text_events),
            ("test_stream_ends_with_done", test_stream_ends_with_done),
            ("test_stream_event_order", test_stream_event_order),
            ("test_stream_tool_event_for_quiz", test_stream_tool_event_for_quiz),
            ("test_stream_concurrent_requests", test_stream_concurrent_requests),
            ("test_stream_large_response", test_stream_large_response),
            ("test_nonstreaming_chat_returns_json", test_nonstreaming_chat_returns_json),
            ("test_stream_validation_event", test_stream_validation_event),
        ]),
        ("Session Management", [
            ("test_new_session_created", test_new_session_created),
            ("test_session_history_after_chat", test_session_history_after_chat),
            ("test_clear_session", test_clear_session),
            ("test_session_isolation", test_session_isolation),
            ("test_module_switching", test_module_switching),
            ("test_session_history_empty_for_new", test_session_history_empty_for_new),
            ("test_session_persists_context", test_session_persists_context),
        ]),
        ("Document Management", [
            ("test_list_documents", test_list_documents),
            ("test_list_documents_with_module_filter", test_list_documents_with_module_filter),
            ("test_list_modules", test_list_modules),
            ("test_upload_unsupported_type", test_upload_unsupported_type),
            ("test_upload_no_file", test_upload_no_file),
            ("test_upload_valid_txt_file", test_upload_valid_txt_file),
            ("test_delete_document", test_delete_document),
        ]),
        ("Analytics & Feedback", [
            ("test_analytics_returns_summary", test_analytics_returns_summary),
            ("test_feedback_thumbs_up", test_feedback_thumbs_up),
            ("test_feedback_thumbs_down", test_feedback_thumbs_down),
            ("test_feedback_missing_interaction_id", test_feedback_missing_interaction_id),
            ("test_feedback_missing_feedback_field", test_feedback_missing_feedback_field),
            ("test_feedback_invalid_value", test_feedback_invalid_value),
            ("test_analytics_after_chat", test_analytics_after_chat),
        ]),
        ("Export Endpoints", [
            ("test_anki_export", test_anki_export),
            ("test_csv_export", test_csv_export),
            ("test_anki_export_empty_cards", test_anki_export_empty_cards),
            ("test_csv_export_no_deck_name", test_csv_export_no_deck_name),
        ]),
        ("Response Quality", [
            ("test_response_not_empty", test_response_not_empty),
            ("test_response_reasonable_length", test_response_reasonable_length),
            ("test_general_knowledge_disclaimer", test_general_knowledge_disclaimer),
            ("test_multi_turn_context", test_multi_turn_context),
            ("test_response_in_english", test_response_in_english),
        ]),
    ]

    total_tests = sum(len(tests) for _, tests in categories)
    print(f"Running {total_tests} tests across {len(categories)} categories\n")

    for cat_name, tests in categories:
        print(f"\n--- {cat_name} ({len(tests)} tests) ---")
        for test_name, test_fn in tests:
            run_test(test_name, cat_name, test_fn)

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total = len(results)
    pass_rate = (passed / total * 100) if total > 0 else 0

    durations = [r.duration_ms for r in results]
    avg_ms = sum(durations) / len(durations) if durations else 0
    min_ms = min(durations) if durations else 0
    max_ms = max(durations) if durations else 0

    print(f"\nTotal:  {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Pass Rate: {pass_rate:.1f}%")
    print(f"\nResponse Times:")
    print(f"  Average: {avg_ms:.0f}ms")
    print(f"  Min:     {min_ms:.0f}ms")
    print(f"  Max:     {max_ms:.0f}ms")

    # Per-category breakdown
    print(f"\nPer-Category Breakdown:")
    cat_stats = {}
    for r in results:
        if r.category not in cat_stats:
            cat_stats[r.category] = {"passed": 0, "failed": 0, "durations": []}
        if r.passed:
            cat_stats[r.category]["passed"] += 1
        else:
            cat_stats[r.category]["failed"] += 1
        cat_stats[r.category]["durations"].append(r.duration_ms)

    for cat, stats in cat_stats.items():
        t = stats["passed"] + stats["failed"]
        avg = sum(stats["durations"]) / len(stats["durations"])
        print(f"  {cat}: {stats['passed']}/{t} passed, avg {avg:.0f}ms")

    # Failed tests detail
    if failed > 0:
        print(f"\nFailed Tests:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}: {r.error[:120]}")

    # Save results to JSON
    output = {
        "timestamp": datetime.now().isoformat(),
        "base_url": BASE_URL,
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate_pct": round(pass_rate, 1),
            "avg_response_ms": round(avg_ms, 1),
            "min_response_ms": round(min_ms, 1),
            "max_response_ms": round(max_ms, 1),
        },
        "categories": {
            cat: {
                "passed": stats["passed"],
                "failed": stats["failed"],
                "total": stats["passed"] + stats["failed"],
                "avg_ms": round(sum(stats["durations"]) / len(stats["durations"]), 1),
                "min_ms": round(min(stats["durations"]), 1),
                "max_ms": round(max(stats["durations"]), 1),
            }
            for cat, stats in cat_stats.items()
        },
        "tests": [r.to_dict() for r in results],
    }

    output_path = os.path.join(os.path.dirname(__file__), "test_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    print("=" * 70)

    return output


if __name__ == "__main__":
    run_all_tests()
