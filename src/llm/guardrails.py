"""Security guardrails: input validation, prompt injection defense, and error shielding."""

import re
import unicodedata
from typing import Tuple, List, Dict, Any
from ..schemas import SearchResult
from ..config import sanitize_error

# Known adversarial prompt injection heuristics
INJECTION_REGEXES = [
    re.compile(r"ignore\s+(all\s+)?(previous|above|system)\s+(instructions|prompts)", re.IGNORECASE),
    re.compile(r"system\s*prompt\s*override", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(in\s+developer\s+mode|dan|unrestricted)", re.IGNORECASE),
    re.compile(r"reveal\s+(your\s+|the\s+)?(system\s+prompt|instructions|api\s+key|secret)", re.IGNORECASE),
    re.compile(r"print\s+(your\s+)?(api\s+key|initial\s+instructions|system\s+prompt)", re.IGNORECASE),
    re.compile(r"execute\s+(the\s+following\s+)?(code|command|script|shell)", re.IGNORECASE),
]


def validate_user_question(question: str, max_chars: int = 600) -> Tuple[bool, str, str]:
    """Validates and cleans input user questions.

    Strips control characters, enforces non-emptiness and max length constraints.

    Returns:
        Tuple of (is_valid: bool, cleaned_question: str, error_msg: str).
    """
    if not question or not question.strip():
        return False, "", "Question cannot be empty."

    # Strip non-printable and control characters except newline and tab
    cleaned = "".join(ch for ch in question if ch in "\n\t" or not unicodedata.category(ch).startswith("C"))
    cleaned = cleaned.strip()

    if len(cleaned) < 3:
        return False, "", "Question is too short."

    if len(cleaned) > max_chars:
        return False, "", f"Question exceeds maximum allowed length of {max_chars} characters."

    return True, cleaned, ""


def detect_injection_attempt(text: str) -> Tuple[bool, str]:
    """Detects adversarial instruction overrides or secret extraction attempts."""
    for pattern in INJECTION_REGEXES:
        if pattern.search(text):
            return True, f"Matched injection pattern: '{pattern.pattern}'"
    return False, ""


def sanitize_untrusted_input(paper_content: str) -> Tuple[str, bool]:
    """Neutralizes prompt injection patterns and strips boundary-breaking XML tags.

    Returns:
        Tuple of (sanitized_content: str, was_injection_detected: bool).
    """
    is_injected, _ = detect_injection_attempt(paper_content)

    # Escape XML boundary delimiters
    cleaned = paper_content.replace("</evidence>", "&lt;/evidence&gt;")
    cleaned = cleaned.replace("<evidence", "&lt;evidence")
    cleaned = cleaned.replace("</context>", "&lt;/context&gt;")
    cleaned = cleaned.replace("<system>", "&lt;system&gt;")
    cleaned = cleaned.replace("</system>", "&lt;/system&gt;")

    if is_injected:
        for pattern in INJECTION_REGEXES:
            cleaned = pattern.sub("[POTENTIAL ADVERSARIAL OVERRIDE NEUTRALIZED]", cleaned)

    return cleaned, is_injected


def check_insufficient_evidence(
    results: List[SearchResult],
    min_confidence_score: float = 0.01,
) -> bool:
    """Determines whether retrieved evidence is insufficient before LLM invocation."""
    if not results or len(results) < 1:
        return True
    if max((r.score for r in results), default=0.0) < min_confidence_score:
        return True
    return False


def shield_user_error(err: Exception) -> str:
    """Masks stack traces, prompts, API keys, and local filesystem paths."""
    return sanitize_error(err)
