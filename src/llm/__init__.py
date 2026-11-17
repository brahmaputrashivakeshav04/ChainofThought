"""LLM, Multimodal Reasoning, Prompt Guardrails, and Comparison engine."""

from .client import get_llm_client, BaseLLMClient, GeminiClient, MockProvider, LLMProviderClient
from .guardrails import sanitize_untrusted_input, validate_user_question, detect_injection_attempt
from .qa_engine import generate_grounded_answer, answer_question
from .multimodal_qa import analyze_visual_chunk
from .comparison import compare_papers_and_synthesize

__all__ = [
    "get_llm_client",
    "BaseLLMClient",
    "GeminiClient",
    "MockProvider",
    "LLMProviderClient",
    "validate_user_question",
    "sanitize_untrusted_input",
    "detect_injection_attempt",
    "generate_grounded_answer",
    "answer_question",
    "analyze_visual_chunk",
    "compare_papers_and_synthesize",
]
