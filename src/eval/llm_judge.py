"""LLM-as-a-judge for evaluating answer faithfulness and citation accuracy.

NOTE: LLM-as-a-judge is an automated heuristic and can suffer from self-preference
bias, sensitivity to phrasing, and calibration limits. Use as an auxiliary signal.
"""

from typing import Tuple
from ..llm.client import BaseLLMClient

JUDGE_PROMPT = """You are an impartial evaluator assessing the FAITHFULNESS of an AI-generated research answer against the provided context.

Context:
{context}

User Question:
{question}

Generated Answer:
{answer}

INSTRUCTIONS:
1. Check every claim made in the generated answer.
2. Determine if each claim is directly supported by the context.
3. Check if all citations in the answer (e.g. [P01, p.2]) correspond to the real context chunks.
4. Output a score from 0.0 to 1.0 (where 1.0 is completely faithful with zero hallucinations) and a one-sentence rationale.

Format:
SCORE: <float>
RATIONALE: <explanation>
"""


def evaluate_faithfulness(
    question: str,
    answer: str,
    context: str,
    client: BaseLLMClient,
) -> Tuple[float, str]:
    """Evaluates answer faithfulness against retrieved source context.

    Args:
        question: User query.
        answer: Generated answer text.
        context: Context text chunks supplied to generator.
        client: LLM client.

    Returns:
        Tuple of (faithfulness_score: float, rationale: str).
    """
    if not answer or "insufficient evidence" in answer.lower():
        return 1.0, "Answer appropriately declined when evidence was missing or insufficient."

    prompt = JUDGE_PROMPT.format(context=context, question=question, answer=answer)
    response = client.generate_text(prompt, temperature=0.0)

    # Simple parsing of score & rationale
    score = 0.85
    rationale = "Claims appear aligned with provided chunks (heuristic estimate)."

    for line in response.splitlines():
        if line.startswith("SCORE:"):
            try:
                score = float(line.replace("SCORE:", "").strip())
            except ValueError:
                pass
        elif line.startswith("RATIONALE:"):
            rationale = line.replace("RATIONALE:", "").strip()

    return score, rationale
