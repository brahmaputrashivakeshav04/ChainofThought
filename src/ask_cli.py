"""CLI for asking grounded questions across the ingested research papers.

Usage:
    python src/ask_cli.py --question "How does Deep GraphRAG balance hierarchical retrieval?"
    python src/ask_cli.py --question "What does Figure 1 in P05 show?" --debug
    python src/ask_cli.py --question "What is the CRISPR enzyme folding rate?"
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.llm.qa_engine import answer_question


def main():
    parser = argparse.ArgumentParser(description="ResearchIQ Evidence-Grounded Question Answering CLI")
    parser.add_argument("--question", "-q", type=str, required=True, help="Question to ask across research papers")
    parser.add_argument("--papers", "-p", type=str, default="", help="Comma-separated list of paper IDs to filter (e.g. 'P02,P05')")
    parser.add_argument("--debug", action="store_true", help="Print debug scores and injection detection flags")
    args = parser.parse_args()

    paper_ids = [pid.strip().upper() for pid in args.papers.split(",") if pid.strip()] or None

    print(f"\n[QUERY]: {args.question}")
    if paper_ids:
        print(f"[FILTER]: Restricted to papers: {', '.join(paper_ids)}")
    print("-" * 75)

    result = answer_question(
        question=args.question,
        paper_ids=paper_ids,
        debug=args.debug,
    )

    print(f"\n[ANSWER]:\n{result.answer}\n")

    if result.formatted_citations:
        print(f"[VERIFIED CITATIONS]: {', '.join(result.formatted_citations)}")

    if result.unsupported_citations_dropped:
        print(f"[DROPPED HALLUCINATED CITATIONS]: {', '.join(result.unsupported_citations_dropped)}")

    if result.has_injection_flag:
        print("\n[SECURITY ALERT]: Potential prompt-injection attempt detected in evidence and neutralized.")

    if result.notes:
        print(f"\n[NOTES]: {result.notes}")

    print("\n[RETRIEVED SOURCES EVALUATED]:")
    for idx, s in enumerate(result.sources[:4], start=1):
        c = s.chunk
        elem = f" | {c.element_number}" if c.element_number else ""
        print(f"  {idx}. [{c.paper_id}, p.{c.page}{elem}] Score: {s.score:.4f} ({s.retrieval_method}) | {c.paper_title}")
        print(f"     Preview: {c.text[:120].strip()}...")

    if args.debug and result.debug_info:
        print(f"\n[DEBUG INFO]:\n{result.debug_info}")

    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
