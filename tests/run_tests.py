"""Self-contained test runner executable via standard python."""

import sys
import tempfile
from pathlib import Path

# Ensure workspace root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tests.test_schemas import test_citation_formatting, test_text_chunk_schema, test_figure_record_schema
from tests.test_guardrails import test_detect_injection_attempt, test_sanitize_untrusted_input, test_check_insufficient_evidence
from tests.test_hybrid_search import test_bm25_search, test_reciprocal_rank_fusion
from tests.test_validator import test_validator_nonexistent_file, test_validator_invalid_header, test_validator_oversized
from tests.test_ingestion import (
    test_caption_regex,
    test_chunk_metadata_completeness,
    test_figure_caption_linking,
    test_index_save_load_round_trip,
)
from tests.test_qa import (
    test_fabricated_citation_removed,
    test_unanswerable_question_returns_insufficient_evidence,
    test_poisoned_chunk_flagged_and_neutralized,
    test_malformed_model_json_handled,
    test_provider_failure_returns_safe_message,
)
from tests.test_multimodal_qa import (
    test_description_generation_and_caching,
    test_storage_separation_and_provenance,
    test_indexing_and_parent_resolution,
    test_vision_qa_arguments_and_uncertainty,
    test_citations_resolve_to_parent_figure_and_drop_fabricated,
)


def run_all():
    print("==================================================")
    print("Running ResearchIQ Complete Test Suite (Phase 2 & Multimodal)...")
    print("==================================================")

    print("[1/7] Schema & Citation Tests...")
    test_citation_formatting()
    test_text_chunk_schema()
    test_figure_record_schema()
    print("  -> Passed.")

    print("[2/7] Prompt Injection & Guardrail Tests...")
    test_detect_injection_attempt()
    test_sanitize_untrusted_input()
    test_check_insufficient_evidence()
    print("  -> Passed.")

    print("[3/7] PDF Validator Tests...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_validator_nonexistent_file(tmp_path)
        test_validator_invalid_header(tmp_path)
        test_validator_oversized(tmp_path)
    print("  -> Passed.")

    print("[4/7] Caption Regex & Metadata Completeness Tests...")
    test_caption_regex()
    test_chunk_metadata_completeness()
    print("  -> Passed.")

    print("[5/7] Figure/Table Extraction & Index Round-Trip Tests...")
    test_figure_caption_linking()
    test_index_save_load_round_trip()
    print("  -> Passed.")

    print("[6/7] Question Answering & Citation Verification Tests (MockProvider)...")
    test_fabricated_citation_removed()
    test_unanswerable_question_returns_insufficient_evidence()
    test_poisoned_chunk_flagged_and_neutralized()
    test_malformed_model_json_handled()
    test_provider_failure_returns_safe_message()
    print("  -> Passed.")

    print("[7/7] Multimodal Vision, Descriptions & Resolution Tests...")
    test_description_generation_and_caching()
    test_storage_separation_and_provenance()
    test_indexing_and_parent_resolution()
    test_vision_qa_arguments_and_uncertainty()
    test_citations_resolve_to_parent_figure_and_drop_fabricated()
    print("  -> Passed.")

    print("==================================================")
    print("ALL TESTS PASSED SUCCESSFULLY! (0 NETWORK CALLS)")
    print("==================================================")


if __name__ == "__main__":
    run_all()
