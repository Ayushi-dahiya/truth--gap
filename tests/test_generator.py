"""
Unit Tests for TruthGap Grounded Generator
=========================================
Tests grounded answer synthesis, citation validation, hallucination detection,
and explicit abstention behavior.
"""

import sys
from pathlib import Path

# Ensure both project root and src are on sys.path
TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

for p in [str(PROJECT_ROOT), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from src.generation.generator import GroundedGenerator
except ImportError:
    from generation.generator import GroundedGenerator


def test_grounded_answer_with_valid_citations():
    """Verify that in-corpus queries produce grounded answers with valid citations."""
    generator = GroundedGenerator(preferred_provider="mock")

    sample_chunks = [
        {
            "chunk_id": "DOC002_P3_C1",
            "document_id": "DOC002",
            "title": "PMKVY 4.0 Guidelines",
            "page": 3,
            "year": 2023,
            "text": "The objective of PMKVY 4.0 is to encourage and promote skill development for the youth across India through standardized certification.",
        }
    ]

    res = generator.generate("What is the objective of PMKVY 4.0?", sample_chunks)

    assert res["status"] == "ANSWERED", f"Expected ANSWERED, got {res['status']}"
    assert res["is_abstention"] is False
    assert len(res["valid_citations"]) > 0, "Expected at least 1 valid citation"
    assert len(res["hallucinated_citations"]) == 0, "Expected 0 hallucinated citations"
    assert "PMKVY 4.0 Guidelines" in res["valid_citations"][0]
    print("PASS: test_grounded_answer_with_valid_citations")


def test_abstention_on_unanswerable_query():
    """Verify that out-of-corpus queries trigger explicit abstention."""
    generator = GroundedGenerator(preferred_provider="mock")

    sample_chunks = [
        {
            "chunk_id": "DOC001_P1_C1",
            "document_id": "DOC001",
            "title": "Draft National Youth Policy 2025",
            "page": 1,
            "year": 2025,
            "text": "The vision of National Youth Policy 2025 is to unlock the potential of India's youth for nation building.",
        }
    ]

    # Query with missing distinctive concepts (Japan)
    res = generator.generate("What are the objectives of the Youth Policy of Japan?", sample_chunks)

    assert res["is_abstention"] is True, "Expected system to abstain on ungrounded query"
    assert res["status"] == "ABSTAINED"
    assert "does not contain sufficient grounded information" in res["answer"]
    print("PASS: test_abstention_on_unanswerable_query")


def test_force_abstain_flag():
    """Verify that force_abstain flag (from Abstention Gate) immediately triggers abstention."""
    generator = GroundedGenerator(preferred_provider="mock")

    sample_chunks = [{"title": "Doc", "page": 1, "text": "Some text"}]
    res = generator.generate("Any question", sample_chunks, force_abstain=True)

    assert res["is_abstention"] is True
    assert res["status"] == "CORPUS_ABSTAINED"
    print("PASS: test_force_abstain_flag")


def test_citation_hallucination_detection():
    """Verify that citations to documents not in context are caught as hallucinated."""
    generator = GroundedGenerator(preferred_provider="mock")

    context_chunks = [
        {
            "title": "Draft National Youth Policy 2025",
            "document_id": "DOC001",
            "page": 5,
            "text": "Youth leadership programs are organized annually.",
        }
    ]

    # Synthesize an answer that falsely cites a different document and page
    fake_answer = (
        "The scholarship provides 10000 rupees [Random Policy Guidelines, Page 99], "
        "and youth programs run yearly [Draft National Youth Policy 2025, Page 5]."
    )

    valid, hallucinated, ratio = generator.extract_and_verify_citations(
        fake_answer, context_chunks
    )

    assert len(valid) == 1
    assert "[Draft National Youth Policy 2025, Page 5]" in valid[0]
    assert len(hallucinated) == 1
    assert "[Random Policy Guidelines, Page 99]" in hallucinated[0]
    assert ratio == 0.5
    print("PASS: test_citation_hallucination_detection")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING GENERATOR TESTS")
    print("=" * 60)
    test_grounded_answer_with_valid_citations()
    test_abstention_on_unanswerable_query()
    test_force_abstain_flag()
    test_citation_hallucination_detection()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
