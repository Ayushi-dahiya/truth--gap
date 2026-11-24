
"""
Unit Tests for TruthGap Calibrated Abstention Gate
==================================================
Tests tri-state decision routing, threshold gating,
and multi-signal confidence.
"""

import sys
from pathlib import Path

# Add project root and src to sys.path
TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

for p in [str(PROJECT_ROOT), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from retrieval.abstention_gate import AbstentionGate


def test_confident_answer_state():
    """Strong retrieval evidence should yield CONFIDENT_ANSWER."""
    gate = AbstentionGate(tau_high=0.54, tau_low=0.44)

    # Maximum RRF consensus score
    max_rrf = 2.0 / 61.0
    decision = gate.evaluate([("DOC001_P1_C1", max_rrf)])

    assert decision.decision == "CONFIDENT_ANSWER"
    assert decision.confidence_score >= 0.54
    assert decision.should_abstain is False
    assert decision.requires_warning is False
    assert decision.disclaimer is None

    print("PASS: test_confident_answer_state")


def test_low_confidence_caution_state():
    """Borderline retrieval evidence should trigger a warning."""
    gate = AbstentionGate(tau_high=0.54, tau_low=0.44)

    # Dense-only RRF score
    # Current implementation produces confidence = 0.45
    dense_only_rrf = 1.0 / 61.0
    decision = gate.evaluate([("DOC002_P4_C2", dense_only_rrf)])

    assert decision.decision == "LOW_CONFIDENCE_CAUTION"
    assert 0.44 <= decision.confidence_score < 0.54
    assert decision.should_abstain is False
    assert decision.requires_warning is True
    assert decision.disclaimer is not None
    assert "Low grounding warning" in decision.disclaimer

    print("PASS: test_low_confidence_caution_state")


def test_corpus_abstention_state():
    """Weak retrieval evidence should trigger CORPUS_ABSTENTION."""
    gate = AbstentionGate(tau_high=0.54, tau_low=0.44)

    # Weak RRF score
    # Current implementation produces confidence = 0.3921
    weak_rrf = 1.0 / 70.0
    decision = gate.evaluate([("DOC004_P80_C1", weak_rrf)])

    assert decision.decision == "CORPUS_ABSTENTION"
    assert decision.confidence_score < 0.44
    assert decision.should_abstain is True
    assert decision.requires_warning is True
    assert decision.disclaimer is not None
    assert "Corpus blind spot" in decision.disclaimer

    print("PASS: test_corpus_abstention_state")


def test_empty_retrieval_abstains():
    """Empty retrieval results should immediately trigger abstention."""
    gate = AbstentionGate(tau_high=0.54, tau_low=0.44)

    decision = gate.evaluate([])

    assert decision.decision == "CORPUS_ABSTENTION"
    assert decision.confidence_score == 0.0
    assert decision.should_abstain is True

    print("PASS: test_empty_retrieval_abstains")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING ABSTENTION GATE TESTS")
    print("=" * 60)

    test_confident_answer_state()
    test_low_confidence_caution_state()
    test_corpus_abstention_state()
    test_empty_retrieval_abstains()

    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
