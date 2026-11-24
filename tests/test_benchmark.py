"""
Unit Tests for TruthGap Benchmark Dataset
========================================
Tests schema validity, class distributions, and referential integrity.
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

from evaluation.benchmark_manager import BenchmarkManager


def test_benchmark_integrity():
    """Ensure benchmark dataset has at least 80 queries and zero schema errors."""
    manager = BenchmarkManager()
    report = manager.validate_integrity()

    assert report["total_queries"] >= 80, f"Expected at least 80 queries, got {report['total_queries']}"
    assert report["is_valid"] is True, f"Integrity errors found: {report['errors']}"
    assert len(report["errors"]) == 0
    print(f"PASS: test_benchmark_integrity (Total queries: {report['total_queries']})")


def test_class_balance():
    """Ensure both answerable and unanswerable classes are well-represented."""
    manager = BenchmarkManager()
    records = manager.load_benchmark()

    answerable = sum(1 for r in records if r["expected_behavior"] == "ANSWER")
    abstained = sum(1 for r in records if r["expected_behavior"] == "ABSTAIN")

    assert answerable >= 60, f"Expected at least 60 answerable queries, got {answerable}"
    assert abstained >= 30, f"Expected at least 30 abstention/unanswerable queries, got {abstained}"
    print(f"PASS: test_class_balance (Answerable: {answerable}, Abstain Traps: {abstained})")


def test_document_coverage():
    """Ensure all active corpus documents (DOC001 to DOC020) are covered."""
    manager = BenchmarkManager()
    report = manager.validate_integrity()
    doc_dist = report["document_distribution"]

    for i in range(1, 21):
        expected_doc = f"DOC{i:03d}"
        assert expected_doc in doc_dist, f"Document {expected_doc} has no benchmark queries!"
        assert doc_dist[expected_doc] >= 1, f"Document {expected_doc} has zero queries"
    print(f"PASS: All 20 policy documents (DOC001 to DOC020) covered in benchmark!")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING BENCHMARK TESTS")
    print("=" * 60)
    test_benchmark_integrity()
    test_class_balance()
    test_document_coverage()
    print("=" * 60)
    print("ALL BENCHMARK TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
