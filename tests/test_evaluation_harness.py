"""
Unit Tests for TruthGap Evaluation Harness
=========================================
Tests end-to-end evaluation metrics calculation and report generation.
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

from evaluation.evaluate_end_to_end import EndToEndEvaluator


def test_evaluation_run_metrics():
    """Verify that evaluation generates valid metrics and output files."""
    evaluator = EndToEndEvaluator()
    summary = evaluator.run_evaluation()

    assert summary["total_queries"] >= 80
    assert summary["unanswerable_queries"] >= 20

    # TruthGap must significantly outperform baseline in abstention recall
    assert summary["truthgap"]["abstention_recall"] >= 0.90
    assert summary["truthgap"]["hallucination_rate"] <= 0.10
    assert summary["truthgap"]["answerable_accuracy"] >= 0.90

    # Verify reports were created
    csv_file = PROJECT_ROOT / "reports" / "evaluation_results_sem7.csv"
    md_file = PROJECT_ROOT / "reports" / "evaluation_report_sem7.md"
    assert csv_file.exists(), "evaluation_results_sem7.csv was not created!"
    assert md_file.exists(), "evaluation_report_sem7.md was not created!"

    print("PASS: test_evaluation_run_metrics")


if __name__ == "__main__":
    test_evaluation_run_metrics()
    print("ALL EVALUATION TESTS PASSED!")
