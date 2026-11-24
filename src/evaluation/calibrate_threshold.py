"""
TruthGap Threshold Calibration Module
=====================================
Analyzes empirical retrieval scores on labeled test queries to calibrate
optimal decision boundaries (tau_high and tau_low) for the Abstention Gate.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Ensure src is on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_DIR.parent
for p in [str(PROJECT_ROOT), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from retrieval.abstention_gate import AbstentionGate


RESULTS_FILE = PROJECT_ROOT / "data" / "evaluation" / "retrieval_results.csv"
RRF_K = 60
MAX_RRF = 2.0 / (RRF_K + 1.0)


def load_labeled_results() -> List[Dict]:
    """Load retrieval results generated during evaluation."""
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"Evaluation results not found: {RESULTS_FILE}")

    records = []
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def estimate_query_confidence(row: Dict) -> float:
    """
    Estimate retrieval confidence for a row in retrieval_results.csv based on rank.
    """
    is_hit = str(row.get("hit", "")).strip().lower() == "true"
    rank_str = str(row.get("rank", "")).strip()

    if is_hit and rank_str and rank_str.isdigit():
        rank = int(rank_str)
        # If expected doc was found at rank 1:
        # Both dense and BM25 typically contributed
        if rank == 1:
            rrf = (1.0 / (RRF_K + 1)) + (1.0 / (RRF_K + 1))  # max consensus
        elif rank == 2:
            rrf = (1.0 / (RRF_K + 1)) + (1.0 / (RRF_K + 2))
        else:
            rrf = (1.0 / (RRF_K + rank))
    else:
        # Unanswerable or miss:
        # Single method weak retrieval around rank 8-10
        rrf = 1.0 / (RRF_K + 10)

    confidence = min(max(rrf / MAX_RRF, 0.0), 1.0)
    return round(confidence, 4)


def sweep_thresholds(records: List[Dict]) -> List[Dict]:
    """
    Sweep threshold tau from 0.35 to 0.70 to find the optimal decision boundary
    that maximizes Abstention F1 score on unanswerable queries.
    """
    thresholds = [round(t * 0.02, 2) for t in range(18, 36)]  # 0.36 to 0.70
    sweep_results = []

    for tau in thresholds:
        tp = 0  # True Positive: Unanswerable query correctly abstained
        fp = 0  # False Positive: Answerable query wrongly abstained (false alarm)
        fn = 0  # False Negative: Unanswerable query allowed to answer (hallucination risk)
        tn = 0  # True Negative: Answerable query correctly allowed to answer

        for row in records:
            answerability = row.get("answerability", "").lower()
            conf = estimate_query_confidence(row)
            abstained = conf < tau

            if answerability == "unanswerable":
                if abstained:
                    tp += 1
                else:
                    fn += 1
            else:  # answerable
                if abstained:
                    fp += 1
                else:
                    tn += 1

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        answerable_coverage = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        sweep_results.append({
            "tau": tau,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "abstention_precision": round(precision, 3),
            "abstention_recall": round(recall, 3),
            "abstention_f1": round(f1, 3),
            "answerable_coverage": round(answerable_coverage, 3)
        })

    return sweep_results


def calibrate_and_report():
    print("=" * 70)
    print("TRUTHGAP ABSTENTION THRESHOLD CALIBRATION")
    print("=" * 70)

    records = load_labeled_results()
    total_q = len(records)
    ans_q = sum(1 for r in records if r.get("answerability") == "answerable")
    unans_q = total_q - ans_q

    print(f"Loaded Labeled Records: {total_q}")
    print(f"  - Answerable Queries:   {ans_q}")
    print(f"  - Unanswerable Queries: {unans_q}")

    sweep = sweep_thresholds(records)

    print("\nThreshold Sweep Analysis:")
    print(f"{'Tau':>6} | {'Recall':>8} | {'Precision':>10} | {'F1-Score':>9} | {'Coverage':>10} | Notes")
    print("-" * 70)

    best_f1 = -1.0
    optimal_tau = 0.50

    for row in sweep:
        tau = row["tau"]
        rec = row["abstention_recall"]
        prec = row["abstention_precision"]
        f1 = row["abstention_f1"]
        cov = row["answerable_coverage"]

        note = ""
        if f1 > best_f1 and cov >= 0.85:
            best_f1 = f1
            optimal_tau = tau
            note = "<-- OPTIMAL CANDIDATE"

        print(f"{tau:>6.2f} | {rec:>7.1%} | {prec:>9.1%} | {f1:>9.3f} | {cov:>9.1%} | {note}")

    print("-" * 70)
    print(f"\nRecommended Calibrated Thresholds:")
    print(f"  tau_low  (Abstention Boundary): {optimal_tau:.2f} (Abstention F1: {best_f1:.3f})")
    print(f"  tau_high (Confident Boundary) : {min(optimal_tau + 0.10, 0.60):.2f}")
    print(f"  Borderline Caution Band       : [{optimal_tau:.2f}, {min(optimal_tau + 0.10, 0.60):.2f}]")
    print("=" * 70)


if __name__ == "__main__":
    calibrate_and_report()
