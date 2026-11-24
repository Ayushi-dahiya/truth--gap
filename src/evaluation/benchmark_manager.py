"""
TruthGap Benchmark Manager & Extensible Dataset Harness
======================================================
Manages, validates, and extends the evaluation benchmark dataset across
any number of institutional policy PDFs.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Ensure src is on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_DIR.parent
for p in [str(PROJECT_ROOT), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

BENCHMARK_FILE = PROJECT_ROOT / "data" / "evaluation" / "benchmark_dataset.csv"
DOCUMENTS_FILE = PROJECT_ROOT / "data" / "metadata" / "documents.csv"


class BenchmarkManager:
    """
    Manages benchmark queries for current and future policy documents.
    Enforces referential integrity with data/metadata/documents.csv.
    """

    def __init__(self, benchmark_path: Path = BENCHMARK_FILE, docs_path: Path = DOCUMENTS_FILE):
        self.benchmark_path = benchmark_path
        self.docs_path = docs_path

    def load_known_documents(self) -> Dict[str, Dict]:
        """Load registered policy documents from metadata CSV."""
        if not self.docs_path.exists():
            return {}

        docs = {}
        with open(self.docs_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                doc_id = row.get("document_id", "").strip()
                if doc_id:
                    docs[doc_id] = row
        return docs

    def load_benchmark(self) -> List[Dict]:
        """Load all benchmark records."""
        if not self.benchmark_path.exists():
            raise FileNotFoundError(f"Benchmark dataset not found: {self.benchmark_path}")

        records = []
        with open(self.benchmark_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
        return records

    def validate_integrity(self) -> Dict:
        """
        Validate referential integrity between benchmark queries and known documents.
        Checks for:
        - Missing required fields
        - Unregistered document IDs
        - Duplicate questions
        - Class balance
        """
        records = self.load_benchmark()
        docs = self.load_known_documents()

        errors = []
        warnings = []
        seen_questions = set()
        class_counts = {}
        doc_counts = {}

        for i, row in enumerate(records, 1):
            qid = row.get("id", f"Row_{i}")
            q = row.get("question", "").strip()
            gt_class = row.get("ground_truth_class", "").strip()
            exp_doc = row.get("expected_document_id", "").strip()
            exp_beh = row.get("expected_behavior", "").strip()

            # 1. Duplicate check
            q_norm = q.lower()
            if q_norm in seen_questions:
                warnings.append(f"[{qid}] Duplicate question text: '{q}'")
            seen_questions.add(q_norm)

            # 2. Document referential check
            if exp_doc and exp_doc != "NONE":
                if exp_doc not in docs:
                    warnings.append(f"[{qid}] Refers to doc '{exp_doc}' which is not in documents.csv yet.")
                doc_counts[exp_doc] = doc_counts.get(exp_doc, 0) + 1
            else:
                doc_counts["NONE (Out-of-Corpus/Trap)"] = doc_counts.get("NONE (Out-of-Corpus/Trap)", 0) + 1

            # 3. Class count
            class_counts[gt_class] = class_counts.get(gt_class, 0) + 1

            # 4. Consistency check
            if gt_class in ["unanswerable_out_of_corpus", "unanswerable_adversarial"] and exp_beh != "ABSTAIN":
                errors.append(f"[{qid}] Unanswerable class must have expected_behavior = 'ABSTAIN', found '{exp_beh}'")

            if gt_class == "answerable" and exp_beh != "ANSWER":
                errors.append(f"[{qid}] Answerable class must have expected_behavior = 'ANSWER', found '{exp_beh}'")

        return {
            "total_queries": len(records),
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "class_distribution": class_counts,
            "document_distribution": doc_counts
        }

    def add_queries_for_new_policy(
        self,
        new_doc_id: str,
        new_doc_title: str,
        category: str,
        new_questions: List[Dict]
    ) -> int:
        """
        Extensible interface: Append new policy questions when user supplies additional PDFs.
        
        Args:
            new_doc_id: e.g., 'DOC005'
            new_doc_title: e.g., 'National Apprenticeship Promotion Scheme 2024'
            category: e.g., 'Apprenticeship & Skilling'
            new_questions: List of dicts with:
                - 'question': str
                - 'ground_truth_class': 'answerable', 'unanswerable_out_of_corpus', etc.
                - 'expected_behavior': 'ANSWER' or 'ABSTAIN'
                - 'ground_truth_summary': str
        """
        records = self.load_benchmark()
        existing_ids = {r["id"] for r in records}

        # Calculate next numeric ID
        max_num = 0
        for qid in existing_ids:
            if qid.startswith("Q") and qid[1:].isdigit():
                max_num = max(max_num, int(qid[1:]))

        added_count = 0
        fieldnames = [
            "id", "category", "question", "expected_document_id",
            "expected_document_title", "ground_truth_class",
            "expected_behavior", "ground_truth_summary"
        ]

        with open(self.benchmark_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            for item in new_questions:
                max_num += 1
                new_id = f"Q{max_num:03d}"
                row = {
                    "id": new_id,
                    "category": category,
                    "question": item["question"].strip(),
                    "expected_document_id": new_doc_id,
                    "expected_document_title": new_doc_title,
                    "ground_truth_class": item.get("ground_truth_class", "answerable"),
                    "expected_behavior": item.get("expected_behavior", "ANSWER"),
                    "ground_truth_summary": item.get("ground_truth_summary", "")
                }
                writer.writerow(row)
                added_count += 1

        print(f"Successfully appended {added_count} new queries for {new_doc_id} to benchmark dataset.")
        return added_count

    def print_summary(self):
        """Display formatted breakdown of the benchmark."""
        report = self.validate_integrity()
        print("=" * 65)
        print("TRUTHGAP BENCHMARK DATASET SUMMARY")
        print("=" * 65)
        print(f"Total Benchmark Queries: {report['total_queries']}")
        print(f"Integrity Status       : {'VALID' if report['is_valid'] else 'INVALID'}")

        print("\n--- Breakdown by Ground Truth Class ---")
        for cls_name, count in sorted(report["class_distribution"].items()):
            pct = (count / report["total_queries"]) * 100
            print(f"  • {cls_name:<30} : {count:>3} ({pct:>5.1f}%)")

        print("\n--- Breakdown by Expected Document ---")
        for doc_name, count in sorted(report["document_distribution"].items()):
            pct = (count / report["total_queries"]) * 100
            print(f"  • {doc_name:<30} : {count:>3} ({pct:>5.1f}%)")

        if report["errors"]:
            print("\nErrors Found:")
            for err in report["errors"]:
                print(f"  ❌ {err}")

        if report["warnings"]:
            print(f"\nWarnings: {len(report['warnings'])}")
            for w in report["warnings"][:3]:
                print(f"  ⚠️ {w}")
        print("=" * 65)


if __name__ == "__main__":
    manager = BenchmarkManager()
    manager.print_summary()
