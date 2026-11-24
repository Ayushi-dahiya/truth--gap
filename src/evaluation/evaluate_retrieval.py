import csv
import json
import re
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

CHUNKS_FILE = Path("data/processed/all_chunks.json")
TEST_FILE = Path("data/evaluation/retrieval_test.csv")

CHROMA_DIR = "data/processed/chroma_db"
COLLECTION_NAME = "truthgap_policies"

MODEL_NAME = "all-MiniLM-L6-v2"

RRF_K = 60
TOP_K = 10


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


# ============================================================
# LOAD CORPUS
# ============================================================

def load_chunks():

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# BUILD BM25
# ============================================================

def build_bm25(chunks):

    tokenized_documents = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(tokenized_documents)


# ============================================================
# DENSE RETRIEVAL
# ============================================================

def dense_search(
    query,
    model,
    collection,
    top_k=TOP_K
):

    query_embedding = model.encode([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    dense_results = []

    for rank, chunk_id in enumerate(
        results["ids"][0],
        start=1
    ):

        dense_results.append({
            "chunk_id": chunk_id,
            "rank": rank
        })

    return dense_results


# ============================================================
# BM25 RETRIEVAL
# ============================================================

def bm25_search(
    query,
    chunks,
    bm25,
    top_k=TOP_K
):

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    ranked_indices = scores.argsort()[::-1][:top_k]

    results = []

    for rank, index in enumerate(
        ranked_indices,
        start=1
    ):

        results.append({
            "chunk_id": chunks[index]["chunk_id"],
            "rank": rank
        })

    return results


# ============================================================
# RRF
# ============================================================

def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    k=RRF_K
):

    scores = {}

    for result in dense_results:

        chunk_id = result["chunk_id"]

        scores.setdefault(chunk_id, 0)

        scores[chunk_id] += (
            1 / (k + result["rank"])
        )

    for result in bm25_results:

        chunk_id = result["chunk_id"]

        scores.setdefault(chunk_id, 0)

        scores[chunk_id] += (
            1 / (k + result["rank"])
        )

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked


# ============================================================
# EVALUATE ONE QUESTION
# ============================================================

def evaluate_question(
    question,
    expected_document_id,
    chunks,
    model,
    collection,
    bm25
):

    dense_results = dense_search(
        question,
        model,
        collection
    )

    bm25_results = bm25_search(
        question,
        chunks,
        bm25
    )

    hybrid_results = reciprocal_rank_fusion(
        dense_results,
        bm25_results
    )

    chunk_lookup = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }

    top_5 = hybrid_results[:5]

    retrieved_document_ids = []

    for chunk_id, score in top_5:

        document_id = chunk_lookup[
            chunk_id
        ]["document_id"]

        retrieved_document_ids.append(
            document_id
        )

    # --------------------------------------------------------
    # Check whether expected document appears in Top 5
    # --------------------------------------------------------

    if expected_document_id:

        hit = (
            expected_document_id
            in retrieved_document_ids
        )

        if hit:

            rank = (
                retrieved_document_ids.index(
                    expected_document_id
                ) + 1
            )

        else:

            rank = None

    else:

        # For unanswerable questions,
        # retrieval should not be considered
        # a successful answer merely because
        # something was retrieved.
        hit = False
        rank = None

    return {
        "question": question,
        "expected_document_id": expected_document_id,
        "retrieved_document_ids": retrieved_document_ids,
        "hit": hit,
        "rank": rank
    }


# ============================================================
# MAIN EVALUATION
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TRUTHGAP RETRIEVAL EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load corpus
    # --------------------------------------------------------

    print("\nLoading corpus...")

    chunks = load_chunks()

    print(
        f"Loaded chunks: {len(chunks)}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("Embedding model loaded.")

    # --------------------------------------------------------
    # Connect to ChromaDB
    # --------------------------------------------------------

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    print(
        f"ChromaDB vectors: "
        f"{collection.count()}"
    )

    # --------------------------------------------------------
    # Build BM25
    # --------------------------------------------------------

    print("\nBuilding BM25 index...")

    bm25 = build_bm25(chunks)

    print("BM25 ready.")

    # --------------------------------------------------------
    # Load test questions
    # --------------------------------------------------------

    print("\nLoading evaluation dataset...")

    with open(
        TEST_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        test_questions = list(
            csv.DictReader(file)
        )

    print(
        f"Test questions: "
        f"{len(test_questions)}"
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    results = []

    answerable_total = 0
    answerable_hits = 0

    print("\n")
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for index, row in enumerate(
        test_questions,
        start=1
    ):

        question = row["question"].strip()

        expected_document_id = (
            row["expected_document_id"].strip()
        )

        answerability = (
            row["answerability"].strip()
        )

        result = evaluate_question(
            question,
            expected_document_id,
            chunks,
            model,
            collection,
            bm25
        )

        result["answerability"] = answerability

        results.append(result)

        print(
            f"\n[{index}/{len(test_questions)}] "
            f"{question}"
        )

        print(
            f"Expected: "
            f"{expected_document_id or 'NONE'}"
        )

        print(
            f"Retrieved Top-5: "
            f"{result['retrieved_document_ids']}"
        )

        if answerability == "answerable":

            answerable_total += 1

            if result["hit"]:

                answerable_hits += 1

                print(
                    f"Result: HIT "
                    f"(Rank {result['rank']})"
                )

            else:

                print("Result: MISS")

        else:

            print(
                "Result: UNANSWERABLE TEST"
            )

    # ========================================================
    # METRICS
    # ========================================================

    if answerable_total > 0:

        recall_at_5 = (
            answerable_hits /
            answerable_total
        )

    else:

        recall_at_5 = 0.0

    print("\n")
    print("=" * 60)
    print("FINAL RETRIEVAL METRICS")
    print("=" * 60)

    print(
        f"Answerable Questions: "
        f"{answerable_total}"
    )

    print(
        f"Correct Top-5 Retrievals: "
        f"{answerable_hits}"
    )

    print(
        f"Recall@5: "
        f"{recall_at_5:.4f}"
    )

    print(
        f"Recall@5 (%): "
        f"{recall_at_5 * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Save evaluation results
    # --------------------------------------------------------

    output_file = Path(
        "data/evaluation/retrieval_results.csv"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "question",
                "expected_document_id",
                "retrieved_document_ids",
                "hit",
                "rank",
                "answerability"
            ]
        )

        writer.writeheader()

        for result in results:

            writer.writerow({
                "question":
                    result["question"],

                "expected_document_id":
                    result[
                        "expected_document_id"
                    ],

                "retrieved_document_ids":
                    "|".join(
                        result[
                            "retrieved_document_ids"
                        ]
                    ),

                "hit":
                    result["hit"],

                "rank":
                    result["rank"],

                "answerability":
                    result["answerability"]
            })

    print(
        f"\nDetailed results saved to:"
        f"\n{output_file}"
    )

    print("\nEvaluation completed.")