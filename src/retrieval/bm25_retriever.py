import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


CHUNKS_FILE = Path("data/processed/all_chunks.json")


def tokenize(text):
    """
    Simple tokenizer for BM25.
    """
    return re.findall(r"\b\w+\b", text.lower())


def load_chunks():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def build_bm25(chunks):
    tokenized_documents = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(tokenized_documents)


def search_bm25(query, chunks, bm25, top_k=5):

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    ranked_indices = scores.argsort()[::-1][:top_k]

    results = []

    for index in ranked_indices:
        results.append(
            {
                "chunk": chunks[index],
                "score": float(scores[index])
            }
        )

    return results


if __name__ == "__main__":

    chunks = load_chunks()

    print(f"Loaded chunks: {len(chunks)}")

    bm25 = build_bm25(chunks)

    query = input("\nEnter your question: ")

    results = search_bm25(
        query,
        chunks,
        bm25,
        top_k=5
    )

    print("\n==============================")
    print("BM25 RETRIEVAL RESULTS")
    print("==============================")

    for rank, result in enumerate(results, start=1):

        chunk = result["chunk"]
        score = result["score"]

        print(f"\nRank {rank}")
        print("------------------------------")

        print(
            f"Document: {chunk['title']}"
        )

        print(
            f"Page: {chunk['page']}"
        )

        print(
            f"BM25 Score: {score:.4f}"
        )

        print(
            f"Text:\n{chunk['text'][:500]}"
        )