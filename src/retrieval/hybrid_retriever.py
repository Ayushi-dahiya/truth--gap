
"""
TruthGap Hybrid Retriever
Combines Dense Retrieval, BM25, Reciprocal Rank Fusion,
and Adjacent-Chunk Expansion.
"""

import json
import re
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "all_chunks.json"
CHROMA_DIR = str(PROJECT_ROOT / "data" / "processed" / "chroma_db")

COLLECTION_NAME = "truthgap_policies"
MODEL_NAME = "all-MiniLM-L6-v2"

RRF_K = 60
TOP_K = 10

# Adjacent chunk expansion settings
EXPANSION_SEED_K = 3
NEIGHBOR_WINDOW = 2
MAX_CONTEXT_CHUNKS = 15


# ------------------------------------------------------------
# TOKENIZATION AND DATA LOADING
# ------------------------------------------------------------

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def load_chunks():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_FILE}"
        )

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not chunks:
        raise ValueError("The chunks file is empty.")

    return chunks


# ------------------------------------------------------------
# BM25
# ------------------------------------------------------------

def build_bm25(chunks):
    tokenized_documents = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(tokenized_documents)


def bm25_search(query, chunks, bm25, top_k=TOP_K):
    scores = bm25.get_scores(tokenize(query))

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for index in ranked_indices:
        score = float(scores[index])

        if score <= 0:
            continue

        results.append({
            "chunk_id": chunks[index]["chunk_id"],
            "rank": len(results) + 1,
            "score": score
        })

        if len(results) >= top_k:
            break

    return results


# ------------------------------------------------------------
# DENSE RETRIEVAL
# ------------------------------------------------------------

def dense_search(query, model, collection, top_k=TOP_K):
    count = collection.count()

    if count == 0:
        return []

    query_embedding = model.encode([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(top_k, count)
    )

    dense_results = []

    for rank, chunk_id in enumerate(
        results["ids"][0],
        start=1
    ):
        distance = None

        if results.get("distances") and results["distances"][0]:
            distance = float(
                results["distances"][0][rank - 1]
            )

        dense_results.append({
            "chunk_id": chunk_id,
            "rank": rank,
            "distance": distance
        })

    return dense_results


# ------------------------------------------------------------
# RECIPROCAL RANK FUSION
# ------------------------------------------------------------

def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    k=RRF_K
):
    scores = {}

    for result in dense_results:
        chunk_id = result["chunk_id"]

        scores[chunk_id] = scores.get(chunk_id, 0.0)
        scores[chunk_id] += 1.0 / (k + result["rank"])

    for result in bm25_results:
        chunk_id = result["chunk_id"]

        scores[chunk_id] = scores.get(chunk_id, 0.0)
        scores[chunk_id] += 1.0 / (k + result["rank"])

    return sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )


def calculate_retrieval_confidence(
    hybrid_results,
    k=RRF_K
):
    if not hybrid_results:
        return 0.0

    max_rrf_score = 2.0 / (k + 1)
    top_rrf_score = hybrid_results[0][1]

    return min(
        max(top_rrf_score / max_rrf_score, 0.0),
        1.0
    )


# ------------------------------------------------------------
# REUSABLE HYBRID RETRIEVER
# ------------------------------------------------------------

class HybridRetriever:
    """
    Loads the corpus, embedding model, ChromaDB collection,
    and BM25 index once.

    Performs dense retrieval, BM25 retrieval, RRF,
    and adjacent-chunk expansion.
    """

    def __init__(
        self,
        top_k=TOP_K,
        expansion_seed_k=EXPANSION_SEED_K,
        neighbor_window=NEIGHBOR_WINDOW,
        max_context_chunks=MAX_CONTEXT_CHUNKS
    ):
        print("Initializing TruthGap Hybrid Retriever...")

        self.top_k = top_k
        self.expansion_seed_k = expansion_seed_k
        self.neighbor_window = neighbor_window
        self.max_context_chunks = max_context_chunks

        self.chunks = load_chunks()

        self.chunk_lookup = {
            chunk["chunk_id"]: chunk
            for chunk in self.chunks
        }

        print(f"Loaded {len(self.chunks)} chunks.")

        self.model = SentenceTransformer(MODEL_NAME)

        self.client = chromadb.PersistentClient(
            path=CHROMA_DIR
        )

        self.collection = self.client.get_collection(
            name=COLLECTION_NAME
        )

        if self.collection.count() == 0:
            raise ValueError(
                "ChromaDB collection contains no vectors."
            )

        collection_metadata = self.collection.metadata or {}

        self.distance_metric = collection_metadata.get(
            "hnsw:space",
            "l2"
        )

        print(f"ChromaDB vectors: {self.collection.count()}")
        print(f"Distance metric: {self.distance_metric}")

        self.bm25 = build_bm25(self.chunks)

        # Build document-wise chunk ordering for expansion.
        self.document_chunk_index = (
            self._build_document_chunk_index()
        )

        print("BM25 index built.")
        print("Adjacent-chunk index built.")
        print("Hybrid Retriever initialized successfully.")

    # --------------------------------------------------------
    # BUILD DOCUMENT-WISE CHUNK INDEX
    # --------------------------------------------------------

    def _build_document_chunk_index(self):
        """
        Groups chunks by document and orders them by page
        and chunk number.

        Example:
        DOC001_P6_C1
        DOC001_P6_C2
        DOC001_P7_C1
        """

        document_chunks = {}

        for chunk in self.chunks:
            document_id = chunk.get("document_id")
            chunk_id = chunk.get("chunk_id", "")

            if not document_id:
                continue

            match = re.match(
                r"^.+_P(\d+)_C(\d+)$",
                chunk_id
            )

            if match:
                page_number = int(match.group(1))
                chunk_number = int(match.group(2))
            else:
                # Fallback for chunk IDs that do not
                # follow the expected naming convention.
                page_number = int(chunk.get("page", 0))
                chunk_number = 0

            document_chunks.setdefault(
                document_id,
                []
            ).append({
                "chunk_id": chunk_id,
                "page": page_number,
                "chunk_number": chunk_number
            })

        for document_id in document_chunks:
            document_chunks[document_id].sort(
                key=lambda item: (
                    item["page"],
                    item["chunk_number"]
                )
            )

        return document_chunks

    # --------------------------------------------------------
    # ADJACENT-CHUNK EXPANSION
    # --------------------------------------------------------

    def expand_context(self, retrieved_chunks):
        """
        Adds neighboring chunks around the top-ranked
        retrieved chunks.

        Original retrieved chunks are kept first.
        Expanded chunks are appended without duplicates.

        Expansion does not change RRF scores or
        retrieval confidence.
        """

        if not retrieved_chunks:
            return []

        # Preserve original ranked chunks.
        context_chunks = [
            dict(chunk)
            for chunk in retrieved_chunks
        ]

        included_ids = {
            chunk["chunk_id"]
            for chunk in context_chunks
        }

        expanded_chunks = []

        # Use only the strongest retrieval results
        # as expansion seeds.
        seed_chunks = retrieved_chunks[
            :self.expansion_seed_k
        ]

        for seed in seed_chunks:
            document_id = seed.get("document_id")
            seed_id = seed.get("chunk_id")

            if not document_id or not seed_id:
                continue

            ordered_chunks = self.document_chunk_index.get(
                document_id,
                []
            )

            # Find the seed's position in its document.
            seed_position = next(
                (
                    index
                    for index, item in enumerate(ordered_chunks)
                    if item["chunk_id"] == seed_id
                ),
                None
            )

            if seed_position is None:
                continue

            start = max(
                0,
                seed_position - self.neighbor_window
            )

            end = min(
                len(ordered_chunks),
                seed_position + self.neighbor_window + 1
            )

            for position in range(start, end):
                neighbor_id = ordered_chunks[position]["chunk_id"]

                # Do not add the seed again or duplicate
                # chunks already present in the context.
                if neighbor_id in included_ids:
                    continue

                neighbor = self.chunk_lookup.get(neighbor_id)

                if neighbor is None:
                    continue

                expanded_chunk = dict(neighbor)

                expanded_chunk["is_expanded_context"] = True
                expanded_chunk["expanded_from"] = seed_id

                expanded_chunks.append(expanded_chunk)
                included_ids.add(neighbor_id)

        # Respect the maximum context size.
        remaining_slots = max(
            0,
            self.max_context_chunks - len(context_chunks)
        )

        context_chunks.extend(
            expanded_chunks[:remaining_slots]
        )

        return context_chunks

    # --------------------------------------------------------
    # RETRIEVE
    # --------------------------------------------------------

    def retrieve(self, query):
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        query = query.strip()

        # 1. Dense retrieval
        dense_results = dense_search(
            query,
            self.model,
            self.collection,
            self.top_k
        )

        # 2. BM25 retrieval
        bm25_results = bm25_search(
            query,
            self.chunks,
            self.bm25,
            self.top_k
        )

        # 3. Reciprocal Rank Fusion
        hybrid_results = reciprocal_rank_fusion(
            dense_results,
            bm25_results
        )

        # 4. Calculate confidence using original results.
        retrieval_confidence = calculate_retrieval_confidence(
            hybrid_results
        )

        # 5. Combine retrieval metadata.
        metadata_lookup = {}

        for item in dense_results:
            metadata_lookup[item["chunk_id"]] = {
                "dense_rank": item["rank"],
                "distance": item["distance"]
            }

        for item in bm25_results:
            metadata = metadata_lookup.setdefault(
                item["chunk_id"],
                {}
            )

            metadata["bm25_rank"] = item["rank"]
            metadata["bm25_score"] = item["score"]

        # 6. Build original ranked chunks.
        retrieved_chunks = []

        for chunk_id, rrf_score in hybrid_results[:self.top_k]:
            chunk = self.chunk_lookup.get(chunk_id)

            if chunk is None:
                continue

            result_chunk = dict(chunk)

            result_chunk["rrf_score"] = float(rrf_score)
            result_chunk["is_expanded_context"] = False
            result_chunk["expanded_from"] = None

            extra_metadata = metadata_lookup.get(
                chunk_id,
                {}
            )

            result_chunk["dense_rank"] = extra_metadata.get(
                "dense_rank"
            )

            result_chunk["bm25_rank"] = extra_metadata.get(
                "bm25_rank"
            )

            result_chunk["bm25_score"] = extra_metadata.get(
                "bm25_score",
                0.0
            )

            # Only expose distance as cosine distance
            # when the collection actually uses cosine.
            if self.distance_metric == "cosine":
                result_chunk["distance"] = extra_metadata.get(
                    "distance"
                )

            retrieved_chunks.append(result_chunk)

        # 7. Expand context using adjacent chunks.
        context_chunks = self.expand_context(
            retrieved_chunks
        )

        expanded_count = sum(
            1
            for chunk in context_chunks
            if chunk.get("is_expanded_context")
        )

        print(
            f"\nOriginal retrieved chunks: "
            f"{len(retrieved_chunks)}"
        )

        print(
            f"Additional adjacent chunks: "
            f"{expanded_count}"
        )

        print(
            f"Total context chunks: "
            f"{len(context_chunks)}"
        )

        return {
            # Expanded chunks for the generator.
            "chunks": context_chunks,

            # Explicit names for both sets.
            "context_chunks": context_chunks,
            "retrieved_chunks": retrieved_chunks,

            # Original retrieval scores remain unchanged.
            "hybrid_results": hybrid_results,
            "retrieval_confidence": retrieval_confidence,

            "dense_results": dense_results,
            "bm25_results": bm25_results
        }


# ------------------------------------------------------------
# MANUAL TEST
# ------------------------------------------------------------

if __name__ == "__main__":
    retriever = HybridRetriever()

    question = input(
        "\nEnter your question: "
    ).strip()

    results = retriever.retrieve(question)

    print("\nHYBRID RETRIEVAL RESULTS")
    print("=" * 60)

    print(
        f"Initial retrieval confidence: "
        f"{results['retrieval_confidence']:.4f}"
    )

    print(
        f"Original retrieved chunks: "
        f"{len(results['retrieved_chunks'])}"
    )

    print(
        f"Expanded context chunks: "
        f"{len(results['context_chunks'])}"
    )

    for rank, chunk in enumerate(
        results["context_chunks"],
        start=1
    ):
        print(f"\nContext Rank: {rank}")

        print(f"Document: {chunk.get('title')}")
        print(f"Page: {chunk.get('page')}")
        print(f"Chunk ID: {chunk.get('chunk_id')}")

        print(
            f"Expanded Context: "
            f"{chunk.get('is_expanded_context', False)}"
        )

        if chunk.get("expanded_from"):
            print(
                f"Expanded From: "
                f"{chunk['expanded_from']}"
            )

        if chunk.get("rrf_score") is not None:
            print(
                f"RRF Score: "
                f"{chunk['rrf_score']:.6f}"
            )

        print(
            f"Text: "
            f"{chunk.get('text', '')[:500]}"
        )
