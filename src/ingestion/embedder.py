import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


CHUNKS_FILE = Path("data/processed/all_chunks.json")
CHROMA_DIR = "data/processed/chroma_db"

COLLECTION_NAME = "truthgap_policies"

MODEL_NAME = "all-MiniLM-L6-v2"


def create_vector_database():

    print("Loading chunks...")

    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        chunks = json.load(file)

    print(f"Total chunks: {len(chunks)}")

    print("\nLoading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Embedding model loaded.")

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "TruthGap Government Youth Policy Corpus"
        }
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    metadatas = [
        {
            "document_id": chunk["document_id"],
            "page": chunk["page"],
            "title": chunk["title"],
            "ministry": chunk["ministry"],
            "category": chunk["category"],
            "year": str(chunk["year"]),
            "source_url": chunk["source_url"]
        }
        for chunk in chunks
    ]

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32
    )

    print("Embeddings generated.")

    print("\nStoring vectors in ChromaDB...")

    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas
    )

    print("\n-----------------------------")
    print("VECTOR DATABASE CREATED")
    print("-----------------------------")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Vectors stored: {collection.count()}")
    print(f"Database: {CHROMA_DIR}")


if __name__ == "__main__":
    create_vector_database()