import chromadb
from sentence_transformers import SentenceTransformer


CHROMA_DIR = "data/processed/chroma_db"
COLLECTION_NAME = "truthgap_policies"
MODEL_NAME = "all-MiniLM-L6-v2"


def search(query: str, top_k: int = 5):

    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    query_embedding = model.encode(
        [query]
    )[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    return results


if __name__ == "__main__":

    query = input("\nEnter your question: ")

    results = search(query)

    print("\n==============================")
    print("DENSE RETRIEVAL RESULTS")
    print("==============================")

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1
    ):

        print(f"\nRank {i}")
        print("------------------------------")

        print(
            f"Document: {metadata['title']}"
        )

        print(
            f"Page: {metadata['page']}"
        )

        print(
            f"Distance: {distance:.4f}"
        )

        print(
            f"Text:\n{document[:500]}"
        )