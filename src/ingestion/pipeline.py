import json
from pathlib import Path

import pandas as pd

from pdf_extractor import extract_pdf
from chunker import chunk_text


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
METADATA_FILE = DATA_DIR / "metadata" / "documents.csv"
OUTPUT_FILE = DATA_DIR / "processed" / "all_chunks.json"


def process_documents():

    metadata = pd.read_csv(METADATA_FILE)

    all_chunks = []

    for _, document in metadata.iterrows():

        document_id = document["document_id"]
        filename = document["filename"]

        pdf_path = RAW_DIR / filename

        print(f"\nProcessing: {filename}")

        pages = extract_pdf(
            str(pdf_path),
            document_id
        )

        chunks = chunk_text(pages)

        for chunk in chunks:
            chunk["title"] = document["title"]
            chunk["ministry"] = document["ministry"]
            chunk["category"] = document["category"]
            chunk["year"] = document["year"]
            chunk["source_url"] = document["source_url"]

        all_chunks.extend(chunks)

        print(f"Pages: {len(pages)}")
        print(f"Chunks: {len(chunks)}")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("\n-----------------------------")
    print("INGESTION COMPLETED")
    print("-----------------------------")
    print(f"Documents: {len(metadata)}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    process_documents()