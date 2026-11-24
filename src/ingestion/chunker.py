from typing import List, Dict
from pathlib import Path
import json

from pdf_extractor import extract_pdf


def chunk_text(
    pages: List[Dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[Dict]:
    """
    Convert extracted PDF pages into smaller text chunks.
    """

    chunks = []

    for page in pages:
        text = page["text"]
        document_id = page["document_id"]
        page_number = page["page"]

        start = 0
        chunk_number = 1

        while start < len(text):

            end = start + chunk_size
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(
                    {
                        "chunk_id": (
                            f"{document_id}_"
                            f"P{page_number}_"
                            f"C{chunk_number}"
                        ),
                        "document_id": document_id,
                        "page": page_number,
                        "text": chunk,
                    }
                )

            if end >= len(text):
                break

            start = end - chunk_overlap
            chunk_number += 1

    return chunks


def save_chunks(chunks: List[Dict], output_path: str):
    """
    Save chunks to a JSON file.
    """

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":

    pdf_path = "data/raw/draft_national_youth_policy_2025.pdf"

    pages = extract_pdf(
        pdf_path,
        "DOC001"
    )

    chunks = chunk_text(pages)

    output_path = "data/processed/chunks.json"

    save_chunks(chunks, output_path)

    print(f"Pages extracted: {len(pages)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Chunks saved to: {output_path}")