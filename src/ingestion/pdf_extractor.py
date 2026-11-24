import pymupdf
from pathlib import Path


def extract_pdf(pdf_path: str, document_id: str):
    """
    Extract text from a PDF page by page.

    Returns a list of dictionaries containing:
    - document_id
    - page number
    - extracted text
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text").strip()

        if text:
            pages.append(
                {
                    "document_id": document_id,
                    "page": page_number,
                    "text": text,
                }
            )

    document.close()

    return pages


if __name__ == "__main__":
    pdf = "data/raw/draft_national_youth_policy_2025.pdf"

    pages = extract_pdf(pdf, "DOC001")

    print(f"Pages extracted: {len(pages)}")

    if pages:
        print("\nFirst page preview:\n")
        print(pages[0]["text"][:1000])