"""
TruthGap Grounded Generation Module

Generates answers from retrieved document excerpts, checks citation provenance,
and supports Gemini, Groq, or an offline mock provider.
"""

import json
import os
import re
import urllib.request
from typing import Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SYSTEM_GROUNDING_RULES = """You are TruthGap's Grounded Answer Engine.
Use only facts explicitly supported by the supplied document excerpts.
Do not use outside knowledge, assumptions, or memory.
Cite every factual claim using [Document Title, Page 4].
Use only document titles and page numbers present in the excerpts.
If the excerpts do not adequately answer the question, respond exactly:
"The provided document corpus does not contain sufficient grounded information to answer this question."
If excerpts conflict, describe each position separately and cite each one.
Answer concisely and professionally.
"""

ABSTENTION_MESSAGE = (
    "The provided document corpus does not contain sufficient grounded "
    "information to answer this question."
)

ABSTENTION_PHRASES = (
    "does not contain sufficient grounded information",
    "does not contain sufficient information",
    "insufficient grounded information",
    "cannot be answered from the provided",
    "no grounded information",
    "no information provided",
    "not mentioned in the provided",
    "corpus does not contain",
    "not supported by the provided",
)

CITATION_PATTERN = re.compile(
    r"\[([^,\]]+),\s*Page\s+(\d+)\]",
    flags=re.IGNORECASE,
)


class GroundedGenerator:
    """Grounded answer generation with Gemini, Groq, and offline mock modes."""

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        preferred_provider: Optional[str] = None,
        gemini_model: Optional[str] = None,
        groq_model: Optional[str] = None,
    ):
        self.gemini_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.groq_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.gemini_model_name = (
            gemini_model or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"
        )
        self.groq_model_name = (
            groq_model
            or os.getenv("GROQ_MODEL")
            or "llama-3.3-70b-versatile"
        )

        provider_choice = (
            preferred_provider or os.getenv("PREFERRED_PROVIDER", "auto")
        ).strip().lower()

        self.provider = self._resolve_provider(provider_choice)
        self.gemini_client = None

        if self.provider == "gemini":
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=self.gemini_key)
                print("[SUCCESS] Gemini client initialized.")
            except Exception as exc:
                raise RuntimeError(
                    "Gemini was selected but its client could not be "
                    "initialized. Install google-genai and verify your API key."
                ) from exc

    def _resolve_provider(self, preferred: str) -> str:
        if preferred not in {"auto", "gemini", "groq", "mock"}:
            raise ValueError(
                "PREFERRED_PROVIDER must be one of: auto, gemini, groq, mock."
            )
        if preferred == "mock":
            return "mock"
        if preferred == "gemini":
            if not self.gemini_key:
                raise ValueError("GEMINI_API_KEY is not set.")
            return "gemini"
        if preferred == "groq":
            if not self.groq_key:
                raise ValueError("GROQ_API_KEY is not set.")
            return "groq"
        if self.gemini_key:
            return "gemini"
        if self.groq_key:
            return "groq"
        return "mock"

    @staticmethod
    def format_context(context_chunks: List[Dict]) -> str:
        if not context_chunks:
            return "NO EXCERPTS PROVIDED."
        parts = []
        for index, chunk in enumerate(context_chunks, start=1):
            title = chunk.get("title") or chunk.get("document_id") or "Unknown Document"
            doc_id = chunk.get("document_id", "")
            page = chunk.get("page", "?")
            year = chunk.get("year", "")
            text = str(chunk.get("text", "")).strip()
            parts.append(
                f"=== EXCERPT {index} [Source: {title} | DocID: {doc_id} "
                f"| Page: {page} | Year: {year}] ===\n{text}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def extract_and_verify_citations(
        answer: str, context_chunks: List[Dict]
    ) -> Tuple[List[str], List[str], float]:
        matches = CITATION_PATTERN.findall(answer or "")
        valid_pairs = set()

        for chunk in context_chunks:
            page = str(chunk.get("page", "")).strip()
            title = str(chunk.get("title", "")).strip().casefold()
            doc_id = str(chunk.get("document_id", "")).strip().casefold()
            if page:
                if title:
                    valid_pairs.add((title, page))
                if doc_id:
                    valid_pairs.add((doc_id, page))

        valid_citations = []
        hallucinated_citations = []
        for cited_doc, cited_page in matches:
            doc_norm = cited_doc.strip().casefold()
            page_norm = cited_page.strip()
            citation = f"[{cited_doc.strip()}, Page {page_norm}]"
            if (doc_norm, page_norm) in valid_pairs:
                valid_citations.append(citation)
            else:
                hallucinated_citations.append(citation)

        total = len(valid_citations) + len(hallucinated_citations)
        ratio = len(valid_citations) / total if total else 0.0
        return valid_citations, hallucinated_citations, ratio

    @staticmethod
    def check_abstention(answer: str) -> bool:
        answer_lower = (answer or "").casefold()
        return any(phrase in answer_lower for phrase in ABSTENTION_PHRASES)

    

    def _call_gemini(self, prompt: str) -> str:
        """Generate an answer using Google's Interactions API."""

        interaction = self.gemini_client.interactions.create(
            model=self.gemini_model_name,
            input=prompt,
            system_instruction=SYSTEM_GROUNDING_RULES,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 1024,
            },
        )

        answer = getattr(interaction, "output_text", None)

        if not answer or not answer.strip():
            raise ValueError("Gemini returned an empty response.")

        return answer.strip()

    def generate(
        self,
        query: str,
        context_chunks: List[Dict],
        force_abstain: bool = False,
    ) -> Dict:
        """Generate an answer grounded in retrieved document chunks."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        if force_abstain or not context_chunks:
            return {
                "answer": ABSTENTION_MESSAGE,
                "status": "CORPUS_ABSTAINED",
                "is_abstention": True,
                "valid_citations": [],
                "hallucinated_citations": [],
                "grounding_validity": 1.0,
                "provider": self.provider,
                "model": "rule_based_gate",
            }

        prompt = (
            f"USER QUESTION:\n{query.strip()}\n\n"
            f"AVAILABLE DOCUMENT EXCERPTS:\n"
            f"{self.format_context(context_chunks)}\n\n"
            "Answer using only these excerpts. Cite every factual claim. "
            f"If evidence is insufficient, respond exactly:\n"
            f"{ABSTENTION_MESSAGE}"
        )

        try:
            if self.provider == "gemini":
                raw_answer = self._call_gemini(prompt)
            elif self.provider == "groq":
                raw_answer = self._call_groq(prompt)
            else:
                raw_answer = self._call_mock(query, context_chunks)

            is_abstention = self.check_abstention(raw_answer)

            valid, hallucinated, ratio = (
                self.extract_and_verify_citations(
                    raw_answer, context_chunks
                )
            )

            if is_abstention:
                status = "ABSTAINED"
            elif not valid or hallucinated:
                status = "CITATION_CHECK_FAILED"
            else:
                status = "ANSWERED"

            model_used = {
                "gemini": self.gemini_model_name,
                "groq": self.groq_model_name,
                "mock": "mock_engine",
            }[self.provider]

            return {
                "answer": raw_answer,
                "status": status,
                "is_abstention": is_abstention,
                "valid_citations": valid,
                "hallucinated_citations": hallucinated,
                "grounding_validity": ratio,
                "provider": self.provider,
                "model": model_used,
            }

        except Exception as exc:
            return {
                "answer": (
                    f"Generation Error: {type(exc).__name__}: {exc}"
                ),
                "status": "ERROR",
                "is_abstention": False,
                "valid_citations": [],
                "hallucinated_citations": [],
                "grounding_validity": 0.0,
                "provider": self.provider,
                "model": self.gemini_model_name,
            }


if __name__ == "__main__":
    print("=" * 60)
    print("TRUTHGAP GROUNDED GENERATOR INITIALIZATION")
    print("=" * 60)
    generator = GroundedGenerator()
    print(f"Active Provider: {generator.provider}")
    if generator.provider == "gemini":
        print(f"Active Model: {generator.gemini_model_name}")
    elif generator.provider == "groq":
        print(f"Active Model: {generator.groq_model_name}")
    else:
        print("Active Model: mock_engine")

    sample_chunks = [{
        "chunk_id": "DOC001_P4_C1",
        "document_id": "DOC001",
        "title": "Draft National Youth Policy 2025",
        "page": 4,
        "year": 2025,
        "text": (
            "The National Youth Policy 2025 aims to foster leadership, "
            "social entrepreneurship, and community engagement across "
            "priority age demographics (15 to 29 years)."
        ),
    }]

    print("\n--- Grounded answer test ---")
    print(json.dumps(
        generator.generate(
            "What does the National Youth Policy 2025 aim to foster?",
            sample_chunks,
        ),
        indent=2,
        ensure_ascii=False,
    ))
    print("\n--- Out-of-corpus abstention test ---")
    print(json.dumps(
        generator.generate("What is the youth policy of Japan?", sample_chunks),
        indent=2,
        ensure_ascii=False,
    ))
