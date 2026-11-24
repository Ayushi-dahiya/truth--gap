
"""
TruthGap Relevance-Aware Abstention Gate
=========================================

Combines:
1. Hybrid retrieval strength
2. Dense/BM25 agreement
3. Query-to-context lexical relevance
4. Document-level country coverage

Important:
- Retrieval confidence is NOT answer correctness probability.
- Country coverage is based on document-level metadata
  and explicitly registered document coverage.
- These signals are heuristic, not calibrated probabilities.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import re


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

DEFAULT_TAU_HIGH = 0.55
DEFAULT_TAU_LOW = 0.45
DEFAULT_MIN_RELEVANCE = 0.20


STOP_WORDS = {
    "what", "which", "who", "where", "when", "why", "how",
    "is", "are", "was", "were", "the", "a", "an", "of",
    "in", "on", "for", "to", "and", "or", "with", "by",
    "from", "do", "does", "did", "can", "could", "would",
    "should", "explain", "tell", "me", "about", "please",
    "main", "core", "give", "list", "describe", "policy",
    "objectives", "objective"
}


COUNTRY_NAMES = {
    "afghanistan", "albania", "algeria", "argentina",
    "australia", "austria", "bangladesh", "belgium",
    "bhutan", "brazil", "canada", "china", "denmark",
    "egypt", "finland", "france", "germany", "ghana",
    "greece", "india", "indonesia", "iran", "iraq",
    "ireland", "israel", "italy", "japan", "kenya",
    "malaysia", "maldives", "mexico", "nepal",
    "netherlands", "new zealand", "nigeria",
    "north korea", "norway", "pakistan", "philippines",
    "poland", "portugal", "qatar", "russia",
    "saudi arabia", "singapore", "south africa",
    "south korea", "spain", "sri lanka", "sweden",
    "switzerland", "thailand", "turkey", "ukraine",
    "united arab emirates", "united kingdom",
    "united states", "usa", "vietnam"
}


# ------------------------------------------------------------
# DOCUMENT-LEVEL COUNTRY COVERAGE
# ------------------------------------------------------------

# These are the four documents currently indexed in TruthGap.
#
# DOC001: Draft National Youth Policy 2025
# DOC002: PMKVY 4.0 Guidelines
# DOC003: JSS 2.0 Guidelines
# DOC004: Ministry of Youth Affairs Annual Report 2023-24
#
# All four are Indian government documents.
#
# Do NOT add a country merely because its name appears
# somewhere in the document text.
#
# Update this mapping when adding new documents.

DOCUMENT_COUNTRY_OVERRIDES = {
    "DOC001": {"india"},
    "DOC002": {"india"},
    "DOC003": {"india"},
    "DOC004": {"india"},
}


# ------------------------------------------------------------
# GATE DECISION
# ------------------------------------------------------------

@dataclass
class GateDecision:

    decision: str
    confidence_score: float
    raw_rrf_score: float
    should_abstain: bool
    requires_warning: bool
    disclaimer: Optional[str]
    signals: Dict[str, float]

    def to_dict(self) -> Dict:
        return asdict(self)


# ------------------------------------------------------------
# ABSTENTION GATE
# ------------------------------------------------------------

class AbstentionGate:

    def __init__(
        self,
        tau_high: float = DEFAULT_TAU_HIGH,
        tau_low: float = DEFAULT_TAU_LOW,
        rrf_k: int = 60,
        min_relevance: float = DEFAULT_MIN_RELEVANCE
    ):

        if not 0 <= tau_low <= tau_high <= 1:
            raise ValueError(
                "Thresholds must satisfy "
                "0 <= tau_low <= tau_high <= 1"
            )

        if not 0 <= min_relevance <= 1:
            raise ValueError(
                "min_relevance must be between 0 and 1"
            )

        self.tau_high = tau_high
        self.tau_low = tau_low
        self.rrf_k = rrf_k
        self.min_relevance = min_relevance

        self.max_rrf_score = 2.0 / (rrf_k + 1.0)

    # --------------------------------------------------------
    # TOKENIZATION
    # --------------------------------------------------------

    @staticmethod
    def _tokens(text: str) -> set:

        return set(
            re.findall(
                r"\b[a-z0-9]+\b",
                text.lower()
            )
        )

    # --------------------------------------------------------
    # COUNTRY ENTITY DETECTION
    # --------------------------------------------------------

    @staticmethod
    def _find_country_entities(text: str) -> List[str]:

        normalized = text.lower()

        found = []

        for country in COUNTRY_NAMES:

            pattern = (
                r"(?<!\w)"
                + re.escape(country)
                + r"(?!\w)"
            )

            if re.search(pattern, normalized):
                found.append(country)

        return found

    # --------------------------------------------------------
    # EXTRACT DOCUMENT ID
    # --------------------------------------------------------

    @staticmethod
    def _get_document_id(
        chunk: Dict
    ) -> Optional[str]:

        chunk_id = str(
            chunk.get("chunk_id", "")
        )

        match = re.match(
            r"(DOC\d+)",
            chunk_id
        )

        if match:
            return match.group(1)

        return None

    # --------------------------------------------------------
    # DOCUMENT-LEVEL COUNTRY COVERAGE
    # --------------------------------------------------------

    def _get_covered_countries(
        self,
        corpus_chunks: Optional[List[Dict]]
    ) -> set:

        """
        Determine country coverage from explicit metadata
        and registered document-level coverage.

        Incidental country mentions inside document text
        are NOT considered proof of country coverage.
        """

        covered_countries = set()

        if not corpus_chunks:
            return covered_countries

        for chunk in corpus_chunks:

            # 1. Read explicit document metadata.

            metadata = chunk.get(
                "metadata",
                {}
            )

            if not isinstance(metadata, dict):
                metadata = {}

            country_values = [
                chunk.get("country"),
                chunk.get("jurisdiction"),
                chunk.get("document_country"),
                metadata.get("country"),
                metadata.get("jurisdiction"),
                metadata.get("document_country")
            ]

            for value in country_values:

                if isinstance(value, str) and value.strip():

                    covered_countries.update(
                        self._find_country_entities(value)
                    )

            # 2. Apply explicitly registered document coverage.

            document_id = self._get_document_id(chunk)

            if document_id in DOCUMENT_COUNTRY_OVERRIDES:

                covered_countries.update(
                    DOCUMENT_COUNTRY_OVERRIDES[document_id]
                )

        return covered_countries

    # --------------------------------------------------------
    # RELEVANCE AND ENTITY SIGNALS
    # --------------------------------------------------------

    def _relevance_signals(
        self,
        query: Optional[str],
        retrieved_chunks: Optional[List[Dict]],
        corpus_chunks: Optional[List[Dict]]
    ) -> Dict[str, float]:

        if not query or not retrieved_chunks:

            return {
                "query_term_coverage": 0.0,
                "relevance_score": 0.0,
                "country_entity_detected": 0.0,
                "country_entity_in_corpus": 0.0,
                "entity_mismatch": 0.0
            }

        # Query keywords

        query_tokens = {

            token
            for token in self._tokens(query)

            if token not in STOP_WORDS
            and len(token) > 1
        }

        # Top 10 retrieved passages

        retrieved_text = " ".join(

            str(chunk.get("title", ""))
            + " "
            + str(chunk.get("text", ""))

            for chunk in retrieved_chunks[:10]
        )

        retrieved_tokens = self._tokens(
            retrieved_text
        )

        # Calculate lexical coverage

        if query_tokens:

            matched = query_tokens.intersection(
                retrieved_tokens
            )

            coverage = (
                len(matched) / len(query_tokens)
            )

        else:

            coverage = 0.0

        # Detect countries explicitly requested by user.

        query_countries = self._find_country_entities(
            query
        )

        # Determine countries covered by indexed documents.
        # Never search all document text for country names.

        covered_countries = self._get_covered_countries(
            corpus_chunks
        )

        missing_countries = [

            country
            for country in query_countries

            if country not in covered_countries
        ]

        entity_mismatch = bool(
            missing_countries
        )

        return {

            "query_term_coverage": round(
                coverage, 4
            ),

            "relevance_score": round(
                coverage, 4
            ),

            "country_entity_detected": float(
                bool(query_countries)
            ),

            "country_entity_in_corpus": float(
                bool(query_countries)
                and not entity_mismatch
            ),

            "entity_mismatch": float(
                entity_mismatch
            )
        }

    # --------------------------------------------------------
    # CONFIDENCE CALCULATION
    # --------------------------------------------------------

    def compute_confidence(
        self,
        hybrid_results: List[Tuple[str, float]],
        top_chunks_metadata: Optional[List[Dict]] = None
    ) -> Tuple[float, Dict[str, float]]:

        if not hybrid_results:

            return 0.0, {

                "rrf_ratio": 0.0,
                "dense_present": 0.0,
                "bm25_present": 0.0,
                "cosine_sim": 0.0,
                "composite_score": 0.0
            }

        top_chunk_id, top_rrf_score = hybrid_results[0]

        # Normalize RRF score.

        rrf_ratio = min(

            max(
                top_rrf_score / self.max_rrf_score,
                0.0
            ),

            1.0
        )

        dense_present = 0.0
        bm25_present = 0.0

        # Check dense/BM25 agreement.

        if top_chunks_metadata:

            for chunk in top_chunks_metadata:

                if chunk.get("chunk_id") == top_chunk_id:

                    dense_present = float(
                        chunk.get("dense_rank") is not None
                    )

                    bm25_present = float(

                        chunk.get("bm25_rank") is not None

                        or chunk.get(
                            "bm25_score", 0
                        ) > 0
                    )

                    break

        # Consensus bonus.

        consensus_bonus = (

            0.10

            if dense_present and bm25_present

            else 0.0
        )

        # ChromaDB uses L2 distance in this project.
        # Do not calculate cosine similarity from L2 distance.

        composite_score = (

            0.90 * rrf_ratio
            + consensus_bonus
        )

        composite_score = min(
            max(composite_score, 0.0),
            1.0
        )

        signals = {

            "rrf_ratio": round(
                rrf_ratio, 4
            ),

            "dense_present": dense_present,

            "bm25_present": bm25_present,

            "cosine_sim": 0.0,

            "composite_score": round(
                composite_score, 4
            )
        }

        return composite_score, signals

    # --------------------------------------------------------
    # MAIN EVALUATION
    # --------------------------------------------------------

    def evaluate(
        self,
        hybrid_results: List[Tuple[str, float]],
        top_chunks_metadata: Optional[List[Dict]] = None,
        override_score: Optional[float] = None,
        query: Optional[str] = None,
        corpus_chunks: Optional[List[Dict]] = None
    ) -> GateDecision:

        # ----------------------------------------------------
        # STEP 1: CONFIDENCE
        # ----------------------------------------------------

        if override_score is not None:

            confidence = min(
                max(override_score, 0.0),
                1.0
            )

            raw_rrf = (
                confidence * self.max_rrf_score
            )

            signals = {

                "rrf_ratio": round(
                    confidence, 4
                ),

                "composite_score": round(
                    confidence, 4
                )
            }

        else:

            confidence, signals = self.compute_confidence(
                hybrid_results,
                top_chunks_metadata
            )

            raw_rrf = (

                hybrid_results[0][1]

                if hybrid_results

                else 0.0
            )

        # ----------------------------------------------------
        # STEP 2: RELEVANCE AND COUNTRY COVERAGE
        # ----------------------------------------------------

        relevance = self._relevance_signals(

            query=query,

            retrieved_chunks=top_chunks_metadata,

            corpus_chunks=corpus_chunks
        )

        signals.update(relevance)

        # ----------------------------------------------------
        # STEP 3: COUNTRY MISMATCH CHECK
        # ----------------------------------------------------

        if relevance["entity_mismatch"] == 1.0:

            return GateDecision(

                decision="CORPUS_ABSTENTION",

                confidence_score=round(
                    confidence, 4
                ),

                raw_rrf_score=round(
                    raw_rrf, 6
                ),

                should_abstain=True,

                requires_warning=True,

                disclaimer=(

                    "Corpus blind spot: the question names a "
                    "country or entity that is not covered by "
                    "the indexed documents. The retrieved "
                    "documents cannot establish an answer "
                    "for the requested entity."
                ),

                signals=signals
            )

        # ----------------------------------------------------
        # STEP 4: LOW RELEVANCE CHECK
        # ----------------------------------------------------

        if (

            query is not None

            and top_chunks_metadata

            and relevance["relevance_score"]
            < self.min_relevance

        ):

            return GateDecision(

                decision="CORPUS_ABSTENTION",

                confidence_score=round(
                    confidence, 4
                ),

                raw_rrf_score=round(
                    raw_rrf, 6
                ),

                should_abstain=True,

                requires_warning=True,

                disclaimer=(

                    "Corpus blind spot: the retrieved passages "
                    "have insufficient lexical relevance to "
                    "the question. TruthGap cannot establish "
                    "a grounded answer."
                ),

                signals=signals
            )

        # ----------------------------------------------------
        # STEP 5: CONFIDENT ANSWER
        # ----------------------------------------------------

        if confidence >= self.tau_high:

            return GateDecision(

                decision="CONFIDENT_ANSWER",

                confidence_score=round(
                    confidence, 4
                ),

                raw_rrf_score=round(
                    raw_rrf, 6
                ),

                should_abstain=False,

                requires_warning=False,

                disclaimer=None,

                signals=signals
            )

        # ----------------------------------------------------
        # STEP 6: LOW CONFIDENCE CAUTION
        # ----------------------------------------------------

        if confidence >= self.tau_low:

            return GateDecision(

                decision="LOW_CONFIDENCE_CAUTION",

                confidence_score=round(
                    confidence, 4
                ),

                raw_rrf_score=round(
                    raw_rrf, 6
                ),

                should_abstain=False,

                requires_warning=True,

                disclaimer=(

                    f"Low grounding warning: retrieval "
                    f"confidence ({confidence:.1%}) is "
                    "borderline. Retrieved excerpts may "
                    "be incomplete or tangential."
                ),

                signals=signals
            )

        # ----------------------------------------------------
        # STEP 7: CORPUS ABSTENTION
        # ----------------------------------------------------

        return GateDecision(

            decision="CORPUS_ABSTENTION",

            confidence_score=round(
                confidence, 4
            ),

            raw_rrf_score=round(
                raw_rrf, 6
            ),

            should_abstain=True,

            requires_warning=True,

            disclaimer=(

                f"Corpus blind spot: retrieval confidence "
                f"({confidence:.1%}) is below the minimum "
                f"threshold ({self.tau_low:.1%}). "
                "There is insufficient grounded evidence "
                "to answer."
            ),

            signals=signals
        )


# ------------------------------------------------------------
# STANDALONE TEST
# ------------------------------------------------------------

if __name__ == "__main__":

    gate = AbstentionGate()

    demo_corpus = [

        {
            "chunk_id": "DOC001_P6_C1",
            "title": "Draft National Youth Policy 2025",
            "text": (
                "National Youth Policy India youth "
                "leadership objectives"
            )
        },

        {
            "chunk_id": "DOC004_P84_C1",
            "title": "Ministry of Youth Affairs Annual Report",
            "text": (
                "India youth affairs international "
                "G20 and Japan discussions"
            )
        }
    ]

    demo_retrieved = [

        {
            "chunk_id": "DOC001_P6_C1",
            "title": "Draft National Youth Policy 2025",
            "text": (
                "National Youth Policy India youth "
                "leadership objectives"
            ),
            "dense_rank": 1,
            "bm25_rank": 1,
            "bm25_score": 5.0
        }
    ]

    hybrid_results = [
        ("DOC001_P6_C1", 2.0 / 61.0)
    ]

    questions = [

        "What are the objectives of India's National Youth Policy?",

        "What are the objectives of Japan's Youth Policy?"
    ]

    print("=" * 65)
    print("TRUTHGAP COUNTRY COVERAGE TEST")
    print("=" * 65)

    for question in questions:

        result = gate.evaluate(

            hybrid_results=hybrid_results,

            top_chunks_metadata=demo_retrieved,

            query=question,

            corpus_chunks=demo_corpus
        )

        print("\nQuestion:", question)

        print("Decision:", result.decision)

        print("Confidence:", result.confidence_score)

        print("Abstain:", result.should_abstain)

        print("Signals:", result.signals)

        print("Reason:", result.disclaimer)

    print("\n" + "=" * 65)
