
"""
TruthGap Core Query Engine
==========================

End-to-end pipeline:
Hybrid Retrieval -> Relevance-Aware Abstention Gate
-> Grounded Generation -> Citation Verification
"""

from pathlib import Path
import sys
from typing import Dict, List, Optional


# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent
PROJECT_ROOT = SRC_DIR.parent

for path in (str(PROJECT_ROOT), str(SRC_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)


# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------

from retrieval.abstention_gate import AbstentionGate
from retrieval.hybrid_retriever import HybridRetriever
from generation.generator import GroundedGenerator


# ------------------------------------------------------------
# TRUTHGAP ENGINE
# ------------------------------------------------------------

class TruthGapEngine:
    """
    Complete TruthGap question-answering engine.
    """

    def __init__(
        self,
        tau_high: float = 0.54,
        tau_low: float = 0.44,
        preferred_provider: str = "gemini",
        retriever=None
    ):
        print("Initializing TruthGap Engine...")

        self.gate = AbstentionGate(
            tau_high=tau_high,
            tau_low=tau_low
        )

        self.retriever = (
            retriever if retriever is not None
            else HybridRetriever()
        )

        self.generator = GroundedGenerator(
            preferred_provider=preferred_provider
        )

        print("TruthGap Engine initialized successfully.")

    # --------------------------------------------------------
    # COMMON RESPONSE FORMAT
    # --------------------------------------------------------

    def _response(
        self,
        question,
        answer,
        decision,
        confidence,
        is_abstained,
        chunks=None,
        valid_citations=None,
        hallucinated_citations=None,
        model_used=None,
        status=None,
        signals=None
    ):
        return {
            "question": question,
            "answer": answer,
            "gate_decision": decision,
            "confidence_score": confidence,
            "is_abstained": is_abstained,
            "valid_citations": valid_citations or [],
            "hallucinated_citations": hallucinated_citations or [],
            "retrieved_chunks_count": len(chunks or []),
            "model_used": model_used,
            "status": status,
            "signals": signals or {}
        }

    # --------------------------------------------------------
    # QUERY
    # --------------------------------------------------------

    def query(
        self,
        question: str,
        retrieved_chunks: Optional[List[Dict]] = None,
        retrieval_confidence: Optional[float] = None
    ) -> Dict:

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty.")

        question = question.strip()

        # STEP 1: RETRIEVAL
        if retrieved_chunks is None:
            retrieval_result = self.retriever.retrieve(question)

            # Original ranked chunks are used by the gate.
            ranked_chunks = retrieval_result.get(
                "retrieved_chunks",
                retrieval_result["chunks"]
            )

            # Expanded chunks are used by Gemini.
            context_chunks = retrieval_result.get(
                "context_chunks",
                retrieval_result["chunks"]
            )

            hybrid_results = retrieval_result["hybrid_results"]

        else:
            # Manual retrieval mode for controlled testing.
            ranked_chunks = retrieved_chunks
            context_chunks = retrieved_chunks

            hybrid_results = [
                (chunk["chunk_id"], chunk["rrf_score"])
                for chunk in ranked_chunks
                if "chunk_id" in chunk and "rrf_score" in chunk
            ]

        # STEP 2: HANDLE EMPTY RETRIEVAL
        if not ranked_chunks:
            decision = self.gate.evaluate(
                hybrid_results=[],
                query=question,
                corpus_chunks=getattr(
                    self.retriever,
                    "chunks",
                    None
                )
            )

            return self._response(
                question=question,
                answer=decision.disclaimer,
                decision=decision.decision,
                confidence=decision.confidence_score,
                is_abstained=True,
                chunks=[],
                status="ABSTAINED",
                model_used="abstention_gatekeeper",
                signals=decision.signals
            )

        # STEP 3: RELEVANCE-AWARE ABSTENTION GATE
        gate_arguments = {
            "hybrid_results": hybrid_results,
            "top_chunks_metadata": ranked_chunks,
            "query": question,
            "corpus_chunks": getattr(
                self.retriever,
                "chunks",
                None
            )
        }

        if retrieval_confidence is not None:
            gate_arguments["override_score"] = retrieval_confidence

        gate_decision = self.gate.evaluate(**gate_arguments)

        print("\nGate diagnostics:")
        for name, value in gate_decision.signals.items():
            print(f"  {name}: {value}")

        # STEP 4: ABSTAIN WHEN THE GATE REJECTS THE EVIDENCE
        if gate_decision.should_abstain:
            return self._response(
                question=question,
                answer=gate_decision.disclaimer,
                decision=gate_decision.decision,
                confidence=gate_decision.confidence_score,
                is_abstained=True,
                chunks=context_chunks,
                model_used="abstention_gatekeeper",
                status="ABSTAINED",
                signals=gate_decision.signals
            )

        # STEP 5: GROUNDED GENERATION
        try:
            gen_result = self.generator.generate(
                query=question,
                context_chunks=context_chunks,
                force_abstain=False
            )

        except Exception:
            return self._response(
                question=question,
                answer=(
                    "Answer generation failed. TruthGap could not "
                    "produce a verified response."
                ),
                decision=gate_decision.decision,
                confidence=gate_decision.confidence_score,
                is_abstained=True,
                chunks=context_chunks,
                model_used="generation_error",
                status="ERROR",
                signals=gate_decision.signals
            )

        # STEP 6: CHECK GENERATOR STATUS
        generation_status = gen_result.get("status", "UNKNOWN")
        answer = gen_result.get("answer", "")

        valid_citations = gen_result.get(
            "valid_citations",
            []
        )

        hallucinated_citations = gen_result.get(
            "hallucinated_citations",
            []
        )

        model_used = gen_result.get("model", "unknown")

        # Never present failed citation verification as a valid answer.
        if generation_status in (
            "ERROR",
            "CITATION_CHECK_FAILED"
        ):
            return self._response(
                question=question,
                answer=(
                    "TruthGap could not verify this generated response. "
                    "Please consult the original source documents."
                ),
                decision=gate_decision.decision,
                confidence=gate_decision.confidence_score,
                is_abstained=True,
                chunks=context_chunks,
                valid_citations=valid_citations,
                hallucinated_citations=hallucinated_citations,
                model_used=model_used,
                status=generation_status,
                signals=gate_decision.signals
            )

        # STEP 7: GENERATOR-LEVEL ABSTENTION
        if gen_result.get("is_abstention", False):
            return self._response(
                question=question,
                answer=answer,
                decision=gate_decision.decision,
                confidence=gate_decision.confidence_score,
                is_abstained=True,
                chunks=context_chunks,
                valid_citations=valid_citations,
                hallucinated_citations=hallucinated_citations,
                model_used=model_used,
                status="ABSTAINED",
                signals=gate_decision.signals
            )

        # STEP 8: ADD CAUTION WARNING
        if gate_decision.requires_warning and gate_decision.disclaimer:
            answer = (
                f"{gate_decision.disclaimer}\n\n{answer}"
            )

        # STEP 9: FINAL RESPONSE
        return self._response(
            question=question,
            answer=answer,
            decision=gate_decision.decision,
            confidence=gate_decision.confidence_score,
            is_abstained=False,
            chunks=context_chunks,
            valid_citations=valid_citations,
            hallucinated_citations=hallucinated_citations,
            model_used=model_used,
            status=generation_status,
            signals=gate_decision.signals
        )


# ------------------------------------------------------------
# MANUAL TEST
# ------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("TRUTHGAP END-TO-END QUERY ENGINE")
    print("=" * 60)

    engine = TruthGapEngine(
        tau_high=0.54,
        tau_low=0.44,
        preferred_provider="gemini"
    )

    while True:
        question = input(
            "\nAsk TruthGap (or type exit): "
        ).strip()

        if question.lower() in ("exit", "quit"):
            break

        try:
            result = engine.query(question)

            print("\n" + "=" * 60)
            print("TRUTHGAP RESPONSE")
            print("=" * 60)

            print("\nAnswer:")
            print(result["answer"])

            print("\nGate Decision:")
            print(result["gate_decision"])

            print("\nConfidence:")
            print(f"{result['confidence_score']:.2%}")

            print("\nStatus:")
            print(result["status"])

            print("\nCitations:")
            print(result["valid_citations"])

            print("\nInvalid Citations:")
            print(result["hallucinated_citations"])

            print("\nContext Chunks:")
            print(result["retrieved_chunks_count"])

            print("\nModel:")
            print(result["model_used"])

            print("\nDiagnostic Signals:")
            for key, value in result["signals"].items():
                print(f"{key}: {value}")

        except Exception as exc:
            print(f"\nTruthGap error: {exc}")
