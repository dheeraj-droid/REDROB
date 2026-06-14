"""
WorthyHire — Cross-Encoder Reranker

Uses a cross-encoder model (ms-marco-MiniLM-L-6-v2) to rerank the top
candidates with higher accuracy than bi-encoder similarity alone.

Cross-encoders process (query, document) pairs jointly, making them
more accurate but too slow for the full candidate set. We only run
this on the top ~200 candidates from the initial ranking.

Performance: ~0.5s for 100 pairs on CPU
Model size:  ~80MB
"""

from backend.config import RERANKER_MODEL


class CrossEncoderReranker:
    """Cross-encoder reranker for top-N candidate refinement."""

    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        jd_text: str,
        candidates: list[dict],
        candidate_texts: list[str],
        top_n: int = 100,
    ) -> list[dict]:
        """
        Rerank candidates using the cross-encoder.

        Args:
            jd_text: The job description text
            candidates: List of candidate dicts (with scores from hybrid ranking)
            candidate_texts: Pre-built embedding texts for each candidate
            top_n: Number of candidates to return after reranking

        Returns:
            Reranked list of candidate dicts with updated scores
        """
        model = self._load_model()

        # Create (JD, candidate) pairs for the cross-encoder
        pairs = [(jd_text, ct) for ct in candidate_texts]

        # Score all pairs
        ce_scores = model.predict(pairs, show_progress_bar=False)

        # Normalize cross-encoder scores to [0, 1]
        min_s = min(ce_scores) if len(ce_scores) > 0 else 0
        max_s = max(ce_scores) if len(ce_scores) > 0 else 1
        if max_s > min_s:
            ce_scores_norm = [(s - min_s) / (max_s - min_s) for s in ce_scores]
        else:
            ce_scores_norm = [0.5] * len(ce_scores)

        # Blend cross-encoder score with existing hybrid score
        # Keep hybrid dominant — CE is a search model, not a job matcher
        # 25% cross-encoder, 75% existing hybrid score
        for i, cand in enumerate(candidates):
            existing_score = cand.get("hybrid_score", cand.get("final_score", 0))
            cand["ce_score"] = float(ce_scores_norm[i])
            cand["final_score"] = 0.25 * ce_scores_norm[i] + 0.75 * existing_score

        # Sort by final score descending
        candidates.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))

        return candidates[:top_n]
