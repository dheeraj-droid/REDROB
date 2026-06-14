"""
WorthyHire — Hybrid Ranking Orchestrator

Orchestrates the full ranking pipeline:
  1. Parse JD → structured requirements
  2. Embed JD text
  3. Stream candidates → honeypot filter → embed
  4. Rule-based scoring (all dimensions)
  5. Semantic similarity scoring
  6. Fuse hybrid scores
  7. Cross-encoder rerank top 200 → top 100
  8. Generate structured explanations
  9. Run fairness audit
  10. Output results
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Optional

from backend.config import (
    FUSION_WEIGHTS,
    RERANKER_TOP_N,
    OUTPUT_TOP_N,
    HONEYPOT_PENALTY,
    NORM_TARGET_MAX,
    NORM_TARGET_MIN,
    NORM_TIE_BREAKER,
)
from backend.parsing.job_parser import get_jd_requirements, JDRequirements
from backend.ranking.scoring import score_candidate
from backend.ranking.honeypot_detector import detect_honeypot
from backend.embeddings.embedder import SemanticEmbedder
from backend.ranking.reranker import CrossEncoderReranker
from backend.explainability.explanations import generate_explanation
from backend.fairness.audit import FairnessAuditor


class HybridRanker:
    """
    End-to-end hybrid ranking pipeline combining semantic embeddings,
    rule-based scoring, and cross-encoder reranking.
    """

    def __init__(
        self,
        use_embeddings: bool = True,
        use_reranker: bool = True,
        verbose: bool = False,
    ):
        self.use_embeddings = use_embeddings
        self.use_reranker = use_reranker
        self.verbose = verbose

        self.embedder = SemanticEmbedder() if use_embeddings else None
        self.reranker = CrossEncoderReranker() if use_reranker else None
        self.auditor = FairnessAuditor()

    def rank(
        self,
        candidates: list[dict],
        jd: Optional[JDRequirements] = None,
        top_n: int = OUTPUT_TOP_N,
    ) -> dict:
        """
        Run the full hybrid ranking pipeline.

        Args:
            candidates: List of candidate dicts
            jd: Job description requirements (uses default if None)
            top_n: Number of top candidates to return

        Returns:
            dict with keys:
                ranked: list of ranked candidate dicts
                fairness_report: dict with fairness audit results
                metadata: dict with timing and stats
        """
        start_time = time.time()
        jd = jd or get_jd_requirements()

        self._log(f"Starting hybrid ranking for {len(candidates)} candidates")
        self._log(f"Target role: {jd.title}")

        # JD embedding is computed lazily in Stage 2 (only if embedder is enabled)

        # ── Stage 1: Rule-based scoring + honeypot detection (ALL candidates) ─
        # This is fast (~40s for 100K) and filters out ~75% of candidates
        self._log("Stage 1: Rule-based scoring + honeypot detection...")
        all_scored = []
        honeypot_count = 0

        for i, candidate in enumerate(candidates):
            cid = candidate.get("candidate_id", f"UNKNOWN_{i}")

            # Honeypot check
            is_honeypot, honeypot_reasons = detect_honeypot(candidate)
            if is_honeypot:
                honeypot_count += 1

            # Rule-based scoring
            rule_result = score_candidate(candidate, jd)

            # Compute rule-only score for pre-filtering
            dim = rule_result["dimension_scores"]
            rule_score = (
                0.35 * dim.get("skills", 0)
                + 0.25 * dim.get("role_fit", 0)
                + 0.15 * dim.get("career", 0)
                + 0.10 * dim.get("behavioral", 0)
                + 0.05 * dim.get("education", 0)
                + 0.10 * dim.get("red_flags", 1.0)
            )

            # Apply honeypot penalty
            if is_honeypot:
                rule_score *= HONEYPOT_PENALTY

            all_scored.append({
                "candidate_id": cid,
                "candidate": candidate,
                "score_result": rule_result,
                "rule_score": rule_score,
                "semantic_score": 0.0,
                "semantic_score_raw": 0.0,
                "hybrid_score": rule_score,
                "final_score": rule_score,
                "is_honeypot": is_honeypot,
                "honeypot_reasons": honeypot_reasons if is_honeypot else [],
                "candidate_text": "",
            })

            if self.verbose and (i + 1) % 10000 == 0:
                elapsed = time.time() - start_time
                self._log(
                    f"  Processed {i+1:,}/{len(candidates):,} "
                    f"({elapsed:.1f}s, {honeypot_count:,} honeypots)"
                )

        scoring_time = time.time() - start_time
        self._log(
            f"Stage 1 complete: {len(candidates):,} candidates in {scoring_time:.1f}s "
            f"({honeypot_count:,} honeypots detected)"
        )

        # ── Stage 2: Sort by rule score, embed only the top N candidates ──
        all_scored.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))

        # How many candidates to embed (top survivors after honeypot filter)
        EMBED_TOP_N = min(500, len(all_scored))
        top_for_embedding = all_scored[:EMBED_TOP_N]
        rest = all_scored[EMBED_TOP_N:]

        if self.embedder and len(top_for_embedding) > 0:
            self._log(f"Stage 2: Embedding top {EMBED_TOP_N} candidates...")
            jd_text = self.embedder.build_jd_text(jd)
            jd_embedding = self.embedder.embed_jd(jd)

            # Build text representations for top candidates only
            for entry in top_for_embedding:
                entry["candidate_text"] = self.embedder.build_candidate_text(
                    entry["candidate"]
                )

            # Batch embed
            self._log("Computing candidate embeddings in batch...")
            candidate_embeddings = self.embedder.embed_candidates_batch(
                [e["candidate"] for e in top_for_embedding]
            )

            # Compute semantic similarities
            semantic_scores = self.embedder.batch_cosine_similarity(
                jd_embedding, candidate_embeddings
            )

            # Normalize semantic scores within this pool
            sem_min = float(np.min(semantic_scores))
            sem_max = float(np.max(semantic_scores))
            if sem_max > sem_min:
                semantic_scores_norm = (semantic_scores - sem_min) / (sem_max - sem_min)
            else:
                semantic_scores_norm = np.full_like(semantic_scores, 0.5)

            # Update scores with semantic component
            for i, entry in enumerate(top_for_embedding):
                sem_raw = float(semantic_scores[i])
                sem_norm = float(semantic_scores_norm[i])
                sem_norm = max(0.0, min(1.0, sem_norm))

                entry["semantic_score_raw"] = round(sem_raw, 4)
                entry["semantic_score"] = round(sem_norm, 4)

                # Re-compute hybrid score with semantic component
                hybrid_score = self._fuse_scores(
                    semantic=sem_norm,
                    rule_result=entry["score_result"],
                )

                if entry["is_honeypot"]:
                    hybrid_score *= HONEYPOT_PENALTY

                entry["hybrid_score"] = hybrid_score
                entry["final_score"] = hybrid_score

            # Re-sort after adding semantic scores
            top_for_embedding.sort(
                key=lambda x: (-x["final_score"], x["candidate_id"])
            )

            embed_time = time.time() - start_time - scoring_time
            self._log(f"Stage 2 complete in {embed_time:.1f}s")
        else:
            jd_text = ""

        # Merge back (top_for_embedding is now re-sorted; rest stays below)
        all_scored = top_for_embedding + rest

        # ── Stage 3: Cross-encoder rerank top N ───────────────────────────
        rerank_n = min(RERANKER_TOP_N, len(all_scored))
        top_for_rerank = all_scored[:rerank_n]

        if self.reranker and len(top_for_rerank) > 0 and jd_text:
            self._log(f"Stage 3: Reranking top {rerank_n} with cross-encoder...")
            rerank_texts = [e["candidate_text"] for e in top_for_rerank]
            top_for_rerank = self.reranker.rerank(
                jd_text=jd_text,
                candidates=top_for_rerank,
                candidate_texts=rerank_texts,
                top_n=top_n,
            )
            self._log("Reranking complete.")
        else:
            top_for_rerank = top_for_rerank[:top_n]

        # ── Step 5: Generate explanations ─────────────────────────────────
        self._log("Generating explanations...")
        ranked_output = []

        max_raw = top_for_rerank[0]["final_score"] if top_for_rerank else 1.0
        min_raw = top_for_rerank[-1]["final_score"] if top_for_rerank else 0.0

        for rank, entry in enumerate(top_for_rerank, 1):
            explanation = generate_explanation(
                candidate=entry["candidate"],
                score_result=entry["score_result"],
                rank=rank,
                semantic_score=entry.get("semantic_score", 0),
                jd=jd,
            )

            norm_score = _normalize_score(
                entry["final_score"], rank, len(top_for_rerank),
                min_raw, max_raw,
            )

            entry["rank"] = rank
            entry["normalized_score"] = round(norm_score, 4)
            entry["explanation"] = explanation
            ranked_output.append(entry)

        # ── Step 6: Fairness audit ────────────────────────────────────────
        self._log("Running fairness audit...")
        fairness_report = self.auditor.audit(
            all_candidates=all_scored,
            top_candidates=ranked_output,
        )

        total_time = time.time() - start_time
        self._log(f"Pipeline complete in {total_time:.1f}s")

        # Check honeypot rate
        top_honeypots = sum(1 for c in ranked_output if c["is_honeypot"])
        honeypot_rate = top_honeypots / max(len(ranked_output), 1)

        return {
            "ranked": ranked_output,
            "fairness_report": fairness_report,
            "metadata": {
                "total_candidates": len(candidates),
                "honeypots_detected": honeypot_count,
                "top_n_honeypots": top_honeypots,
                "honeypot_rate": round(honeypot_rate, 4),
                "scoring_time_seconds": round(scoring_time, 2),
                "total_time_seconds": round(total_time, 2),
                "use_embeddings": self.use_embeddings,
                "use_reranker": self.use_reranker,
            },
        }

    def _fuse_scores(self, semantic: float, rule_result: dict) -> float:
        """Fuse semantic and rule-based scores using configured weights."""
        dim = rule_result["dimension_scores"]
        signals = rule_result.get("signals", {})

        skill_score = dim.get("skills", 0)
        role_fit = dim.get("role_fit", 0)

        # ── Skill gate: crush semantic score for candidates with no relevant skills ──
        # If skill_score < 0.2, the candidate has near-zero must-have skills.
        # The embedding model can't distinguish "well-written irrelevant profile"
        # from "genuinely relevant profile", so we gate the semantic contribution.
        if skill_score < 0.15:
            # No relevant skills at all — semantic is unreliable, crush it
            gated_semantic = semantic * 0.15
        elif skill_score < 0.3:
            # Very few relevant skills — reduce semantic contribution
            gated_semantic = semantic * 0.4
        else:
            gated_semantic = semantic

        # Similarly gate on role_fit — if title is completely irrelevant,
        # the semantic model is likely matching on text patterns, not job fit
        if role_fit < 0.15:
            gated_semantic *= 0.5

        # Map rule-based dimensions to fusion categories
        fused = (
            FUSION_WEIGHTS["semantic"] * gated_semantic
            + FUSION_WEIGHTS["must_have_skills"] * skill_score
            + FUSION_WEIGHTS["experience_fit"] * role_fit
            + FUSION_WEIGHTS["career"] * dim.get("career", 0)
            + FUSION_WEIGHTS["behavioral"] * dim.get("behavioral", 0)
            + FUSION_WEIGHTS["education"] * dim.get("education", 0)
            + FUSION_WEIGHTS["red_flags"] * dim.get("red_flags", 1.0)
        )

        return max(0.0, min(1.0, fused))

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [WorthyHire] {msg}")


def _normalize_score(
    raw_score: float, rank: int, total: int,
    min_raw: float, max_raw: float,
) -> float:
    """
    Normalize scores proportionally to preserve actual score gaps.
    Maps to [NORM_TARGET_MIN, NORM_TARGET_MAX] range.
    Falls back to linear-by-rank if all scores are identical.
    """
    if max_raw == min_raw or max_raw <= 0:
        base_norm = NORM_TARGET_MAX - (rank - 1) * (
            NORM_TARGET_MAX - NORM_TARGET_MIN
        ) / max(total - 1, 1)
    else:
        ratio = (raw_score - min_raw) / (max_raw - min_raw)
        base_norm = NORM_TARGET_MIN + ratio * (NORM_TARGET_MAX - NORM_TARGET_MIN)

    tie_breaker = (rank - 1) * NORM_TIE_BREAKER
    norm_score = max(
        base_norm - tie_breaker,
        NORM_TARGET_MIN + (total - rank) * NORM_TIE_BREAKER,
    )
    return norm_score
