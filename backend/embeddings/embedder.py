"""
WorthyHire — Semantic Embedding Engine

Uses sentence-transformers (BAAI/bge-small-en-v1.5) to compute dense
embeddings for job descriptions and candidate profiles, enabling
semantic matching beyond keyword overlap.

Performance: ~500 candidates/sec on CPU (bge-small-en-v1.5)
Model size:  ~130MB
"""

import numpy as np
from typing import Optional

from backend.config import (
    EMBEDDING_MODEL,
    MAX_SKILLS_IN_EMBEDDING,
    MAX_JOBS_IN_EMBEDDING,
    MAX_DESC_LENGTH,
)


class SemanticEmbedder:
    """Manages embedding model lifecycle and provides embedding functions."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    # ── Text construction ─────────────────────────────────────────────────

    @staticmethod
    def build_jd_text(jd) -> str:
        """
        Build embedding text from a JDRequirements object.

        Concatenates: title + must-have categories + nice-to-have skills
        """
        parts = [
            f"Job Title: {jd.title}",
            f"Company: {jd.company} ({jd.stage})",
            f"Experience: {jd.min_years}-{jd.max_years} years",
            f"Must-have skills: {', '.join(jd.must_have_categories)}",
            f"Nice-to-have skills: {', '.join(jd.nice_to_have_skills)}",
        ]
        return " | ".join(parts)

    @staticmethod
    def build_candidate_text(candidate: dict) -> str:
        """
        Build embedding text from a candidate dict.

        Concatenates: headline + summary + top N skills + current title
        + last M job descriptions
        """
        profile = candidate.get("profile", {})
        career = candidate.get("career_history", [])
        skills = candidate.get("skills", [])

        parts = []

        # Current title and headline
        title = profile.get("current_title", "")
        headline = profile.get("headline", "")
        if title:
            parts.append(f"Current Role: {title}")
        if headline:
            parts.append(f"Headline: {headline}")

        # Summary (truncated)
        summary = profile.get("summary", "")
        if summary:
            parts.append(f"Summary: {summary[:500]}")

        # Top N skills by proficiency
        prof_order = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}
        sorted_skills = sorted(
            skills,
            key=lambda s: prof_order.get(s.get("proficiency", ""), 0),
            reverse=True,
        )
        top_skills = [s.get("name", "") for s in sorted_skills[:MAX_SKILLS_IN_EMBEDDING]]
        if top_skills:
            parts.append(f"Key Skills: {', '.join(top_skills)}")

        # Recent job descriptions
        for job in career[:MAX_JOBS_IN_EMBEDDING]:
            job_title = job.get("title", "")
            company = job.get("company", "")
            desc = job.get("description", "")[:MAX_DESC_LENGTH]
            if desc:
                parts.append(f"Role at {company} ({job_title}): {desc}")

        return " | ".join(parts)

    # ── Embedding computation ─────────────────────────────────────────────

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        model = self._load_model()
        return model.encode(text, normalize_embeddings=True)

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Embed a batch of text strings."""
        model = self._load_model()
        return model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def embed_jd(self, jd) -> np.ndarray:
        """Embed a JDRequirements object."""
        text = self.build_jd_text(jd)
        return self.embed_text(text)

    def embed_candidate(self, candidate: dict) -> np.ndarray:
        """Embed a single candidate."""
        text = self.build_candidate_text(candidate)
        return self.embed_text(text)

    def embed_candidates_batch(
        self, candidates: list[dict], batch_size: int = 64
    ) -> np.ndarray:
        """Embed a batch of candidates."""
        texts = [self.build_candidate_text(c) for c in candidates]
        return self.embed_texts(texts, batch_size=batch_size)

    # ── Similarity computation ────────────────────────────────────────────

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two normalized vectors."""
        return float(np.dot(a, b))

    @staticmethod
    def batch_cosine_similarity(
        query: np.ndarray, candidates: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between one query and many candidates.

        Args:
            query: (D,) normalized embedding
            candidates: (N, D) normalized embeddings

        Returns:
            (N,) similarity scores
        """
        return candidates @ query
