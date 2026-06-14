"""
WorthyHire — Centralized Configuration

All tunable parameters, model paths, and scoring weights live here.
"""

from dataclasses import dataclass, field


# ── Model Configuration ──────────────────────────────────────────────────

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Maximum candidates to pass to the cross-encoder reranker
RERANKER_TOP_N = 200

# Final output size
OUTPUT_TOP_N = 100


# ── Hybrid Score Fusion Weights ──────────────────────────────────────────
# These weights combine semantic similarity with rule-based dimension scores
# into a single hybrid score.

FUSION_WEIGHTS = {
    "semantic":         0.20,   # Was 0.30 — reduced, gated by skill score in fusion
    "must_have_skills": 0.35,   # Was 0.25 — skill match is the strongest signal
    "experience_fit":   0.15,   # Role fit + experience match
    "career":           0.10,   # Career quality (tenure, progression, product co.)
    "behavioral":       0.10,   # Availability, responsiveness, engagement
    "education":        0.05,   # Education tier + field relevance
    "red_flags":        0.05,   # Penalty for keyword-stuffers, consulting-only, etc.
}


# ── Legacy Dimension Weights (for rule-based scorer) ─────────────────────
# These are used internally by scoring.py for its sub-dimension computation.
# The final fusion uses FUSION_WEIGHTS above.

DIMENSION_WEIGHTS = {
    "role_fit":      0.35,
    "skills":        0.30,
    "career":        0.15,
    "education":     0.05,
    "behavioral":    0.10,
    "red_flags":     0.05,
}


# ── Behavioral Sub-Weights ───────────────────────────────────────────────

BEHAVIORAL_WEIGHTS = {
    "open_to_work":     0.09,
    "recency":          0.12,
    "response_rate":    0.14,
    "response_speed":   0.06,
    "notice_period":    0.10,
    "completeness":     0.05,
    "github":           0.12,
    "interview":        0.05,
    "offer":            0.03,
    "verification":     0.05,
    "location":         0.08,
    "market_demand":    0.05,
    "applications":     0.03,
    # Must sum to ~1.00 (currently 0.97, small rounding OK)
}


# ── Score Normalization ──────────────────────────────────────────────────

NORM_TARGET_MAX = 0.9990
NORM_TARGET_MIN = 0.2000
NORM_TIE_BREAKER = 0.00001


# ── Honeypot Detection ───────────────────────────────────────────────────

HONEYPOT_THRESHOLD = 2.0        # Suspicion score threshold
HONEYPOT_PENALTY = 0.05         # Multiplicative penalty for honeypots


# ── Embedding Text Construction ──────────────────────────────────────────

MAX_SKILLS_IN_EMBEDDING = 5     # Top N skills to include in embedding text
MAX_JOBS_IN_EMBEDDING = 2       # Number of recent jobs to include
MAX_DESC_LENGTH = 300           # Max chars per job description in embedding


# ── Confidence Thresholds ────────────────────────────────────────────────

CONFIDENCE_HIGH = 0.70
CONFIDENCE_MEDIUM = 0.45
