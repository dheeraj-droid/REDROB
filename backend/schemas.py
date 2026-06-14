"""
WorthyHire — API Schemas (Pydantic models)

Request/response schemas for the FastAPI backend.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ── Request Models ────────────────────────────────────────────────────────

class RankRequest(BaseModel):
    """Request body for the /rank endpoint."""
    candidates: list[dict] = Field(
        ...,
        description="List of candidate profile dicts",
    )
    top_n: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Number of top candidates to return",
    )
    use_embeddings: bool = Field(
        default=True,
        description="Whether to use semantic embeddings",
    )
    use_reranker: bool = Field(
        default=True,
        description="Whether to use cross-encoder reranking",
    )


# ── Response Models ───────────────────────────────────────────────────────

class ExplanationResponse(BaseModel):
    """Structured explanation for a ranked candidate."""
    matched_skills: list[str] = []
    missing_skills: list[str] = []
    confidence: str = "low"
    explanation: str = ""
    risk_notes: list[str] = []
    semantic_score: float = 0.0


class RankedCandidateResponse(BaseModel):
    """A single ranked candidate in the response."""
    candidate_id: str
    rank: int
    score: float
    semantic_score: float = 0.0
    is_honeypot: bool = False
    explanation: ExplanationResponse


class FairnessFlagResponse(BaseModel):
    """A single fairness flag."""
    group: str
    metric: str
    value: float
    severity: str
    message: str


class FairnessReportResponse(BaseModel):
    """Fairness audit report."""
    audit_timestamp: str
    groups_analyzed: list[str] = []
    flags: list[FairnessFlagResponse] = []
    summary: str = ""


class MetadataResponse(BaseModel):
    """Pipeline execution metadata."""
    total_candidates: int
    honeypots_detected: int
    top_n_honeypots: int
    honeypot_rate: float
    scoring_time_seconds: float
    total_time_seconds: float
    use_embeddings: bool
    use_reranker: bool


class RankResponse(BaseModel):
    """Full response from the /rank endpoint."""
    ranked: list[RankedCandidateResponse]
    fairness_report: FairnessReportResponse
    metadata: MetadataResponse


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""
    status: str = "ok"
    version: str = "1.0.0"
    name: str = "WorthyHire"


class ConfigResponse(BaseModel):
    """Response from the /config endpoint."""
    fusion_weights: dict
    embedding_model: str
    reranker_model: str
    output_top_n: int
