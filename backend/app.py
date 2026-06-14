"""
WorthyHire — FastAPI Backend

API server for candidate ranking.

Endpoints:
  POST /rank    — Upload candidates, get ranked results
  GET  /health  — Health check
  GET  /config  — Current configuration
"""

import sys
import os
import json
from pathlib import Path

# Ensure project root is on path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.config import (
    FUSION_WEIGHTS,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    OUTPUT_TOP_N,
)
from backend.schemas import (
    RankRequest,
    RankResponse,
    RankedCandidateResponse,
    ExplanationResponse,
    FairnessReportResponse,
    FairnessFlagResponse,
    MetadataResponse,
    HealthResponse,
    ConfigResponse,
)
from backend.ranking.hybrid_ranker import HybridRanker


app = FastAPI(
    title="WorthyHire",
    description=(
        "An explainable AI recruiter that ranks candidates using semantic "
        "understanding, structured career signals, and India-specific "
        "hiring context."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse()


@app.get("/config", response_model=ConfigResponse)
async def config():
    """Return current scoring configuration."""
    return ConfigResponse(
        fusion_weights=FUSION_WEIGHTS,
        embedding_model=EMBEDDING_MODEL,
        reranker_model=RERANKER_MODEL,
        output_top_n=OUTPUT_TOP_N,
    )


@app.post("/rank")
async def rank_candidates(request: RankRequest):
    """
    Rank candidates against the job description.

    Accepts a list of candidate dicts and returns ranked results
    with structured explanations and a fairness audit report.
    """
    if not request.candidates:
        raise HTTPException(status_code=400, detail="No candidates provided")

    ranker = HybridRanker(
        use_embeddings=request.use_embeddings,
        use_reranker=request.use_reranker,
        verbose=True,
    )

    result = ranker.rank(
        candidates=request.candidates,
        top_n=request.top_n,
    )

    # Build response
    ranked_response = []
    for entry in result["ranked"]:
        expl = entry.get("explanation", {})
        ranked_response.append({
            "candidate_id": entry["candidate_id"],
            "rank": entry["rank"],
            "score": entry["normalized_score"],
            "semantic_score": entry.get("semantic_score", 0),
            "is_honeypot": entry.get("is_honeypot", False),
            "explanation": expl,
        })

    return {
        "ranked": ranked_response,
        "fairness_report": result["fairness_report"],
        "metadata": result["metadata"],
    }


@app.post("/rank/file")
async def rank_from_file(
    file: UploadFile = File(...),
    top_n: int = Form(default=100),
    use_embeddings: bool = Form(default=True),
    use_reranker: bool = Form(default=True),
):
    """
    Rank candidates from an uploaded JSON/JSONL file.
    """
    content = await file.read()
    text = content.decode("utf-8")

    # Detect format
    if file.filename and file.filename.endswith(".jsonl"):
        candidates = []
        for line in text.strip().split("\n"):
            if line.strip():
                candidates.append(json.loads(line))
    else:
        data = json.loads(text)
        if isinstance(data, list):
            candidates = data
        else:
            candidates = [data]

    if not candidates:
        raise HTTPException(status_code=400, detail="No candidates found in file")

    ranker = HybridRanker(
        use_embeddings=use_embeddings,
        use_reranker=use_reranker,
        verbose=True,
    )

    result = ranker.rank(candidates=candidates, top_n=top_n)

    ranked_response = []
    for entry in result["ranked"]:
        expl = entry.get("explanation", {})
        ranked_response.append({
            "candidate_id": entry["candidate_id"],
            "rank": entry["rank"],
            "score": entry["normalized_score"],
            "semantic_score": entry.get("semantic_score", 0),
            "is_honeypot": entry.get("is_honeypot", False),
            "explanation": expl,
        })

    return {
        "ranked": ranked_response,
        "fairness_report": result["fairness_report"],
        "metadata": result["metadata"],
    }
