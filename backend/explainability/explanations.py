"""
WorthyHire — Structured Explainability Engine

Generates structured explanations for candidate rankings including:
  - matched_skills: Skills the candidate has that match JD requirements
  - missing_skills: JD must-have skills the candidate lacks
  - confidence: high/medium/low based on score
  - explanation: Natural language reasoning
  - risk_notes: Red flags and concerns
"""

import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from backend.config import CONFIDENCE_HIGH, CONFIDENCE_MEDIUM

from skill_taxonomy import (
    classify_skill,
    get_category_relevance,
    MUST_HAVE_CLUSTERS,
    CATEGORY_RELEVANCE,
    NON_TECHNICAL_TITLES,
    CONSULTING_SERVICES_COMPANIES,
)
from reasoning_generator import generate_reasoning


def generate_explanation(
    candidate: dict,
    score_result: dict,
    rank: int,
    semantic_score: float = 0.0,
    jd=None,
) -> dict:
    """
    Generate a structured explanation for a ranked candidate.

    Returns:
        dict with keys: matched_skills, missing_skills, confidence,
                        explanation, risk_notes, semantic_score
    """
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])

    # ── Matched skills ────────────────────────────────────────────────────
    matched_skills = []
    candidate_skill_names = set()
    for s in skills:
        name = s.get("name", "")
        candidate_skill_names.add(name.lower())
        cat = classify_skill(name)
        rel = get_category_relevance(cat)
        if rel >= 0.4:
            matched_skills.append(name)

    # ── Missing skills ────────────────────────────────────────────────────
    missing_skills = []
    for cluster_name, cluster_info in MUST_HAVE_CLUSTERS.items():
        cluster_skills = cluster_info["skills"]
        has_any = any(
            s.lower() in candidate_skill_names
            for s in cluster_skills
        )
        if not has_any:
            missing_skills.append(f"{cluster_info['description']} ({cluster_name})")

    # Also check JD nice-to-have skills
    if jd:
        for nice_skill in jd.nice_to_have_skills:
            if nice_skill.lower() not in candidate_skill_names:
                # Only add top nice-to-haves to missing, not all
                cat = classify_skill(nice_skill)
                if get_category_relevance(cat) >= 0.6:
                    missing_skills.append(nice_skill)

    # ── Confidence ────────────────────────────────────────────────────────
    final_score = score_result.get("final_score", 0)
    if final_score >= CONFIDENCE_HIGH:
        confidence = "high"
    elif final_score >= CONFIDENCE_MEDIUM:
        confidence = "medium"
    else:
        confidence = "low"

    # ── Risk notes ────────────────────────────────────────────────────────
    risk_notes = []

    # Non-technical title
    title = profile.get("current_title", "").lower().strip()
    if title in NON_TECHNICAL_TITLES:
        risk_notes.append(f"Non-technical current title: {profile.get('current_title', '')}")

    # Consulting-only career
    companies = [j.get("company", "").lower().strip() for j in career]
    if companies and all(c in CONSULTING_SERVICES_COMPANIES for c in companies if c):
        risk_notes.append("Consulting-only career history")

    # Long notice period
    notice = candidate.get("redrob_signals", {}).get("notice_period_days", 0)
    if notice and notice > 90:
        risk_notes.append(f"{notice}-day notice period")

    # Low response rate
    resp_rate = candidate.get("redrob_signals", {}).get("recruiter_response_rate", 0)
    if resp_rate and resp_rate < 0.2:
        risk_notes.append(f"Low response rate ({resp_rate:.0%})")

    # ── Natural language explanation ──────────────────────────────────────
    text_reasoning = generate_reasoning(candidate, score_result, rank)

    # Enhance with semantic score info
    if semantic_score > 0.7:
        text_reasoning = f"HIGH semantic match ({semantic_score:.2f}). " + text_reasoning
    elif semantic_score > 0.4:
        text_reasoning = f"Moderate semantic match ({semantic_score:.2f}). " + text_reasoning

    return {
        "matched_skills": matched_skills[:10],
        "missing_skills": missing_skills[:5],
        "confidence": confidence,
        "explanation": text_reasoning,
        "risk_notes": risk_notes,
        "semantic_score": round(semantic_score, 4),
    }
