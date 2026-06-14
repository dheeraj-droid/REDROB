"""
Multi-dimensional scoring engine for the Redrob AI Ranking Challenge.

Scores each candidate across 6 dimensions:
  1. Role Fit        (0.30) — title match, career trajectory, product-co experience
  2. Skills Match    (0.25) — semantic skill matching with trust multiplier
  3. Career Quality  (0.15) — company diversity, progression, tenure
  4. Education       (0.05) — tier, relevant field
  5. Behavioral      (0.15) — availability, responsiveness, platform engagement
  6. Red Flags       (0.10) — penalty for keyword-stuffers, consulting-only, etc.

All scores are normalized to [0, 1] before weighted fusion.
"""

import math
import re
from datetime import datetime, date
from typing import Any

from jd_parser import JDRequirements, get_jd_requirements
from skill_taxonomy import (
    classify_skill,
    get_category_relevance,
    get_title_relevance,
    is_non_technical_title,
    is_consulting_company,
    MUST_HAVE_CLUSTERS,
    PROFICIENCY_WEIGHT,
    CATEGORY_RELEVANCE,
    AI_ML_TITLES,
    NON_TECHNICAL_TITLES,
)

# ── Dimension weights ─────────────────────────────────────────────────────
DIMENSION_WEIGHTS = {
    "role_fit":      0.30,
    "skills":        0.25,
    "career":        0.15,
    "education":     0.05,
    "behavioral":    0.15,
    "red_flags":     0.10,
}


def score_candidate(candidate: dict, jd: JDRequirements | None = None) -> dict:
    """
    Score a single candidate against the JD.

    Returns a dict with:
      - dimension_scores: dict of dimension → float [0, 1]
      - final_score: float [0, 1]
      - signals: dict of notable signals for reasoning
    """
    if jd is None:
        jd = get_jd_requirements()

    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills_list = candidate.get("skills", [])
    education = candidate.get("education", [])
    certs = candidate.get("certifications", [])
    signals = candidate.get("redrob_signals", {})

    # Compute each dimension
    role_score, role_signals = _score_role_fit(profile, career, jd)
    skills_score, skills_signals = _score_skills(skills_list, certs, signals, jd)
    career_score, career_signals = _score_career(career, profile, jd)
    edu_score, edu_signals = _score_education(education, jd)
    behavioral_score, behav_signals = _score_behavioral(signals, profile, jd)
    red_flag_penalty, red_flag_signals = _score_red_flags(
        profile, career, skills_list, signals, jd
    )

    # Clamp all to [0, 1]
    role_score = _clamp(role_score)
    skills_score = _clamp(skills_score)
    career_score = _clamp(career_score)
    edu_score = _clamp(edu_score)
    behavioral_score = _clamp(behavioral_score)
    red_flag_penalty = _clamp(red_flag_penalty)  # 1.0 = no flags, 0.0 = severe

    dimension_scores = {
        "role_fit":   role_score,
        "skills":     skills_score,
        "career":     career_score,
        "education":  edu_score,
        "behavioral": behavioral_score,
        "red_flags":  red_flag_penalty,
    }

    # Weighted fusion
    # Red flags act as a multiplicative penalty on the additive score
    base_score = (
        DIMENSION_WEIGHTS["role_fit"] * role_score +
        DIMENSION_WEIGHTS["skills"] * skills_score +
        DIMENSION_WEIGHTS["career"] * career_score +
        DIMENSION_WEIGHTS["education"] * edu_score +
        DIMENSION_WEIGHTS["behavioral"] * behavioral_score
    )

    # Red flag penalty is multiplicative — a keyword-stuffer with perfect
    # skills score still gets crushed
    final_score = base_score * (
        DIMENSION_WEIGHTS["red_flags"] + (1 - DIMENSION_WEIGHTS["red_flags"]) * red_flag_penalty
    )

    all_signals = {}
    all_signals.update(role_signals)
    all_signals.update(skills_signals)
    all_signals.update(career_signals)
    all_signals.update(edu_signals)
    all_signals.update(behav_signals)
    all_signals.update(red_flag_signals)

    return {
        "dimension_scores": dimension_scores,
        "final_score": _clamp(final_score),
        "signals": all_signals,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Dimension 1: Role Fit (0.30)
# ═══════════════════════════════════════════════════════════════════════════

def _score_role_fit(profile: dict, career: list, jd: JDRequirements) -> tuple[float, dict]:
    """
    Evaluate how well the candidate's role history fits the target position.
    """
    signals = {}
    score = 0.0

    # ── Current title relevance ───────────────────────────────────────────
    current_title = profile.get("current_title", "")
    title_score = get_title_relevance(current_title)
    score += title_score * 0.35
    signals["title_relevance"] = round(title_score, 2)
    signals["current_title"] = current_title

    # ── Career trajectory: has the person been moving toward AI/ML? ───────
    trajectory_score = 0.0
    n_roles = len(career)
    ai_ml_roles = 0
    product_roles = 0
    recent_ai_role = False

    for i, job in enumerate(career):
        title = job.get("title", "").lower()
        description = job.get("description", "").lower()
        company = job.get("company", "")
        is_current = job.get("is_current", False)

        # Check if role is AI/ML related (by title or description)
        title_is_ai = any(t in title for t in [
            "ai", "ml", "machine learning", "data scien",
            "nlp", "deep learning", "research"
        ])
        desc_is_ai = _description_has_ai_signal(description)

        if title_is_ai or desc_is_ai:
            ai_ml_roles += 1
            if is_current or i == 0:
                recent_ai_role = True

        # Product company detection
        if not is_consulting_company(company):
            product_roles += 1

    if n_roles > 0:
        trajectory_score = (ai_ml_roles / n_roles) * 0.6
        if recent_ai_role:
            trajectory_score += 0.4  # Huge boost for currently in AI role

    score += trajectory_score * 0.30
    signals["ai_role_count"] = ai_ml_roles
    signals["recent_ai_role"] = recent_ai_role

    # ── Years of experience fit ───────────────────────────────────────────
    yoe = profile.get("years_of_experience", 0)
    yoe_score = _experience_fit_score(yoe, jd)
    score += yoe_score * 0.20
    signals["years_of_experience"] = yoe

    # ── Product company experience ────────────────────────────────────────
    if n_roles > 0:
        product_ratio = product_roles / n_roles
    else:
        product_ratio = 0
    score += product_ratio * 0.15
    signals["product_company_ratio"] = round(product_ratio, 2)

    return score, signals


def _description_has_ai_signal(desc: str) -> bool:
    """Check if a role description contains AI/ML work signals."""
    ai_terms = [
        "machine learning", "deep learning", "neural network",
        "embedding", "vector", "nlp", "natural language",
        "transformer", "bert", "gpt", "llm", "language model",
        "ranking", "retrieval", "recommendation", "search",
        "model training", "model serving", "model deploy",
        "data science", "feature engineering", "a/b test",
        "classification", "regression", "clustering",
        "tensorflow", "pytorch", "scikit",
        "fine-tun", "inference", "prediction",
    ]
    return any(term in desc for term in ai_terms)


def _experience_fit_score(yoe: float, jd: JDRequirements) -> float:
    """Score how well experience years fit the JD range."""
    if jd.ideal_min_years <= yoe <= jd.ideal_max_years:
        return 1.0
    elif jd.min_years <= yoe <= jd.max_years:
        return 0.85
    elif (jd.min_years - jd.flexible_band) <= yoe <= (jd.max_years + jd.flexible_band):
        return 0.6
    elif yoe > jd.max_years + jd.flexible_band:
        # Over-experienced — mild penalty
        return max(0.2, 0.6 - (yoe - jd.max_years - jd.flexible_band) * 0.05)
    else:
        # Under-experienced
        return max(0.1, 0.5 - (jd.min_years - jd.flexible_band - yoe) * 0.1)


# ═══════════════════════════════════════════════════════════════════════════
# Dimension 2: Skills Match (0.25)
# ═══════════════════════════════════════════════════════════════════════════

def _score_skills(
    skills: list, certs: list, signals: dict, jd: JDRequirements
) -> tuple[float, dict]:
    """
    Score skills with semantic matching, proficiency weighting,
    and duration-based trust.
    """
    sig = {}
    if not skills:
        return 0.0, {"skill_count": 0}

    # ── Must-have cluster coverage ────────────────────────────────────────
    skill_names = {s.get("name", "").lower() for s in skills}
    clusters_hit = 0
    total_clusters = len(jd.must_have_categories)

    for cluster_name in jd.must_have_categories:
        cluster = MUST_HAVE_CLUSTERS.get(cluster_name, {})
        cluster_skills = {s.lower() for s in cluster.get("skills", [])}
        if skill_names & cluster_skills:
            clusters_hit += 1

    cluster_coverage = clusters_hit / max(total_clusters, 1)
    sig["must_have_coverage"] = f"{clusters_hit}/{total_clusters}"

    # ── Per-skill weighted scoring ────────────────────────────────────────
    weighted_skill_score = 0.0
    max_possible = 0.0
    core_ai_skills = 0
    relevant_skills = 0

    for skill in skills:
        name = skill.get("name", "")
        prof = skill.get("proficiency", "intermediate")
        dur = skill.get("duration_months", 0)
        endorse = skill.get("endorsements", 0)

        category = classify_skill(name)
        relevance = get_category_relevance(category)
        prof_weight = PROFICIENCY_WEIGHT.get(prof, 0.3)

        # Trust multiplier: endorsements + duration
        # Higher duration + endorsements = more trustworthy claim
        trust = _trust_multiplier(dur, endorse)

        skill_value = relevance * prof_weight * trust
        weighted_skill_score += skill_value
        max_possible += 1.0  # If every skill were perfect

        if relevance >= 0.8:
            core_ai_skills += 1
        if relevance >= 0.4:
            relevant_skills += 1

    # Normalize
    if max_possible > 0:
        skill_score = min(weighted_skill_score / max(max_possible * 0.3, 1), 1.0)
    else:
        skill_score = 0.0

    sig["core_ai_skill_count"] = core_ai_skills
    sig["relevant_skill_count"] = relevant_skills
    sig["skill_count"] = len(skills)

    # ── Assessment scores boost ───────────────────────────────────────────
    assessments = signals.get("skill_assessment_scores", {})
    if assessments:
        avg_assessment = sum(assessments.values()) / len(assessments)
        assessment_boost = (avg_assessment / 100) * 0.1
        skill_score += assessment_boost
        sig["avg_assessment_score"] = round(avg_assessment, 1)

    # ── Certification boost ───────────────────────────────────────────────
    relevant_certs = _count_relevant_certifications(certs)
    if relevant_certs > 0:
        skill_score += min(relevant_certs * 0.03, 0.1)
        sig["relevant_certs"] = relevant_certs

    # Final: weighted combo of cluster coverage and skill depth
    final_skills = cluster_coverage * 0.4 + skill_score * 0.6

    return final_skills, sig


def _trust_multiplier(duration_months: int, endorsements: int) -> float:
    """
    Trust multiplier based on skill duration and endorsements.
    Returns [0.3, 1.5] range.
    """
    # Duration component: log scale, maxes out around 48 months
    dur_factor = min(math.log1p(duration_months) / math.log1p(48), 1.0)

    # Endorsement component: log scale, maxes out around 30
    end_factor = min(math.log1p(endorsements) / math.log1p(30), 1.0)

    # Combined: 70% duration, 30% endorsements
    trust = 0.3 + 1.2 * (0.7 * dur_factor + 0.3 * end_factor)
    return trust


def _count_relevant_certifications(certs: list) -> int:
    """Count AI/ML-relevant certifications."""
    relevant_keywords = [
        "aws", "gcp", "azure", "cloud",
        "machine learning", "deep learning", "ai",
        "data science", "tensorflow", "pytorch",
        "kubernetes", "docker",
    ]
    count = 0
    for cert in certs:
        name = cert.get("name", "").lower()
        if any(kw in name for kw in relevant_keywords):
            count += 1
    return count


# ═══════════════════════════════════════════════════════════════════════════
# Dimension 3: Career Quality (0.15)
# ═══════════════════════════════════════════════════════════════════════════

def _score_career(career: list, profile: dict, jd: JDRequirements) -> tuple[float, dict]:
    """
    Assess career quality: product-company experience, tenure stability,
    industry progression, and hands-on coding signals.
    """
    sig = {}
    if not career:
        return 0.0, sig

    n_roles = len(career)

    # ── Company quality: product vs consulting ────────────────────────────
    consulting_count = sum(
        1 for job in career if is_consulting_company(job.get("company", ""))
    )
    product_count = n_roles - consulting_count
    company_quality = product_count / max(n_roles, 1)
    sig["consulting_roles"] = consulting_count
    sig["product_roles"] = product_count

    # ── Tenure stability ──────────────────────────────────────────────────
    # Penalize job-hopping: avg tenure < 18 months is concerning
    avg_tenure = sum(
        job.get("duration_months", 0) for job in career
    ) / max(n_roles, 1)

    if avg_tenure >= 30:
        tenure_score = 1.0
    elif avg_tenure >= 24:
        tenure_score = 0.85
    elif avg_tenure >= 18:
        tenure_score = 0.6
    elif avg_tenure >= 12:
        tenure_score = 0.3
    else:
        tenure_score = 0.1
    sig["avg_tenure_months"] = round(avg_tenure, 1)

    # ── Career progression: moving toward more senior/AI roles ────────────
    progression_score = 0.0
    if n_roles >= 2:
        # Compare recent role vs older roles
        recent_title_score = get_title_relevance(
            career[0].get("title", "")
        )
        oldest_title_score = get_title_relevance(
            career[-1].get("title", "")
        )
        if recent_title_score > oldest_title_score:
            progression_score = 0.8
        elif recent_title_score == oldest_title_score:
            progression_score = 0.5
        else:
            progression_score = 0.2  # Moving away from AI

    # ── Hands-on coding signal from descriptions ─────────────────────────
    hands_on_score = 0.0
    for job in career[:2]:  # Check recent roles
        desc = job.get("description", "").lower()
        coding_signals = [
            "implemented", "built", "developed", "designed", "wrote",
            "deployed", "shipped", "coded", "engineered", "architected",
            "pipeline", "system", "infrastructure", "api", "service",
        ]
        if sum(1 for s in coding_signals if s in desc) >= 3:
            hands_on_score = 1.0
            break
        elif sum(1 for s in coding_signals if s in desc) >= 1:
            hands_on_score = 0.6

    # Combine
    final = (
        company_quality * 0.30 +
        tenure_score * 0.25 +
        progression_score * 0.20 +
        hands_on_score * 0.25
    )

    return final, sig


# ═══════════════════════════════════════════════════════════════════════════
# Dimension 4: Education (0.05)
# ═══════════════════════════════════════════════════════════════════════════

def _score_education(education: list, jd: JDRequirements) -> tuple[float, dict]:
    """Score education quality — tier, field relevance."""
    sig = {}
    if not education:
        return 0.5, sig  # No education info → neutral

    best_score = 0.0
    for edu in education:
        tier = edu.get("tier", "unknown")
        field = edu.get("field_of_study", "").lower()
        degree = edu.get("degree", "").lower()

        # Tier scoring
        tier_scores = {
            "tier_1": 1.0,
            "tier_2": 0.75,
            "tier_3": 0.5,
            "tier_4": 0.3,
            "unknown": 0.4,
        }
        tier_score = tier_scores.get(tier, 0.3)

        # Field relevance
        relevant_fields = [
            "computer science", "machine learning", "artificial intelligence",
            "data science", "information technology", "electrical",
            "electronics", "mathematics", "statistics",
            "computational", "software",
        ]
        irrelevant_fields = [
            "chemical", "civil", "mechanical", "commerce",
            "arts", "humanities", "biology", "agriculture",
        ]

        if any(f in field for f in relevant_fields):
            field_score = 1.0
        elif any(f in field for f in irrelevant_fields):
            field_score = 0.2
        else:
            field_score = 0.5

        # Degree level boost
        degree_boost = 0.0
        if "ph.d" in degree or "phd" in degree:
            degree_boost = 0.15
        elif "m.tech" in degree or "m.sc" in degree or "m.e." in degree or "ms" in degree:
            degree_boost = 0.1

        edu_score = tier_score * 0.5 + field_score * 0.4 + degree_boost
        best_score = max(best_score, edu_score)

        sig["best_edu_tier"] = tier
        sig["edu_field"] = edu.get("field_of_study", "")

    return min(best_score, 1.0), sig


# ═══════════════════════════════════════════════════════════════════════════
# Dimension 5: Behavioral Signals (0.15)
# ═══════════════════════════════════════════════════════════════════════════

def _score_behavioral(signals: dict, profile: dict, jd: JDRequirements) -> tuple[float, dict]:
    """
    Score platform behavioral signals — availability, responsiveness,
    engagement, and hiring-readiness.
    """
    sig = {}

    # ── Open to work (binary boost) ───────────────────────────────────────
    open_to_work = signals.get("open_to_work_flag", False)
    otw_score = 1.0 if open_to_work else 0.3
    sig["open_to_work"] = open_to_work

    # ── Recency: last active date ─────────────────────────────────────────
    last_active = signals.get("last_active_date", "")
    recency_score = _recency_score(last_active)
    sig["last_active"] = last_active

    # ── Recruiter response rate ───────────────────────────────────────────
    response_rate = signals.get("recruiter_response_rate", 0)
    resp_score = min(response_rate / 0.7, 1.0)  # 0.7+ is great
    sig["response_rate"] = response_rate

    # ── Notice period ─────────────────────────────────────────────────────
    notice_days = signals.get("notice_period_days", 90)
    if notice_days <= jd.ideal_notice_days:
        notice_score = 1.0
    elif notice_days <= jd.max_preferred_notice:
        notice_score = 0.7
    elif notice_days <= jd.notice_penalty_threshold:
        notice_score = 0.4
    else:
        notice_score = 0.15
    sig["notice_days"] = notice_days

    # ── Profile completeness ──────────────────────────────────────────────
    completeness = signals.get("profile_completeness_score", 0)
    completeness_score = min(completeness / 80, 1.0)  # 80+ is good

    # ── GitHub activity ───────────────────────────────────────────────────
    github = signals.get("github_activity_score", -1)
    if github < 0:
        github_score = 0.3  # No GitHub — neutral
    else:
        github_score = min(github / 60, 1.0)  # 60+ is strong
    sig["github_score"] = github if github >= 0 else "N/A"

    # ── Interview & offer history ─────────────────────────────────────────
    interview_rate = signals.get("interview_completion_rate", 0)
    offer_rate = signals.get("offer_acceptance_rate", -1)

    interview_score = interview_rate  # Already [0, 1]
    offer_score = max(offer_rate, 0) if offer_rate >= 0 else 0.5  # Neutral if no data

    # ── Verification boost ────────────────────────────────────────────────
    verified_email = signals.get("verified_email", False)
    verified_phone = signals.get("verified_phone", False)
    linkedin = signals.get("linkedin_connected", False)
    verification_score = sum([
        0.4 if verified_email else 0,
        0.3 if verified_phone else 0,
        0.3 if linkedin else 0,
    ])

    # ── Location fit ──────────────────────────────────────────────────────
    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    work_mode = signals.get("preferred_work_mode", "")
    relocate = signals.get("willing_to_relocate", False)

    location_score = _location_fit_score(location, country, work_mode, relocate, jd)
    sig["location"] = profile.get("location", "")
    sig["country"] = profile.get("country", "")

    # ── Speed of Communication ────────────────────────────────────────────
    resp_time = signals.get("avg_response_time_hours", -1)
    if resp_time < 0:
        speed_score = 0.5
    elif resp_time <= 24:
        speed_score = 1.0
    elif resp_time <= 48:
        speed_score = 0.8
    elif resp_time <= 120:
        speed_score = 0.5
    else:
        speed_score = 0.2
    sig["response_speed_hours"] = resp_time if resp_time >= 0 else "N/A"

    # ── Market Demand / Social Proof ──────────────────────────────────────
    saved = signals.get("saved_by_recruiters_30d", 0)
    views = signals.get("profile_views_received_30d", 0)
    searches = signals.get("search_appearance_30d", 0)
    market_score = (views / 50.0 + searches / 100.0 + saved / 10.0) / 3.0
    demand_score = min(market_score, 1.0)
    sig["market_demand_score"] = round(demand_score, 2)

    # ── Active Job Hunting Volume ─────────────────────────────────────────
    apps = signals.get("applications_submitted_30d", -1)
    if apps < 0:
        app_score = 0.5
    elif apps == 0:
        app_score = 0.4
    elif apps <= 50:
        app_score = 1.0
    elif apps <= 100:
        app_score = 0.7
    else:
        app_score = 0.3
    sig["applications_30d"] = apps if apps >= 0 else "N/A"

    # Combine with weights
    final = (
        otw_score * 0.09 +            # Was 0.12 (-0.03 for apps)
        recency_score * 0.12 +
        resp_score * 0.12 +           # Was 0.18 (-0.06 for speed)
        speed_score * 0.06 +
        notice_score * 0.10 +
        completeness_score * 0.05 +
        github_score * 0.10 +
        interview_score * 0.05 +
        offer_score * 0.03 +
        verification_score * 0.05 +
        location_score * 0.15 +
        demand_score * 0.05 +         # Replaces saved_score
        app_score * 0.03
    )

    return final, sig


def _recency_score(last_active_str: str) -> float:
    """Score based on how recently the candidate was active."""
    if not last_active_str:
        return 0.2
    try:
        last_active = datetime.strptime(last_active_str, "%Y-%m-%d").date()
        # Reference date: ~June 2026 (dataset context)
        reference = date(2026, 6, 1)
        days_ago = (reference - last_active).days
        if days_ago <= 30:
            return 1.0
        elif days_ago <= 90:
            return 0.8
        elif days_ago <= 180:
            return 0.5
        elif days_ago <= 365:
            return 0.2
        else:
            return 0.05
    except (ValueError, TypeError):
        return 0.2


def _location_fit_score(
    location: str, country: str, work_mode: str,
    relocate: bool, jd: JDRequirements
) -> float:
    """Score location fit."""
    # Country check first
    if country not in jd.preferred_countries and country:
        # International — mild penalty unless willing to relocate
        if relocate:
            return 0.4
        return 0.2

    # Check if location matches preferred cities
    for loc in jd.preferred_locations:
        if re.search(rf"\b{re.escape(loc)}\b", location):
            return 1.0

    # In India but not preferred city
    if country in jd.preferred_countries:
        if relocate:
            return 0.7
        return 0.5

    return 0.3


# ═══════════════════════════════════════════════════════════════════════════
# Dimension 6: Red Flags (penalty dimension)
# ═══════════════════════════════════════════════════════════════════════════

def _score_red_flags(
    profile: dict, career: list, skills: list,
    signals: dict, jd: JDRequirements
) -> tuple[float, dict]:
    """
    Detect red flags and return a penalty multiplier.
    1.0 = no flags, 0.0 = severe (essentially disqualified).
    """
    sig = {}
    penalty = 1.0  # Start with no penalty

    # ── Red Flag 1: Pure consulting career ────────────────────────────────
    if career:
        all_consulting = all(
            is_consulting_company(job.get("company", ""))
            for job in career
        )
        if all_consulting and len(career) >= 2:
            penalty *= jd.disqualifier_weights["pure_consulting_career"]
            sig["flag_pure_consulting"] = True

    # ── Red Flag 2: Keyword stuffer (non-tech title + AI skill dump) ──────
    current_title = profile.get("current_title", "")
    if is_non_technical_title(current_title):
        # Count AI-relevant skills
        ai_skill_count = sum(
            1 for s in skills
            if classify_skill(s.get("name", "")) in (
                "core_ai_ml", "vector_db_retrieval", "ml_foundations"
            )
        )
        if ai_skill_count >= 5:
            # Non-technical title with 5+ AI skills = keyword stuffer
            penalty *= jd.disqualifier_weights["keyword_stuffer"]
            sig["flag_keyword_stuffer"] = True
            sig["flag_stuffer_detail"] = (
                f"Title '{current_title}' with {ai_skill_count} AI skills"
            )
        elif ai_skill_count >= 3:
            penalty *= 0.5  # Mild penalty

    # ── Red Flag 3: Title chaser (many companies, short tenure) ───────────
    if len(career) >= 3:
        short_stints = sum(
            1 for job in career
            if job.get("duration_months", 0) <= 18 and not job.get("is_current", False)
        )
        if short_stints >= 3:
            penalty *= jd.disqualifier_weights["title_chaser"]
            sig["flag_title_chaser"] = True

    # ── Red Flag 4: Domain mismatch (CV/speech without NLP) ───────────────
    has_cv_speech = False
    has_nlp_ir = False
    for skill in skills:
        cat = classify_skill(skill.get("name", ""))
        if cat in ("computer_vision", "speech_audio"):
            has_cv_speech = True
        if cat in ("core_ai_ml", "vector_db_retrieval", "ml_foundations", "deep_learning"):
            has_nlp_ir = True

    if has_cv_speech and not has_nlp_ir:
        # Only CV/speech skills, no NLP/IR — JD explicitly warns
        penalty *= jd.disqualifier_weights["domain_mismatch"]
        sig["flag_domain_mismatch"] = True

    # ── Red Flag 5: Very low responsiveness (basically unavailable) ───────
    response_rate = signals.get("recruiter_response_rate", 0)
    if response_rate < 0.1:
        penalty *= 0.6
        sig["flag_low_response"] = True

    # ── Red Flag 6: Very long inactive period ─────────────────────────────
    last_active = signals.get("last_active_date", "")
    if last_active:
        try:
            last = datetime.strptime(last_active, "%Y-%m-%d").date()
            days_inactive = (date(2026, 6, 1) - last).days
            if days_inactive > 365:
                penalty *= 0.5
                sig["flag_long_inactive"] = True
        except (ValueError, TypeError):
            pass

    # ── Red Flag 7: Salary expectation mismatch ───────────────────────────
    expected_salary = signals.get("expected_salary_range_inr_lpa", {})
    if isinstance(expected_salary, dict):
        min_expected = expected_salary.get("min", 0.0)
        if min_expected > jd.salary_max_lpa:
            penalty *= 0.3
            sig["flag_budget_mismatch"] = True
            sig["budget_mismatch_detail"] = f"Min {min_expected}LPA > max {jd.salary_max_lpa}LPA"

    return penalty, sig


# ── Utility ───────────────────────────────────────────────────────────────

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))
