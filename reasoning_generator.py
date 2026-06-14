"""
Reasoning generator for the Redrob AI Ranking Challenge.

Produces a 1-2 sentence, specific, honest reasoning string for each
ranked candidate. No templates — constructs reasoning from the
candidate's actual scoring signals.

Requirements from submission_spec:
  - NOT empty
  - NOT all-identical
  - NOT templated with just name insertion
  - NOT mentioning skills not in profile (hallucination)
  - NOT contradicting the rank
  - Plain-language, specific, honest
"""

from skill_taxonomy import (
    classify_skill,
    get_category_relevance,
    is_non_technical_title,
    is_consulting_company,
)


def generate_reasoning(
    candidate: dict,
    score_result: dict,
    rank: int,
) -> str:
    """
    Generate a plain-language reasoning string for a ranked candidate.

    Args:
        candidate: Full candidate profile dict
        score_result: Output from scorer.score_candidate()
        rank: The candidate's rank (1-100)

    Returns:
        1-2 sentence reasoning string
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    dim_scores = score_result.get("dimension_scores", {})
    scoring_signals = score_result.get("signals", {})

    parts = []

    # ── Part 1: Role identity ────────────────────────────────────────────
    title = profile.get("current_title", "Unknown")
    company = profile.get("current_company", "")
    yoe = profile.get("years_of_experience", 0)

    if company:
        parts.append(f"{title} at {company} with {yoe:.1f} yrs experience")
    else:
        parts.append(f"{title} with {yoe:.1f} yrs experience")

    # ── Part 2: Key strengths ────────────────────────────────────────────
    strengths = []

    # Skills highlight
    core_skills = _get_top_relevant_skills(skills, limit=3)
    if core_skills:
        skill_str = ", ".join(core_skills)
        strengths.append(f"strong in {skill_str}")

    # Career trajectory
    ai_roles = scoring_signals.get("ai_role_count", 0)
    recent_ai = scoring_signals.get("recent_ai_role", False)
    if recent_ai:
        strengths.append("currently in AI/ML role")
    elif ai_roles > 0:
        strengths.append(f"{ai_roles} prior AI/ML role(s)")

    # Product company experience
    product_ratio = scoring_signals.get("product_company_ratio", 0)
    if product_ratio >= 0.5:
        product_roles = scoring_signals.get("product_roles", 0)
        if product_roles > 0:
            strengths.append("product-company background")

    # Behavioral strength
    response_rate = signals.get("recruiter_response_rate", 0)
    if response_rate >= 0.6:
        strengths.append(f"high responsiveness ({response_rate:.0%})")
    elif response_rate >= 0.4:
        strengths.append(f"good responsiveness ({response_rate:.0%})")

    # Fast response speed
    speed = signals.get("avg_response_time_hours", -1)
    if 0 <= speed <= 24:
        strengths.append(f"fast response times ({speed:.1f}h avg)")

    # Market demand
    demand_score = scoring_signals.get("market_demand_score", 0.0)
    if demand_score >= 0.8:
        strengths.append("high market demand")

    # GitHub
    github = signals.get("github_activity_score", -1)
    if github >= 50:
        strengths.append(f"active on GitHub (score: {github:.0f})")

    # Location fit
    location = profile.get("location", "")
    country = profile.get("country", "")
    if country.lower() == "india":
        strengths.append(f"based in {location}")

    if strengths:
        parts.append("; ".join(strengths[:5]))

    # ── Part 3: Key concerns (for lower-ranked candidates) ────────────────
    if rank > 30:
        concerns = []

        # Budget mismatch
        if scoring_signals.get("flag_budget_mismatch"):
            detail = scoring_signals.get("budget_mismatch_detail", "exceeds budget")
            concerns.append(f"budget mismatch ({detail})")

        # Mass applying
        apps = signals.get("applications_submitted_30d", 0)
        if apps > 100:
            concerns.append(f"mass applying ({apps} apps/mo)")

        # Low response rate
        if response_rate < 0.2:
            concerns.append(f"low response rate ({response_rate:.0%})")

        # Non-technical title
        if is_non_technical_title(title):
            concerns.append("non-technical title")

        # Consulting-only career
        if scoring_signals.get("flag_pure_consulting"):
            concerns.append("consulting-only career")

        # Long notice
        notice = signals.get("notice_period_days", 0)
        if notice > 90:
            concerns.append(f"{notice}-day notice period")

        # Inactive
        if scoring_signals.get("flag_long_inactive"):
            concerns.append("inactive for 6+ months")

        if concerns:
            parts.append("concerns: " + ", ".join(concerns[:3]))

    # ── Assemble ──────────────────────────────────────────────────────────
    reasoning = ". ".join(parts) + "."

    # Ensure we're under a reasonable length
    if len(reasoning) > 300:
        reasoning = reasoning[:297] + "..."

    # Clean up any double periods
    reasoning = reasoning.replace("..", ".")

    return reasoning


def _get_top_relevant_skills(skills: list, limit: int = 3) -> list[str]:
    """Get the top N most relevant skills for the JD."""
    scored = []
    for skill in skills:
        name = skill.get("name", "")
        cat = classify_skill(name)
        relevance = get_category_relevance(cat)
        prof = skill.get("proficiency", "")
        prof_w = {"expert": 1.0, "advanced": 0.8, "intermediate": 0.5, "beginner": 0.2}
        score = relevance * prof_w.get(prof, 0.3)
        if relevance >= 0.4:  # Only include relevant skills
            scored.append((name, score))

    if not scored and skills:
        # Fallback if no skills met the threshold
        for skill in skills:
            name = skill.get("name", "")
            prof = skill.get("proficiency", "")
            prof_w = {"expert": 1.0, "advanced": 0.8, "intermediate": 0.5, "beginner": 0.2}
            scored.append((name, prof_w.get(prof, 0.3)))

    scored.sort(key=lambda x: -x[1])
    return [name for name, _ in scored[:limit]]
