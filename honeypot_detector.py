"""
Honeypot / trap candidate detection for the Redrob AI Ranking Challenge.

The dataset contains ~80 honeypot candidates with subtly impossible profiles.
Submissions with honeypot rate > 10% in top 100 are disqualified.

Detection heuristics:
1. Impossible experience duration vs. career history
2. Expert proficiency with zero duration or endorsements
3. Excessive skill count with uniformly low backing
4. Timeline inconsistencies
5. Title-skill coherence failures
"""

from datetime import datetime, date
from typing import Any


def detect_honeypot(candidate: dict) -> tuple[bool, list[str]]:
    """
    Analyze a candidate profile for honeypot signals.

    Returns:
        (is_honeypot: bool, reasons: list[str])
    """
    flags = []
    score = 0.0  # Accumulate suspicion score; threshold = 3.0

    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])
    signals = candidate.get("redrob_signals", {})

    # ── Check 1: Experience vs. career history mismatch ───────────────────
    claimed_yoe = profile.get("years_of_experience", 0)
    total_career_months = sum(
        job.get("duration_months", 0) for job in career
    )
    total_career_years = total_career_months / 12.0

    # If claimed YoE is much larger than career history supports
    if claimed_yoe > 0 and total_career_years > 0:
        ratio = claimed_yoe / total_career_years
        if ratio > 2.5:  # Claims 2.5x more than career shows
            flags.append(
                f"YoE mismatch: claims {claimed_yoe:.1f}y but career "
                f"history totals {total_career_years:.1f}y"
            )
            score += 1.5
        # Inverse: career totals far exceed claimed YoE (overlaps are normal)
        if total_career_years > claimed_yoe * 2 and claimed_yoe < 3:
            flags.append(
                f"Career exceeds claimed YoE: {total_career_years:.1f}y "
                f"in history vs {claimed_yoe:.1f}y claimed"
            )
            score += 0.5

    # ── Check 2: Expert proficiency with zero/very low duration ───────────
    expert_zero_duration = 0
    expert_zero_endorsements = 0
    for skill in skills:
        prof = skill.get("proficiency", "")
        dur = skill.get("duration_months", 0)
        end = skill.get("endorsements", 0)

        if prof == "expert":
            if dur == 0:
                expert_zero_duration += 1
            if end == 0:
                expert_zero_endorsements += 1
        elif prof == "advanced":
            if dur == 0:
                expert_zero_duration += 0.5

    if expert_zero_duration >= 3:
        flags.append(
            f"Expert/advanced proficiency with 0 months in "
            f"{int(expert_zero_duration)} skills"
        )
        score += 2.0
    elif expert_zero_duration >= 2:
        score += 1.0

    if expert_zero_endorsements >= 5:
        flags.append(
            f"Expert proficiency with 0 endorsements in "
            f"{expert_zero_endorsements} skills"
        )
        score += 1.5

    # ── Check 3: Excessive skill count with low backing ───────────────────
    n_skills = len(skills)
    if n_skills > 0:
        avg_endorsements = sum(
            s.get("endorsements", 0) for s in skills
        ) / n_skills
        avg_duration = sum(
            s.get("duration_months", 0) for s in skills
        ) / n_skills

        # Many skills but almost no endorsements
        if n_skills >= 15 and avg_endorsements < 2:
            flags.append(
                f"{n_skills} skills with avg {avg_endorsements:.1f} endorsements"
            )
            score += 1.5

        # Many expert skills — suspicious
        expert_count = sum(
            1 for s in skills if s.get("proficiency") == "expert"
        )
        if expert_count >= 10:
            flags.append(f"{expert_count} 'expert'-level skills")
            score += 2.0
        elif expert_count >= 7:
            score += 1.0

    # ── Check 4: Career timeline impossibilities ──────────────────────────
    for job in career:
        start_str = job.get("start_date", "")
        end_str = job.get("end_date")
        duration = job.get("duration_months", 0)

        if start_str and end_str:
            try:
                start = _parse_date(start_str)
                end = _parse_date(end_str)

                if end < start:
                    flags.append(
                        f"End date before start date at {job.get('company', '?')}"
                    )
                    score += 2.0

                # Check if stated duration is wildly off from dates
                actual_months = (end.year - start.year) * 12 + (end.month - start.month)
                if duration > 0 and abs(actual_months - duration) > 12:
                    flags.append(
                        f"Duration mismatch at {job.get('company', '?')}: "
                        f"dates suggest {actual_months}m, claims {duration}m"
                    )
                    score += 1.5
            except (ValueError, TypeError):
                pass

        # Company founded recently but person claims many years there
        if duration > 96:  # 8+ years at a single company
            # Check if start date is suspiciously old for a startup
            if start_str:
                try:
                    start = _parse_date(start_str)
                    company_name = job.get("company", "").lower()
                    company_size = job.get("company_size", "")
                    # Small startup with 8+ years of tenure starting recently
                    if company_size in ("1-10", "11-50") and start.year >= 2022:
                        if duration > 48:
                            flags.append(
                                f"Claims {duration}m at small company "
                                f"{job.get('company')} starting {start_str}"
                            )
                            score += 1.5
                except (ValueError, TypeError):
                    pass

    # ── Check 5: Education impossibilities ────────────────────────────────
    for edu in education:
        start_y = edu.get("start_year", 0)
        end_y = edu.get("end_year", 0)
        if start_y and end_y:
            if end_y < start_y:
                flags.append(
                    f"Education end year ({end_y}) before start ({start_y})"
                )
                score += 2.0
            duration_years = end_y - start_y
            degree = edu.get("degree", "").lower()
            # PhD in 1 year or B.Tech in 1 year
            if "ph.d" in degree or "phd" in degree:
                if duration_years < 2:
                    flags.append(
                        f"PhD completed in {duration_years} year(s)"
                    )
                    score += 1.5
            elif duration_years > 8:
                flags.append(
                    f"Degree took {duration_years} years"
                )
                score += 1.0

    # ── Check 6: Signal impossibilities ───────────────────────────────────
    # Profile completeness is very high but many verified flags are false
    completeness = signals.get("profile_completeness_score", 0)
    verified_email = signals.get("verified_email", False)
    verified_phone = signals.get("verified_phone", False)
    linkedin = signals.get("linkedin_connected", False)

    # Very high completeness but nothing verified
    if completeness > 90 and not verified_email and not verified_phone and not linkedin:
        flags.append(
            f"Profile completeness {completeness} but no verifications"
        )
        score += 0.5

    # Assessment scores for skills the candidate doesn't have
    assessments = signals.get("skill_assessment_scores", {})
    candidate_skill_names = {s.get("name", "").lower() for s in skills}
    for assessed_skill in assessments:
        if assessed_skill.lower() not in candidate_skill_names:
            # Some tolerance — assessment names might not exactly match
            pass

    # ── Check 7: Title-description mismatch ───────────────────────────────
    # If current title is "AI Engineer" but all career descriptions
    # talk about accounting or marketing — suspicious
    current_title = profile.get("current_title", "").lower()
    if career:
        current_job = career[0] if career[0].get("is_current") else career[0]
        description = current_job.get("description", "").lower()

        # AI/ML title but description talks about non-tech work
        ai_keywords = {"ml", "ai", "machine learning", "model", "embedding",
                       "neural", "deep learning", "nlp", "transformer",
                       "data science", "algorithm", "pipeline"}
        non_tech_keywords = {"marketing", "accounting", "hr", "recruitment",
                             "supply chain", "logistics", "warehouse",
                             "brand", "packaging", "sales", "customer support",
                             "support agents", "tickets"}

        desc_has_ai = any(kw in description for kw in ai_keywords)
        desc_has_non_tech = any(kw in description for kw in non_tech_keywords)

        if ("ai" in current_title or "ml" in current_title or
                "machine learning" in current_title or "data scien" in current_title):
            if desc_has_non_tech and not desc_has_ai:
                flags.append(
                    f"Title '{profile.get('current_title')}' but description "
                    f"is non-technical"
                )
                score += 1.5

    is_honeypot = score >= 2.0
    return is_honeypot, flags


def _parse_date(date_str: str) -> date:
    """Parse a date string in YYYY-MM-DD format."""
    if isinstance(date_str, date):
        return date_str
    return datetime.strptime(date_str, "%Y-%m-%d").date()
