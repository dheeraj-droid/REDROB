"""
WorthyHire — Candidate Profile Parser

Normalizes raw candidate profiles for consistent processing.
"""


def normalize_candidate(candidate: dict) -> dict:
    """
    Normalize a raw candidate dict for consistent downstream processing.

    - Ensures all expected keys exist with sensible defaults
    - Lowercases location/country for matching
    - Strips whitespace from skill names
    - Ensures numeric fields are proper types
    """
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    education = candidate.get("education", [])
    signals = candidate.get("redrob_signals", {})

    # Normalize profile
    profile.setdefault("current_title", "")
    profile.setdefault("headline", "")
    profile.setdefault("summary", "")
    profile.setdefault("location", "")
    profile.setdefault("country", "")
    profile.setdefault("years_of_experience", 0)
    profile.setdefault("anonymized_name", "Unknown")
    profile.setdefault("current_company", "Unknown")

    # Ensure numeric
    try:
        profile["years_of_experience"] = float(profile["years_of_experience"])
    except (ValueError, TypeError):
        profile["years_of_experience"] = 0.0

    # Normalize skills
    for s in skills:
        s["name"] = s.get("name", "").strip()
        s.setdefault("proficiency", "")
        s.setdefault("duration_months", 0)
        s.setdefault("endorsements", 0)

    # Normalize career
    for job in career:
        job.setdefault("title", "")
        job.setdefault("company", "")
        job.setdefault("description", "")
        job.setdefault("duration_months", 0)
        job.setdefault("is_current", False)

    # Normalize signals
    signals.setdefault("recruiter_response_rate", 0)
    signals.setdefault("notice_period_days", 0)
    signals.setdefault("profile_completeness_score", 0)
    signals.setdefault("response_speed_hours", None)

    candidate["profile"] = profile
    candidate["skills"] = skills
    candidate["career_history"] = career
    candidate["education"] = education
    candidate["redrob_signals"] = signals

    return candidate
