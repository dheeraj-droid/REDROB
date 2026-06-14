"""
Honeypot / trap candidate detection for the WorthyHire Ranking System.

The dataset contains ~80 honeypot candidates with subtly impossible profiles.
Submissions with honeypot rate > 10% in top 100 are disqualified.

Detection strategy (10 checks, ordered by reliability):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Summary-title contradiction (summary says "marketing manager", title is "Civil Engineer")
  2. Title-description domain clash (title says one thing, job description says completely different)
  3. Self-learner contradiction (summary says "self-learner" but skills say "advanced")
  4. Career description incoherence (descriptions from completely different domains)
  5. Skill-career domain mismatch (advanced AI skills, zero AI in career history)
  6. Non-tech title + advanced AI skills
  7. Impossible experience duration vs career
  8. Expert/advanced proficiency with zero duration
  9. Timeline impossibilities
  10. Skill duration exceeds career scope
"""

from datetime import datetime, date
from typing import Any

from skill_taxonomy import classify_skill


# ── Domain keyword sets for cross-checking ────────────────────────────────

_DOMAIN_KEYWORDS = {
    "brand_design": {
        "brand", "branding", "logo", "typography", "packaging", "creative direction",
        "visual system", "rebrand", "adobe suite", "figma", "design system",
    },
    "marketing": {
        "marketing", "demand-generation", "content marketing", "paid acquisition",
        "seo", "email nurture", "account-based marketing", "campaign",
    },
    "sales": {
        "sales", "quota", "revenue", "prospecting", "discovery", "negotiation",
        "arr", "consultative selling", "pipeline",
    },
    "support": {
        "customer support", "support agents", "tickets", "escalation",
        "tier-1", "tier-2", "knowledge base", "support team",
    },
    "finance": {
        "accounting", "financial reporting", "statutory compliance", "gaap",
        "ind-as", "tax filings", "general ledger", "audit", "fixed-asset",
        "month-end close",
    },
    "hr": {
        "hr", "recruitment", "talent acquisition", "onboarding", "employee",
        "people-management", "agent training",
    },
    "operations": {
        "operations management", "logistics", "warehouse", "fulfillment",
        "receiving", "picking", "packing", "outbound", "cost per order",
        "continuous improvement",
    },
    "mechanical_eng": {
        "mechanical engineering", "cad", "solidworks", "creo", "fea", "ansys",
        "dfm", "dfma", "prototype", "production tooling", "hardware",
    },
    "consulting": {
        "business diagnostics", "process re-engineering", "digital transformation",
        "strategy projects", "stakeholder management", "slide-craft",
        "excel modeling", "executive communication", "consulting",
    },
    "tech_ai_ml": {
        "ml", "ai", "machine learning", "model", "embedding", "neural",
        "deep learning", "nlp", "transformer", "data science", "algorithm",
        "feature pipeline", "vector", "search", "ranking", "recommendation",
        "fine-tuning", "llm", "training", "inference", "gpu",
    },
    "tech_software": {
        "backend", "frontend", "api", "microservice", "database", "react",
        "python", "java", "kotlin", "typescript", "node", "django", "flask",
        "fastapi", "docker", "kubernetes", "ci/cd", "devops",
        "streaming", "kafka", "spark", "airflow", "pipeline", "data pipeline",
        "test automation", "selenium", "pytest", "mobile development",
        "android", "ios", "webpack", "redux",
    },
}

# AI/ML skill categories from taxonomy
_AI_SKILL_CATS = {
    "core_ai_ml", "ml_foundations", "vector_db_retrieval", "mlops",
    "computer_vision",
}

# Non-technical current titles
_NON_TECH_TITLES = {
    "civil engineer", "mechanical engineer", "marketing manager",
    "operations manager", "accountant", "hr manager",
    "content writer", "graphic designer", "customer support",
    "project manager", "business analyst", "supply chain manager",
    "procurement manager", "administrative assistant", "office manager",
    "sales executive",
}


def detect_honeypot(candidate: dict) -> tuple[bool, list[str]]:
    """
    Analyze a candidate profile for honeypot signals.

    Returns:
        (is_honeypot: bool, reasons: list[str])
    """
    flags = []
    score = 0.0  # Accumulate suspicion score; threshold = 2.0

    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])
    signals = candidate.get("redrob_signals", {})

    current_title = profile.get("current_title", "").lower().strip()
    summary = profile.get("summary", "").lower()
    headline = profile.get("headline", "").lower()

    # ── Pre-compute: advanced AI skills ───────────────────────────────────
    advanced_ai_skills = []
    for skill in skills:
        prof = skill.get("proficiency", "")
        name = skill.get("name", "")
        if prof in ("advanced", "expert"):
            cat = classify_skill(name)
            if cat in _AI_SKILL_CATS:
                advanced_ai_skills.append(name)

    # ── Pre-compute: career domain detection ──────────────────────────────
    all_descriptions = " ".join(
        job.get("description", "").lower() for job in career
    )

    desc_domains = _detect_domains(all_descriptions)
    has_tech_career = ("tech_ai_ml" in desc_domains or "tech_software" in desc_domains)

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 1: Summary-title contradiction
    # If the summary says "my background is in marketing manager" but the
    # title is "Civil Engineer" — that's a dead giveaway of shuffled data.
    # ══════════════════════════════════════════════════════════════════════
    summary_role_claims = []
    for role_phrase in [
        "marketing manager", "operations manager", "hr manager",
        "project manager", "business analyst", "accountant",
        "sales executive", "customer support",
    ]:
        if f"background is in {role_phrase}" in summary or \
           f"career in {role_phrase}" in summary or \
           f"spent my career in {role_phrase}" in summary:
            summary_role_claims.append(role_phrase)

    if summary_role_claims:
        for claimed_role in summary_role_claims:
            # If summary claims a role but title is different
            if claimed_role not in current_title:
                flags.append(
                    f"Summary claims background in '{claimed_role}' "
                    f"but title is '{profile.get('current_title')}'"
                )
                score += 2.0

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 2: Title-description domain clash
    # If title is "Civil Engineer" but the job description talks about
    # "brand design" or "customer support" — the profile is incoherent.
    # ══════════════════════════════════════════════════════════════════════
    if career:
        current_job = next(
            (j for j in career if j.get("is_current")),
            career[0]
        )
        current_desc = current_job.get("description", "").lower()
        current_desc_domains = _detect_domains(current_desc)

        title_domain = _title_to_domain(current_title)
        if title_domain and current_desc_domains:
            # Check if title domain appears in description domains
            if title_domain not in current_desc_domains:
                # Title and description are from different worlds
                desc_domain_str = ", ".join(current_desc_domains)
                flags.append(
                    f"Title '{profile.get('current_title')}' ({title_domain}) "
                    f"but description is about: {desc_domain_str}"
                )
                score += 2.0

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 3: Self-learner contradiction
    # Only catch the EXACT TEMPLATE phrases used by honeypots.
    # Genuine phrases like "self-directed ML projects" should NOT trigger.
    # ══════════════════════════════════════════════════════════════════════
    template_self_learner_phrases = [
        "curious about how ai tools could augment my work",
        "experimented with chatgpt and a few other tools for productivity",
        "i think the space is exciting",
    ]
    is_template_self_learner = any(p in summary for p in template_self_learner_phrases)

    # Only flag if it's the template AND they claim advanced AI skills
    # AND they don't have a tech career
    if is_template_self_learner and len(advanced_ai_skills) >= 2 and not has_tech_career:
        flags.append(
            f"Template self-learner summary but claims advanced in: "
            f"{', '.join(advanced_ai_skills[:4])}"
        )
        score += 2.5

    # Even without advanced AI skills, the template summary is a signal
    if is_template_self_learner:
        score += 0.5  # Weak signal on its own

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 4: Career description incoherence
    # If career entries describe completely unrelated domains
    # (brand design → customer support → sales) — descriptions are shuffled.
    # ══════════════════════════════════════════════════════════════════════
    if len(career) >= 2:
        per_job_domains = []
        for job in career:
            desc = job.get("description", "").lower()
            domains = _detect_domains(desc)
            per_job_domains.append(domains)

        # Count unique domains across career entries
        all_career_domains = set()
        for d in per_job_domains:
            all_career_domains.update(d)

        # If career spans 4+ completely unrelated non-tech domains, suspicious
        non_tech_domains = all_career_domains - {"tech_ai_ml", "tech_software"}
        if len(non_tech_domains) >= 4:
            flags.append(
                f"Career descriptions span {len(non_tech_domains)} unrelated domains: "
                f"{', '.join(sorted(non_tech_domains)[:4])}"
            )
            score += 1.5
        elif len(non_tech_domains) >= 3:
            score += 0.75

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 5: Advanced AI skills + zero AI in career
    # If candidate claims advanced AI skills but NO career description
    # mentions any AI/ML/data science work at all.
    # ══════════════════════════════════════════════════════════════════════
    if len(advanced_ai_skills) >= 2 and not has_tech_career:
        flags.append(
            f"Claims advanced in {', '.join(advanced_ai_skills[:4])} "
            f"but no career entry mentions AI/ML/tech work"
        )
        score += 2.5

    # Even if career is "tech", check more specifically for AI mentions
    # But be lenient — many engineers build AI skills on the side legitimately
    if len(advanced_ai_skills) >= 4 and "tech_ai_ml" not in desc_domains:
        ai_desc_keywords = {
            "ml", "ai", "machine learning", "model", "embedding",
            "deep learning", "nlp", "transformer", "training",
            "inference", "vector", "recommendation", "ranking",
            "data science", "feature",
        }
        career_mentions_ai = any(
            kw in all_descriptions for kw in ai_desc_keywords
        )
        if not career_mentions_ai:
            flags.append(
                f"{len(advanced_ai_skills)} advanced AI skills but "
                f"career descriptions never mention ML/AI work"
            )
            score += 1.0  # Weaker signal for tech candidates

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 6: Non-tech title + advanced AI skills
    # ══════════════════════════════════════════════════════════════════════
    if current_title in _NON_TECH_TITLES and len(advanced_ai_skills) >= 2:
        flags.append(
            f"Non-technical title '{profile.get('current_title')}' with "
            f"{len(advanced_ai_skills)} advanced AI skills"
        )
        score += 2.0

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 7: Experience vs career history mismatch
    # ══════════════════════════════════════════════════════════════════════
    claimed_yoe = profile.get("years_of_experience", 0)
    total_career_months = sum(
        job.get("duration_months", 0) for job in career
    )
    total_career_years = total_career_months / 12.0

    if claimed_yoe > 0 and total_career_years > 0:
        ratio = claimed_yoe / total_career_years
        if ratio > 2.5:
            flags.append(
                f"YoE mismatch: claims {claimed_yoe:.1f}y but career "
                f"totals {total_career_years:.1f}y"
            )
            score += 1.5
        if total_career_years > claimed_yoe * 2 and claimed_yoe < 3:
            flags.append(
                f"Career exceeds claimed YoE: {total_career_years:.1f}y "
                f"in history vs {claimed_yoe:.1f}y claimed"
            )
            score += 0.5

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 8: Expert proficiency with zero/very low duration
    # ══════════════════════════════════════════════════════════════════════
    expert_zero_duration = 0
    for skill in skills:
        prof = skill.get("proficiency", "")
        dur = skill.get("duration_months", 0)
        if prof == "expert" and dur == 0:
            expert_zero_duration += 1
        elif prof == "advanced" and dur == 0:
            expert_zero_duration += 0.5

    if expert_zero_duration >= 3:
        flags.append(
            f"Expert/advanced proficiency with 0 months in "
            f"{int(expert_zero_duration)} skills"
        )
        score += 2.0
    elif expert_zero_duration >= 2:
        score += 1.0

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 9: Career timeline impossibilities
    # ══════════════════════════════════════════════════════════════════════
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

                actual_months = (end.year - start.year) * 12 + (end.month - start.month)
                if duration > 0 and abs(actual_months - duration) > 12:
                    flags.append(
                        f"Duration mismatch at {job.get('company', '?')}: "
                        f"dates suggest {actual_months}m, claims {duration}m"
                    )
                    score += 1.5
            except (ValueError, TypeError):
                pass

    # Education impossibilities
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
            if ("ph.d" in degree or "phd" in degree) and duration_years < 2:
                flags.append(f"PhD completed in {duration_years} year(s)")
                score += 1.5
            elif duration_years > 8:
                flags.append(f"Degree took {duration_years} years")
                score += 1.0

    # ══════════════════════════════════════════════════════════════════════
    # CHECK 10: Skill duration exceeds career scope
    # If someone claims 59 months of "advanced OpenSearch" but total
    # relevant career is only 33 months of frontend work.
    # ══════════════════════════════════════════════════════════════════════
    total_months = sum(j.get("duration_months", 0) for j in career)
    for skill in skills:
        prof = skill.get("proficiency", "")
        dur = skill.get("duration_months", 0)
        if prof in ("advanced", "expert") and dur > 0:
            cat = classify_skill(skill.get("name", ""))
            if cat in _AI_SKILL_CATS:
                if total_months > 0 and dur > total_months * 0.75:
                    flags.append(
                        f"Skill '{skill.get('name')}' duration ({dur}m) "
                        f"exceeds 75% of total career ({total_months}m)"
                    )
                    score += 1.0
                    break

    # ── Final assessment ──────────────────────────────────────────────────
    is_honeypot = score >= 2.0
    return is_honeypot, flags


# ── Helper functions ──────────────────────────────────────────────────────

def _detect_domains(text: str) -> set[str]:
    """Detect which career domains are present in a text."""
    found = set()
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            found.add(domain)
    return found


def _title_to_domain(title: str) -> str | None:
    """Map a job title to its expected career domain."""
    title = title.lower().strip()

    if any(t in title for t in ["civil engineer"]):
        return "mechanical_eng"  # Civil and mechanical are same domain
    if "mechanical engineer" in title:
        return "mechanical_eng"
    if "marketing" in title:
        return "marketing"
    if "accountant" in title or "finance" in title:
        return "finance"
    if "hr" in title or "human resource" in title:
        return "hr"
    if "operations" in title:
        return "operations"
    if "customer support" in title or "support" in title:
        return "support"
    if "sales" in title:
        return "sales"
    if any(t in title for t in [
        "software", "developer", "backend", "frontend", "full stack",
        "devops", "sre", "data engineer", "qa engineer",
    ]):
        return "tech_software"
    if any(t in title for t in [
        "ai", "ml", "machine learning", "data scientist",
        "research scientist", "nlp",
    ]):
        return "tech_ai_ml"
    if "graphic designer" in title or "designer" in title:
        return "brand_design"
    if "content writer" in title:
        return "marketing"
    if "project manager" in title:
        return "consulting"
    if "business analyst" in title:
        return "consulting"

    return None


def _parse_date(date_str: str) -> date:
    """Parse a date string in YYYY-MM-DD format."""
    if isinstance(date_str, date):
        return date_str
    return datetime.strptime(date_str, "%Y-%m-%d").date()
