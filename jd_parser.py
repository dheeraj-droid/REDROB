"""
Job Description parser for the Redrob AI Ranking Challenge.

Encodes the structured requirements extracted from the Senior AI Engineer JD
into a data class that the scorer can consume. This module hard-codes the
specific JD since ranking must run offline (no LLM calls allowed).
"""

from dataclasses import dataclass, field


@dataclass
class JDRequirements:
    """Structured representation of the Senior AI Engineer JD at Redrob."""

    # ── Role metadata ─────────────────────────────────────────────────────
    title: str = "Senior AI Engineer — Founding Team"
    company: str = "Redrob AI"
    stage: str = "Series A"

    # ── Experience ────────────────────────────────────────────────────────
    min_years: float = 5.0
    max_years: float = 9.0
    flexible_band: float = 2.0  # Consider candidates ±2 years

    ideal_min_years: float = 6.0
    ideal_max_years: float = 8.0

    # ── Location ──────────────────────────────────────────────────────────
    preferred_locations: list[str] = field(default_factory=lambda: [
        "pune", "noida", "hyderabad", "mumbai", "delhi", "ncr",
        "delhi ncr", "gurgaon", "gurugram", "bengaluru", "bangalore",
        "chennai",
    ])
    preferred_countries: list[str] = field(default_factory=lambda: [
        "india",
    ])
    preferred_work_modes: list[str] = field(default_factory=lambda: [
        "hybrid", "onsite", "flexible",
    ])
    remote_penalty: float = 0.3  # Remote-only gets a mild penalty

    # ── Must-have skill categories ────────────────────────────────────────
    # These map to skill_taxonomy.MUST_HAVE_CLUSTERS
    must_have_categories: list[str] = field(default_factory=lambda: [
        "embeddings_retrieval",
        "vector_db",
        "python",
    ])

    # ── Nice-to-have skill areas ──────────────────────────────────────────
    nice_to_have_skills: list[str] = field(default_factory=lambda: [
        "LoRA", "QLoRA", "PEFT",
        "XGBoost", "LightGBM", "Learning to Rank",
        "HR-tech", "Recruiting", "Marketplace",
        "Docker", "Kubernetes", "Distributed Systems",
    ])

    # ── Disqualifiers (hard negatives from JD) ────────────────────────────
    # "Things we explicitly do NOT want" section

    # Penalty factor (0.0 to 1.0) applied when disqualifier is detected
    disqualifier_weights: dict = field(default_factory=lambda: {
        "pure_consulting_career": 0.2,      # Only TCS/Infosys/Wipro career
        "title_chaser": 0.4,                # New company every 1-1.5y for title
        "framework_enthusiast_only": 0.5,   # LangChain-only recent experience
        "pure_research_no_production": 0.3, # Academic labs only
        "no_recent_code": 0.3,              # Moved to architecture 18m+ ago
        "domain_mismatch": 0.4,             # CV/speech/robotics only
        "keyword_stuffer": 0.15,            # Non-tech title + AI keyword dump
    })

    # ── Notice period preferences ─────────────────────────────────────────
    ideal_notice_days: int = 30
    max_preferred_notice: int = 60
    notice_penalty_threshold: int = 90  # Beyond this, heavy penalty

    # ── Salary range (from JD context — Series A, senior role) ────────────
    # Approximate expected range in LPA
    salary_min_lpa: float = 25.0
    salary_max_lpa: float = 60.0

    # ── Behavioral preferences ────────────────────────────────────────────
    min_response_rate: float = 0.3      # Below this = likely unavailable
    min_profile_completeness: float = 50.0
    recency_weight_days: int = 90       # Active within 90 days is good


def get_jd_requirements() -> JDRequirements:
    """Return the parsed JD requirements for the Redrob Senior AI Engineer role."""
    return JDRequirements()
