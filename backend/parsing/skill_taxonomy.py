"""
WorthyHire — Skill Taxonomy (backend.parsing.skill_taxonomy)

Re-exports from the original skill_taxonomy module.
"""

import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from skill_taxonomy import (
    SKILL_TO_CATEGORY,
    CATEGORY_RELEVANCE,
    MUST_HAVE_CLUSTERS,
    PROFICIENCY_WEIGHT,
    CONSULTING_SERVICES_COMPANIES,
    NON_TECHNICAL_TITLES,
    AI_ML_TITLES,
    TITLE_RELEVANCE,
    classify_skill,
    get_category_relevance,
    get_title_relevance,
    is_non_technical_title,
    is_consulting_company,
)

__all__ = [
    "SKILL_TO_CATEGORY", "CATEGORY_RELEVANCE", "MUST_HAVE_CLUSTERS",
    "PROFICIENCY_WEIGHT", "CONSULTING_SERVICES_COMPANIES",
    "NON_TECHNICAL_TITLES", "AI_ML_TITLES", "TITLE_RELEVANCE",
    "classify_skill", "get_category_relevance", "get_title_relevance",
    "is_non_technical_title", "is_consulting_company",
]
