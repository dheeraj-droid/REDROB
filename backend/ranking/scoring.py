"""
WorthyHire — Rule-Based Scoring (backend.ranking.scoring)

Re-exports from the original scorer module.
"""

import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scorer import score_candidate, DIMENSION_WEIGHTS

__all__ = ["score_candidate", "DIMENSION_WEIGHTS"]
