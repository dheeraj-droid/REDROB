"""
WorthyHire — Honeypot Detector (backend.ranking.honeypot_detector)

Re-exports from the original honeypot_detector module.
"""

import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from honeypot_detector import detect_honeypot

__all__ = ["detect_honeypot"]
