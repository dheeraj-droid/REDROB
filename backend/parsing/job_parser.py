"""
WorthyHire — Job Description Parser (backend.parsing.job_parser)

Re-exports from the original jd_parser module for backward compatibility,
while providing the canonical import path for the new package structure.
"""

import sys
import os

# Add project root to path so we can import the original flat modules
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from jd_parser import JDRequirements, get_jd_requirements

__all__ = ["JDRequirements", "get_jd_requirements"]
