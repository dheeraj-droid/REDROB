#!/usr/bin/env python3
"""
WorthyHire — Legacy CLI Wrapper

This file exists for backward compatibility with the submission spec's
reproduce_command. It delegates to the main pipeline at scripts/run_ranking.py.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./dheeraj-droid.csv
"""

import sys
import os

# Add project root to path
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Import and run the main pipeline
from scripts.run_ranking import main

if __name__ == "__main__":
    main()
