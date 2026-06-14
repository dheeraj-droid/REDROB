#!/usr/bin/env python3
"""
WorthyHire — CLI Ranking Pipeline

Usage:
  python scripts/run_ranking.py --candidates data/sample_input/sample_candidates.json --out submission.csv --verbose

This is the main entry point for running the WorthyHire ranking pipeline
from the command line.
"""

import argparse
import csv
import json
import sys
import os
import time
from pathlib import Path

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from backend.ranking.hybrid_ranker import HybridRanker
from backend.fairness.audit import save_fairness_report


def load_candidates(path: str) -> list[dict]:
    """Load candidates from a JSON array or JSONL file.
    
    Handles:
      - Standard JSON arrays
      - Standard JSONL (one object per line)
      - Concatenated JSONL (multiple objects per line without separators)
    """
    path = Path(path)
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return [data]
    else:
        # JSONL — handle concatenated objects
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

        # Split concatenated objects: }{"candidate_id" → }\n{"candidate_id"
        raw = raw.replace('}{"candidate_id"', '}\n{"candidate_id"')
        
        candidates = []
        errors = 0
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError:
                errors += 1
        
        del raw  # free ~500MB of memory
        
        if errors:
            print(f"  WARN: Skipped {errors} malformed JSON lines")
        return candidates


def write_csv(ranked: list[dict], output_path: str):
    """Write ranked results to CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for entry in ranked:
            explanation = entry.get("explanation", {})
            reasoning = explanation.get("explanation", "") if isinstance(explanation, dict) else str(explanation)
            writer.writerow([
                entry["candidate_id"],
                entry["rank"],
                f"{entry['normalized_score']:.4f}",
                reasoning,
            ])


def main():
    parser = argparse.ArgumentParser(
        description="WorthyHire — Intelligent Candidate Ranking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_ranking.py --candidates data/sample_input/sample_candidates.json --out dheeraj-droid.csv
  python scripts/run_ranking.py --candidates candidates.jsonl --out dheeraj-droid.csv --no-embeddings --verbose
        """,
    )
    parser.add_argument(
        "--candidates", required=True,
        help="Path to candidate JSON/JSONL file",
    )
    parser.add_argument(
        "--out", default="dheeraj-droid.csv",
        help="Output CSV file path (default: dheeraj-droid.csv)",
    )
    parser.add_argument(
        "--top-n", type=int, default=100,
        help="Number of top candidates (default: 100)",
    )
    parser.add_argument(
        "--no-embeddings", action="store_true",
        help="Disable semantic embedding scoring",
    )
    parser.add_argument(
        "--no-reranker", action="store_true",
        help="Disable cross-encoder reranking",
    )
    parser.add_argument(
        "--fairness-report", default=None,
        help="Path to write fairness report JSON (default: none)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed progress",
    )

    args = parser.parse_args()

    # Banner
    print("=" * 60)
    print("  WorthyHire — Intelligent Candidate Ranking")
    print("=" * 60)
    print(f"  Candidates:  {args.candidates}")
    print(f"  Output:      {args.out}")
    print(f"  Top N:       {args.top_n}")
    print(f"  Embeddings:  {'ON' if not args.no_embeddings else 'OFF'}")
    print(f"  Reranker:    {'ON' if not args.no_reranker else 'OFF'}")
    print("=" * 60)
    print()

    # Load candidates
    print("[1/4] Loading candidates...")
    candidates = load_candidates(args.candidates)
    print(f"  Loaded {len(candidates):,} candidates")

    # Run ranking
    print("[2/4] Running hybrid ranking pipeline...")
    ranker = HybridRanker(
        use_embeddings=not args.no_embeddings,
        use_reranker=not args.no_reranker,
        verbose=args.verbose,
    )

    result = ranker.rank(candidates=candidates, top_n=args.top_n)
    ranked = result["ranked"]
    metadata = result["metadata"]

    # Write output
    print(f"[3/4] Writing {args.out}...")
    write_csv(ranked, args.out)

    # Write fairness report
    if args.fairness_report:
        print(f"[4/4] Writing fairness report to {args.fairness_report}...")
        save_fairness_report(result["fairness_report"], args.fairness_report)
    else:
        print("[4/4] Skipping fairness report (use --fairness-report to enable)")

    # Summary
    print()
    print("=" * 60)
    print(f"  [OK] Done! {len(ranked)} candidates ranked in {metadata['total_time_seconds']:.1f}s")
    print(f"  Output: {args.out}")
    print(f"  Honeypots: {metadata['honeypots_detected']} detected, "
          f"{metadata['top_n_honeypots']} in top {len(ranked)}")
    print("=" * 60)
    print()

    # Top 10
    print("  Top 10 candidates:")
    print(f"  {'Rank':<6}{'ID':<17}{'Score':<9}{'Semantic':<10}{'Title'}")
    print(f"  {'-'*5:<6}{'-'*16:<17}{'-'*8:<9}{'-'*8:<10}{'-'*30}")
    for entry in ranked[:10]:
        profile = entry["candidate"].get("profile", {})
        title = profile.get("current_title", "N/A")[:30]
        print(
            f"  {entry['rank']:<6}"
            f"{entry['candidate_id']:<17}"
            f"{entry['normalized_score']:<9.4f}"
            f"{entry.get('semantic_score', 0):<10.4f}"
            f"{title}"
        )


if __name__ == "__main__":
    main()
