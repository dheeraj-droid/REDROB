#!/usr/bin/env python3
"""
Redrob AI — Intelligent Candidate Discovery & Ranking System

Main pipeline entry point. Processes 100K candidates from JSONL,
scores them against the Senior AI Engineer JD, and produces a
top-100 ranked CSV submission.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
    python rank.py  # Uses default paths

Architecture:
    1. Parse JD → structured requirements
    2. Stream candidates → honeypot filter → feature extraction
    3. Multi-dimensional scoring (role fit, skills, career, education,
       behavioral signals, red flags)
    4. Rank by composite score → top 100
    5. Generate reasoning for each ranked candidate
    6. Write CSV output

Constraints:
    - ≤5 min on CPU, ≤16 GB RAM, no GPU, no network calls
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from jd_parser import get_jd_requirements
from honeypot_detector import detect_honeypot
from scorer import score_candidate
from reasoning_generator import generate_reasoning


def main():
    parser = argparse.ArgumentParser(
        description="Redrob AI Candidate Ranking System"
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default="candidates.jsonl",
        help="Path to candidates JSONL file",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="submission.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Number of top candidates to output (default: 100)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress and debug info",
    )
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    out_path = Path(args.out)
    top_n = args.top_n
    verbose = args.verbose

    # Resolve candidate file — check common locations
    if not candidates_path.exists():
        alt_paths = [
            Path("[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"),
            Path("data/candidates.jsonl"),
        ]
        for alt in alt_paths:
            if alt.exists():
                candidates_path = alt
                break

    if not candidates_path.exists():
        print(f"Error: Candidates file not found: {candidates_path}")
        print("Provide path via --candidates flag.")
        sys.exit(1)

    print("=" * 60)
    print("  Redrob AI — Candidate Ranking System")
    print("=" * 60)
    print(f"  Candidates: {candidates_path}")
    print(f"  Output:     {out_path}")
    print(f"  Top N:      {top_n}")
    print("=" * 60)

    start_time = time.time()

    # ── Step 1: Parse JD ──────────────────────────────────────────────────
    jd = get_jd_requirements()
    print(f"\n[1/5] JD parsed: {jd.title}")

    # ── Step 2: Stream & score all candidates ─────────────────────────────
    print("[2/5] Scoring candidates...")

    all_scored = []
    honeypot_count = 0
    total_count = 0
    skipped_honeypots = []

    with open(candidates_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                if verbose:
                    print(f"  WARN: Skipping malformed JSON at line {line_num}")
                continue

            total_count += 1
            cid = candidate.get("candidate_id", f"UNKNOWN_{line_num}")

            # ── Honeypot check ────────────────────────────────────────────
            is_honeypot, honeypot_reasons = detect_honeypot(candidate)
            if is_honeypot:
                honeypot_count += 1
                skipped_honeypots.append((cid, honeypot_reasons))
                # Don't skip entirely — score them but apply massive penalty
                # so they naturally fall out of top 100

            # ── Score ─────────────────────────────────────────────────────
            result = score_candidate(candidate, jd)
            final_score = result["final_score"]

            # Apply honeypot penalty
            if is_honeypot:
                final_score *= 0.05  # Crush the score

            all_scored.append({
                "candidate_id": cid,
                "candidate": candidate,
                "score_result": result,
                "final_score": final_score,
                "is_honeypot": is_honeypot,
            })

            # Progress
            if verbose and total_count % 10000 == 0:
                elapsed = time.time() - start_time
                print(
                    f"  Processed {total_count:,} candidates "
                    f"({elapsed:.1f}s elapsed, "
                    f"{honeypot_count} honeypots detected)"
                )

    scoring_time = time.time() - start_time
    print(
        f"  Scored {total_count:,} candidates in {scoring_time:.1f}s "
        f"({honeypot_count} honeypots detected)"
    )

    # ── Step 3: Rank ──────────────────────────────────────────────────────
    print("[3/5] Ranking...")

    # Sort by score descending, tie-break by candidate_id ascending
    all_scored.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))

    # Take top N
    top_candidates = all_scored[:top_n]

    # Check honeypot rate in top N
    top_honeypots = sum(1 for c in top_candidates if c["is_honeypot"])
    honeypot_rate = top_honeypots / top_n if top_n > 0 else 0
    print(
        f"  Top {top_n}: {top_honeypots} honeypots "
        f"({honeypot_rate:.1%} rate, threshold: <10%)"
    )

    if honeypot_rate >= 0.10:
        print("  [WARNING] Honeypot rate exceeds 10% -- submission may be DQ'd!")

    # ── Step 4: Generate reasoning ────────────────────────────────────────
    print("[4/5] Generating reasoning...")

    ranked_output = []
    
    if top_candidates:
        max_raw = top_candidates[0]["final_score"]
        min_raw = top_candidates[-1]["final_score"]
    else:
        max_raw = 1.0
        min_raw = 0.0

    for rank, entry in enumerate(top_candidates, 1):
        reasoning = generate_reasoning(
            entry["candidate"],
            entry["score_result"],
            rank,
        )
        # Normalize score to submission range
        norm_score = _normalize_score(
            entry["final_score"], rank, top_n, min_raw, max_raw
        )

        ranked_output.append({
            "candidate_id": entry["candidate_id"],
            "rank": rank,
            "score": round(norm_score, 4),
            "reasoning": reasoning,
        })

    # ── Step 5: Write CSV ─────────────────────────────────────────────────
    print(f"[5/5] Writing {out_path}...")

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["candidate_id", "rank", "score", "reasoning"],
        )
        writer.writeheader()
        writer.writerows(ranked_output)

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  [OK] Done! {top_n} candidates ranked in {total_time:.1f}s")
    print(f"  Output: {out_path}")
    print(f"{'=' * 60}")

    # Print top 10 summary
    print(f"\n  Top 10 candidates:")
    print(f"  {'Rank':<5} {'ID':<16} {'Score':<8} {'Title'}")
    print(f"  {'-'*5} {'-'*16} {'-'*8} {'-'*30}")
    for entry in ranked_output[:10]:
        cid = entry["candidate_id"]
        # Find the candidate data
        cand_data = next(
            (c for c in top_candidates if c["candidate_id"] == cid),
            None
        )
        title = ""
        if cand_data:
            title = cand_data["candidate"].get("profile", {}).get(
                "current_title", ""
            )
        print(
            f"  {entry['rank']:<5} {entry['candidate_id']:<16} "
            f"{entry['score']:<8.4f} {title}"
        )

    # Print honeypot summary
    if skipped_honeypots and verbose:
        print(f"\n  Honeypots detected ({honeypot_count} total):")
        for cid, reasons in skipped_honeypots[:5]:
            print(f"    {cid}: {'; '.join(reasons[:2])}")
        if len(skipped_honeypots) > 5:
            print(f"    ... and {len(skipped_honeypots) - 5} more")


def _normalize_score(raw_score: float, rank: int, total: int, min_raw: float, max_raw: float) -> float:
    """
    Normalize scores proportionally to preserve actual score gaps.
    Maps to [0.2000, 0.9990] range.
    """
    target_max = 0.9990
    target_min = 0.2000
    
    if max_raw == min_raw or max_raw <= 0:
        base_norm = target_max - (rank - 1) * (target_max - target_min) / max(total - 1, 1)
    else:
        # Proportional mapping based on max and min raw scores in the top N
        ratio = (raw_score - min_raw) / (max_raw - min_raw)
        base_norm = target_min + ratio * (target_max - target_min)
        
    # Subtract a tiny fraction to guarantee strict descent for exact ties
    tie_breaker = (rank - 1) * 0.00001
    return base_norm - tie_breaker


if __name__ == "__main__":
    main()
