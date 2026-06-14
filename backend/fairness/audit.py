"""
WorthyHire — Fairness Auditor

Performs statistical fairness auditing on ranking results to detect
systematic bias against candidate groups.

Groups analyzed:
  - College tier (tier_1, tier_2, tier_3, tier_4, unknown)
  - Location tier (preferred city, same country, international)
  - Experience band (junior, mid, senior, staff+)

Outputs a fairness_report.json with flags for any group that is
significantly underrepresented in the top rankings.
"""

import json
from datetime import datetime
from typing import Any


class FairnessAuditor:
    """Statistical fairness auditor for candidate rankings."""

    def audit(
        self,
        all_candidates: list[dict],
        top_candidates: list[dict],
    ) -> dict:
        """
        Run fairness audit comparing top candidates vs full population.

        Returns a report dict with group statistics and flags.
        """
        report = {
            "audit_timestamp": datetime.now().isoformat(),
            "total_candidates": len(all_candidates),
            "top_candidates": len(top_candidates),
            "groups_analyzed": [],
            "group_stats": {},
            "flags": [],
            "summary": "",
        }

        # ── Analyze by college tier ───────────────────────────────────────
        tier_stats = self._analyze_group(
            all_candidates, top_candidates,
            group_fn=self._get_college_tier,
            group_name="college_tier",
        )
        report["groups_analyzed"].append("college_tier")
        report["group_stats"]["college_tier"] = tier_stats

        # ── Analyze by location tier ──────────────────────────────────────
        loc_stats = self._analyze_group(
            all_candidates, top_candidates,
            group_fn=self._get_location_tier,
            group_name="location_tier",
        )
        report["groups_analyzed"].append("location_tier")
        report["group_stats"]["location_tier"] = loc_stats

        # ── Analyze by experience band ────────────────────────────────────
        exp_stats = self._analyze_group(
            all_candidates, top_candidates,
            group_fn=self._get_experience_band,
            group_name="experience_band",
        )
        report["groups_analyzed"].append("experience_band")
        report["group_stats"]["experience_band"] = exp_stats

        # ── Generate flags ────────────────────────────────────────────────
        all_stats = {**tier_stats, **loc_stats, **exp_stats}
        flags = self._generate_flags(all_stats)
        report["flags"] = flags

        if not flags:
            report["summary"] = "No significant fairness concerns detected."
        elif any(f["severity"] == "critical" for f in flags):
            report["summary"] = (
                f"CRITICAL: {len(flags)} fairness concern(s) detected. "
                "Review the flags for details."
            )
        else:
            report["summary"] = (
                f"{len(flags)} minor fairness concern(s) detected. "
                "Review the flags for details."
            )

        return report

    def _analyze_group(
        self,
        all_candidates: list[dict],
        top_candidates: list[dict],
        group_fn,
        group_name: str,
    ) -> dict:
        """Compute statistics for a grouping function."""
        # Count groups in full population
        pop_counts: dict[str, int] = {}
        pop_scores: dict[str, list] = {}
        for c in all_candidates:
            group = group_fn(c)
            pop_counts[group] = pop_counts.get(group, 0) + 1
            scores = pop_scores.setdefault(group, [])
            scores.append(c.get("final_score", 0))

        # Count groups in top N
        top_counts: dict[str, int] = {}
        top_scores: dict[str, list] = {}
        for c in top_candidates:
            group = group_fn(c)
            top_counts[group] = top_counts.get(group, 0) + 1
            scores = top_scores.setdefault(group, [])
            scores.append(c.get("final_score", 0))

        total_pop = max(len(all_candidates), 1)
        total_top = max(len(top_candidates), 1)

        stats = {}
        for group in sorted(pop_counts.keys()):
            pop_n = pop_counts.get(group, 0)
            top_n = top_counts.get(group, 0)
            pop_pct = pop_n / total_pop
            top_pct = top_n / total_top

            p_scores = pop_scores.get(group, [0])
            t_scores = top_scores.get(group, [])

            stats[f"{group_name}_{group}"] = {
                "population_count": pop_n,
                "population_pct": round(pop_pct, 4),
                "top_count": top_n,
                "top_pct": round(top_pct, 4),
                "representation_ratio": round(top_pct / pop_pct, 2) if pop_pct > 0 else 0,
                "avg_score_population": round(sum(p_scores) / len(p_scores), 4),
                "avg_score_top": round(sum(t_scores) / len(t_scores), 4) if t_scores else 0,
            }

        return stats

    def _generate_flags(self, all_stats: dict) -> list[dict]:
        """Generate fairness flags from group statistics."""
        flags = []

        for group_key, stats in all_stats.items():
            pop_pct = stats["population_pct"]
            top_pct = stats["top_pct"]
            rep_ratio = stats["representation_ratio"]

            # Flag if a group with >5% population has <50% representation ratio
            if pop_pct >= 0.05 and rep_ratio < 0.5:
                severity = "critical" if rep_ratio < 0.25 else "warning"
                flags.append({
                    "group": group_key,
                    "metric": "representation_ratio",
                    "value": rep_ratio,
                    "population_pct": pop_pct,
                    "top_pct": top_pct,
                    "severity": severity,
                    "message": (
                        f"Group '{group_key}' makes up {pop_pct:.0%} of candidates "
                        f"but only {top_pct:.0%} of top rankings "
                        f"(ratio: {rep_ratio:.2f}x)"
                    ),
                })

        return flags

    # ── Group extraction functions ────────────────────────────────────────

    @staticmethod
    def _get_college_tier(entry: dict) -> str:
        """Extract college tier from candidate."""
        candidate = entry.get("candidate", entry)
        education = candidate.get("education", [])
        if not education:
            return "no_education"

        # Use the best (lowest tier number)
        best_tier = "unknown"
        tier_rank = {"tier_1": 1, "tier_2": 2, "tier_3": 3, "tier_4": 4, "unknown": 5}
        for edu in education:
            tier = edu.get("tier", "unknown")
            if tier_rank.get(tier, 5) < tier_rank.get(best_tier, 5):
                best_tier = tier

        return best_tier

    @staticmethod
    def _get_location_tier(entry: dict) -> str:
        """Extract location tier from candidate."""
        candidate = entry.get("candidate", entry)
        profile = candidate.get("profile", {})
        country = profile.get("country", "").lower()

        if country == "india":
            location = profile.get("location", "").lower()
            preferred = [
                "pune", "noida", "hyderabad", "mumbai", "delhi",
                "gurgaon", "gurugram", "bengaluru", "bangalore", "chennai",
            ]
            if any(p in location for p in preferred):
                return "preferred_city"
            return "india_other"
        elif country:
            return "international"
        return "unknown_location"

    @staticmethod
    def _get_experience_band(entry: dict) -> str:
        """Extract experience band from candidate."""
        candidate = entry.get("candidate", entry)
        yoe = candidate.get("profile", {}).get("years_of_experience", 0)
        if yoe < 2:
            return "junior_0_2y"
        elif yoe < 5:
            return "mid_2_5y"
        elif yoe < 10:
            return "senior_5_10y"
        else:
            return "staff_10y_plus"


def save_fairness_report(report: dict, output_path: str):
    """Save fairness report to JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
