# WorthyHire — Fairness & Bias Prevention

## Philosophy

WorthyHire is designed to evaluate candidates on **what they can do**, not **who they are**. The ranking system intentionally avoids:

- ❌ Gender-based signals
- ❌ Age or generation bias
- ❌ Religion, caste, or ethnicity
- ❌ College prestige as a dominant factor
- ❌ Geographic discrimination (beyond job-relevant location fit)

## Implementation

### Low Education Weight (5%)

Education is weighted at only 5% of the total score. A Tier-3 college graduate with strong skills and a good career trajectory can easily outrank a Tier-1 graduate with weaker technical skills.

### No Name-Based Scoring

Candidate names are anonymized in the dataset. The system never processes names for scoring or ranking decisions.

### Notice Period as Availability, Not Penalty

Long notice periods (90+ days) are treated as an availability signal, not a disqualifying factor. They reduce the behavioral score slightly but don't eliminate candidates.

### Career Gap Tolerance

Career gaps are not penalized. Non-linear career paths (freelancing, open-source, career changes) are valued as signs of diverse experience.

## Fairness Audit

Every ranking run generates a `fairness_report.json` that checks for bias across three dimensions:

### 1. College Tier
- Groups: Tier 1 / Tier 2 / Tier 3 / Unknown
- Checks: Representation ratio (top_pct / population_pct)
- Flag: Ratio < 0.5 or > 2.0 triggers a warning

### 2. Location
- Groups: Metro / Tier 2 City / Other / International
- Checks: Same representation ratio analysis
- Flag: Geographic concentration > 80% triggers a warning

### 3. Experience Band
- Groups: Junior (0-3y) / Mid (3-7y) / Senior (7-12y) / Staff (12+y)
- Checks: Band-level representation vs. population
- Flag: Any band with 0% representation triggers a warning

### Metrics Computed

For each group in each dimension:
- `population_count` / `population_pct`: How many candidates are in this group
- `top_count` / `top_pct`: How many of the top-ranked candidates are from this group
- `representation_ratio`: top_pct / population_pct (1.0 = perfect representation)
- `avg_score_population`: Average score across all candidates in this group
- `avg_score_top`: Average score among top-ranked candidates from this group

## Limitations

- The fairness audit is **descriptive**, not prescriptive. It flags disparities but does not enforce quotas.
- Gender is not available in the dataset, so gender-based analysis is not possible.
- The audit runs on the ranked output, not during ranking — it does not modify scores.
