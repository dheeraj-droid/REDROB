# WorthyHire — Methodology

## Problem Statement

Given a job description for a "Senior AI Engineer — Founding Team" and ~100K candidate profiles, rank the top 100 most relevant candidates while:
- Detecting and filtering ~80 honeypot (trap) candidates
- Providing explainable reasoning for each ranking decision
- Ensuring fairness across demographics

## Approach: Three-Stage Hybrid Pipeline

### Why Hybrid?

Pure rule-based systems miss semantic similarity ("NLP Engineer" should match "AI Engineer").
Pure embedding systems can be fooled by well-written irrelevant profiles (a marketer mentioning AI buzzwords).
Our hybrid approach uses rules for precision and embeddings for recall.

### Stage 1: Rule-Based Pre-Filter (~60s for 100K)

Every candidate is scored across 6 dimensions:

**Skills Match (35%)** — The dominant signal. We built a hand-crafted taxonomy of 150+ skills mapped to 11 relevance categories. Each skill gets a trust score based on:
- Proficiency level (expert: 1.0, advanced: 0.8, intermediate: 0.5, beginner: 0.2)
- Duration months (logarithmic scaling)
- Endorsement count

Must-have skill clusters (vector_db, ml_frameworks, llm_tools) are explicitly checked.

**Role Fit (20%)** — Title relevance is scored using a curated mapping of AI/ML titles. Career trajectory analysis checks if the candidate is moving toward AI roles.

**Behavioral (10%)** — Platform engagement signals: response rate, average response time, profile completeness. High responsiveness indicates active job seekers.

**Career Quality (10%)** — Product-company experience is valued over consulting. Company diversity and tenure patterns are analyzed.

**Education (5%)** — STEM degrees from Tier 1/2 institutions get a small bonus. This is intentionally low-weighted to avoid college-brand bias.

**Red Flags (penalty)** — Non-technical titles, consulting-only careers, keyword stuffing, long notice periods reduce scores.

### Stage 2: Semantic Embedding (Top 500 only, ~60s)

Only the top 500 rule-scored candidates are embedded. This is the key insight that makes the pipeline fast:
- 100K candidates → rule score → sort → take top 500
- Embed 500 candidates using `BAAI/bge-small-en-v1.5` (~5s)
- Compute cosine similarity with JD embedding
- Fuse with rule scores using weighted combination

**Skill Gate**: If a candidate's must-have skill score is below 0.15, their semantic score is crushed by 85%. This prevents "well-written irrelevant" profiles from scoring high semantically.

### Stage 3: Cross-Encoder Reranking (Top 200, ~20s)

`cross-encoder/ms-marco-MiniLM-L-6-v2` jointly encodes (JD, candidate) pairs for more accurate similarity. Blended 25% CE + 75% hybrid to keep rule-based scoring dominant.

## Honeypot Detection

10 independent checks accumulate a suspicion score:

1. **Summary-title contradiction**: The dataset uses template summaries that say "my background is in marketing manager" even when the title is "Civil Engineer"
2. **Title-description domain clash**: Cross-referencing current title against job description domain
3. **Template self-learner**: Detecting exact template phrases like "curious about how AI tools could augment my work"
4. **Career description incoherence**: Jobs spanning 4+ unrelated domains (brand_design, support, finance, operations)
5. **Skill-career mismatch**: Advanced AI skills with zero technical career history
6. **Non-tech title + AI skills**: "Mechanical Engineer" claiming advanced FAISS
7. **Experience mismatch**: Claimed YoE far exceeds career history
8. **Expert with zero duration**: "Expert" proficiency with 0 months practiced
9. **Timeline impossibilities**: End dates before start dates, impossible degree durations
10. **Skill duration exceeds career**: 59 months of OpenSearch experience in a 60-month career

Candidates with suspicion score ≥ 2.0 receive a 0.05× penalty (effectively removed from ranking).

## Key Design Decisions

1. **Skills dominate (35%)** because the JD is for a technical AI role — you either have the skills or you don't.
2. **Semantic score is gated** by skill score to prevent "well-written noise" from ranking highly.
3. **Two-stage embedding** reduces compute from hours to minutes.
4. **Cross-encoder is conservative** (25% blend) because ms-marco is a search model, not a job matcher.
5. **Honeypot detection is multi-signal** — no single check has > 2.5 score impact, reducing false positives.
