# WorthyHire — System Architecture

## Overview

WorthyHire uses a **three-stage pipeline** to rank 100K candidates in under 3 minutes:

```
100K candidates ──▶ Stage 1 (Rule + Honeypot) ──▶ Top 500 ──▶ Stage 2 (Embeddings) ──▶ Top 200 ──▶ Stage 3 (Reranker) ──▶ Top 100
```

## Component Responsibilities

### `scripts/run_ranking.py` — CLI Orchestrator
- Loads candidates from JSON or JSONL (handles concatenated format)
- Invokes `HybridRanker` with configuration flags
- Writes CSV output and fairness report

### `backend/ranking/hybrid_ranker.py` — Pipeline Core
- **Stage 1**: Iterates all candidates, runs `score_candidate()` and `detect_honeypot()`
- **Stage 2**: Embeds only top 500 survivors using `SemanticEmbedder`
- **Stage 3**: Reranks top 200 with `CrossEncoderReranker`
- Generates explanations and fairness audit

### `scorer.py` — 6-Dimension Scoring Engine
- Role Fit (20%): Title match, career trajectory, product-company bonus
- Skills Match (35%): Taxonomy-based matching with trust multipliers
- Career Quality (10%): Tenure, progression, product vs consulting
- Education (5%): Degree level and STEM relevance
- Behavioral (10%): Response rate, speed, availability
- Red Flags (–penalty): Non-tech title, consulting-only, keyword stuffing

### `honeypot_detector.py` — 10-Check Trap Detection
Accumulates a suspicion score from 10 independent checks. Threshold ≥ 2.0 → flagged.
Flagged candidates receive a 0.05× score penalty.

### `skill_taxonomy.py` — Skill Classification
Maps 150+ skills into 11 relevance categories:
- `core_ai_ml` (1.0) — PyTorch, TensorFlow, Transformers
- `vector_db_retrieval` (0.95) — FAISS, Pinecone, Milvus
- `ml_foundations` (0.85) — NLP, Computer Vision, Recommendation Systems
- ... down to `non_technical` (0.0) — Photoshop, SEO, Marketing

### `backend/embeddings/embedder.py` — Semantic Embedder
Uses `BAAI/bge-small-en-v1.5` (384-dim) for bi-encoder similarity.

### `backend/ranking/reranker.py` — Cross-Encoder
Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` with 25/75 blend ratio.

### `backend/fairness/audit.py` — Fairness Auditor
Checks representation across college tier, location, and experience band.

## Data Flow

```
candidates.jsonl (100K)
  │
  ├──▶ JD Parser → JDRequirements (title, skills, must-haves)
  │
  ├──▶ Stage 1: For each candidate:
  │      ├── detect_honeypot() → is_honeypot, reasons
  │      ├── score_candidate() → dimension_scores
  │      └── rule_score = weighted sum of dimensions
  │
  ├──▶ Sort by rule_score, take top 500
  │
  ├──▶ Stage 2: Embed top 500
  │      ├── embed_jd() → jd_embedding
  │      ├── embed_candidates_batch() → candidate_embeddings
  │      ├── cosine_similarity() → semantic_scores
  │      ├── min-max normalize → [0, 1]
  │      └── _fuse_scores(semantic, rule) → hybrid_score
  │
  ├──▶ Stage 3: Rerank top 200
  │      ├── cross-encoder joint encoding
  │      └── 25% CE + 75% hybrid blend
  │
  └──▶ Output: top 100 with explanations + fairness report
```
