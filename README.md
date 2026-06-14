# WorthyHire 🎯

**Intelligent Candidate Ranking for the India Runs Data & AI Challenge**

> A three-stage hybrid pipeline that combines rule-based scoring, semantic embeddings, and cross-encoder reranking to rank 100K candidates in under 3 minutes — with zero honeypots in the top 100.

---

## 🏆 Results

| Metric | Value |
|--------|-------|
| **Candidates processed** | 100,000 |
| **Honeypots in top 100** | **0** |
| **Top 10 titles** | All Senior AI / ML / NLP Engineers |
| **Total runtime** | 143 seconds (CPU only) |
| **Honeypots detected** | 68,360 (68.4%) |

---

## 🏗️ Architecture

```
                    ┌─────────────────────────────────────┐
                    │     WorthyHire Pipeline              │
                    └─────────────────────────────────────┘

  Stage 1: Rule-Based Scoring (100K candidates, ~60s)
  ┌──────────────────────────────────────────────────────┐
  │  JD Parser ──▶ 6-Dimension Scorer ──▶ Honeypot      │
  │  (title,       (role_fit, skills,     Detector       │
  │   skills,       career, education,    (10 checks,    │
  │   reqs)         behavioral, flags)    68% filtered)  │
  └──────────────────────────────────┬───────────────────┘
                                     ▼
  Stage 2: Semantic Embedding (top 500 candidates, ~60s)
  ┌──────────────────────────────────────────────────────┐
  │  bge-small-en-v1.5 ──▶ Cosine Similarity ──▶ Hybrid │
  │  (bi-encoder,           (JD ↔ candidate)     Score   │
  │   384-dim)               min-max normalized   Fusion │
  └──────────────────────────────────┬───────────────────┘
                                     ▼
  Stage 3: Cross-Encoder Rerank (top 200, ~20s)
  ┌──────────────────────────────────────────────────────┐
  │  ms-marco-MiniLM ──▶ 25/75 Blend ──▶ Final Ranking  │
  │  (cross-encoder,      (CE / hybrid)   + Explanations │
  │   joint encoding)                     + Fairness Audit│
  └──────────────────────────────────────────────────────┘
```

### Scoring Dimensions

| Dimension | Weight | Signal |
|-----------|--------|--------|
| **Semantic Similarity** | 20% | Embedding cosine similarity (JD ↔ candidate) |
| **Must-Have Skills** | 35% | Taxonomy-based skill cluster matching with trust multipliers |
| **Role Fit** | 20% | Title relevance + career trajectory + product-company bonus |
| **Career Quality** | 10% | Tenure, progression, product vs consulting |
| **Behavioral Signals** | 10% | Response rate, speed, availability, engagement |
| **Education** | 5% | Degree level + field relevance |

### Honeypot Detection (Hardcoded Python Heuristics)

These are explicit, rule-based `if/else` heuristics (string matching and arithmetic) running in `honeypot_detector.py` to catch traps without needing LLMs.

| Hardcoded Python Rule | What It Catches | Hit Rate |
|-------|----------------|----------|
| Summary-title contradiction | Summary says "marketing" but title is "Civil Engineer" | 38,577 |
| Title-description domain clash | Title domain ≠ description domain | ~35,000 |
| Template self-learner | Fake "curious about AI" template summaries | ~5,000 |
| Career description incoherence | Jobs spanning 4+ unrelated domains | ~13,000 |
| Skill-career mismatch | Advanced AI skills + zero tech career | ~5,000 |
| Non-tech title + AI skills | "Mechanical Engineer" with advanced FAISS | ~5,000 |
| Experience mismatch | Claims 10y but career totals 3y | varies |
| Expert with zero duration | "Expert" proficiency, 0 months practiced | varies |
| Timeline impossibility | End date before start date | varies |
| Skill duration exceeds career | 59 months of OpenSearch in 60-month career | varies |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- ~500MB disk space (for transformer models, auto-downloaded on first run)

### Installation

```bash
git clone https://github.com/dheeraj-droid/REDROB.git
cd REDROB
pip install -r requirements.txt

# Pre-download ML models to local cache (Required for offline execution)
python download_models.py
```

### Run Ranking

```bash
# Full pipeline  — embeddings + cross-encoder + fairness
python scripts/run_ranking.py \
  --candidates candidates.jsonl \
  --out WorthyHire.csv \
  --fairness-report fairness_report.json \
  --verbose
```

### Run API Server

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

### Run Dashboard

```bash
streamlit run demo/dashboard.py
```

### Docker

```bash
docker-compose up --build
# API: http://localhost:8000
# Dashboard: http://localhost:8501
```

---

## 📁 Project Structure

```
WorthyHire/
├── scripts/
│   └── run_ranking.py              # CLI entry point — handles JSON/JSONL input
├── backend/
│   ├── config.py                   # Weights, thresholds, model paths
│   ├── app.py                      # FastAPI server
│   ├── schemas.py                  # Pydantic request/response models
│   ├── ranking/
│   │   ├── hybrid_ranker.py        # Three-stage pipeline orchestrator
│   │   └── reranker.py             # Cross-encoder reranker (ms-marco)
│   ├── embeddings/
│   │   └── embedder.py             # Semantic embedder (bge-small)
│   ├── explainability/
│   │   └── explanations.py         # Structured reasoning generator
│   └── fairness/
│       └── audit.py                # Bias detection across 3 dimensions
├── scorer.py                       # 6-dimension rule-based scoring engine
├── skill_taxonomy.py               # 150+ skills → 11 categories mapping
├── honeypot_detector.py            # 10-check trap profile detection
├── jd_parser.py                    # Job description parser
├── reasoning_generator.py          # Natural language explanation builder
├── demo/
│   └── dashboard.py                # Streamlit interactive dashboard
├── docs/
│   ├── architecture.md             # System design documentation
│   ├── methodology.md              # Approach deep-dive
│   └── fairness.md                 # Fairness philosophy & metrics
├── Dockerfile                      # Container image
├── docker-compose.yml              # Multi-service deployment
├── requirements.txt                # Python dependencies
├── submission_metadata.yaml        # Challenge submission metadata
└── WorthyHire.csv                  # Final ranked output (100 candidates)
```

---

## 🔬 Methodology Deep-Dive

### Stage 1: Rule-Based Scoring + Honeypot Detection

Runs on **all 100K candidates** in ~60 seconds. Each candidate is scored across 6 dimensions:

1. **Skills Match (35%)** — Maps candidate skills to 11 taxonomy categories (core_ai_ml, ml_foundations, vector_db_retrieval, etc.). Uses a trust multiplier based on proficiency × duration × endorsements. Must-have skill clusters are explicitly checked.

2. **Role Fit (20%)** — Title relevance scoring with bonus for AI/ML-specific titles, product-company experience, and career trajectory analysis.

3. **Semantic (20%)** — Added in Stage 2 (see below).

4. **Career Quality (10%)** — Company diversity, tenure patterns, product vs. consulting background.

5. **Behavioral (10%)** — Platform engagement, response rate, response speed, availability.

6. **Education (5%)** — Degree level and field relevance (STEM preferred).

The **honeypot detector** runs 10 independent checks and assigns a suspicion score. Candidates scoring ≥ 2.0 are flagged and receive a 0.05× penalty.

### Stage 2: Semantic Embedding (Top 500 Only)

After Stage 1 ranks all candidates, only the **top 500** are embedded using `BAAI/bge-small-en-v1.5`. This reduces embedding time from hours to ~60 seconds.

- JD text and candidate text are encoded independently (bi-encoder)
- Cosine similarity is min-max normalized across the pool
- Hybrid fusion combines semantic (20%) with rule scores (80%)

### Stage 3: Cross-Encoder Reranking (Top 200)

The top 200 candidates are jointly re-scored using `cross-encoder/ms-marco-MiniLM-L-6-v2`:

- Joint (query, document) encoding is more accurate than bi-encoder
- Blended 25% cross-encoder + 75% hybrid score (keeps rule-based dominant)
- Only adds ~20 seconds of latency

### Skill Gate

A critical anti-gaming mechanism: if a candidate's must-have skill score is below 0.15, their semantic score is **crushed by 85%**. This prevents "well-written irrelevant" profiles (e.g., marketers who mention AI buzzwords) from ranking highly.

---

## 🛡️ Fairness & Safety

The system explicitly avoids ranking based on:
- ❌ Gender
- ❌ Age
- ❌ Religion or caste
- ❌ College prestige alone
- ❌ Location bias (unless job requires specific location)

A `fairness_report.json` is generated with every run, checking representation ratios across:
- **College tier** (Tier 1 / 2 / 3)
- **Location** (Metro / Tier 2 / Remote)
- **Experience band** (Junior / Mid / Senior)

---

## 📊 API Documentation

### `POST /rank`

```json
{
  "candidates": [...],
  "top_n": 100,
  "use_embeddings": true,
  "use_reranker": true
}
```

### `GET /health` — Health check
### `GET /config` — Current scoring configuration

---

## 📜 License

MIT

---

Built with ❤️ for the India Runs Data & AI Challenge.
