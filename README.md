# 🤖 Redrob AI — Intelligent Candidate Discovery & Ranking System

> An AI-powered candidate ranking system that goes beyond keyword matching to understand who genuinely fits a role — analyzing career trajectories, skill depth, behavioral signals, and detecting trap profiles.

Built for the **India Runs Data & AI Challenge** by Redrob.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    rank.py (Main Pipeline)                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐   ┌──────────────┐   ┌───────────────────┐   │
│  │ JD Parser │   │  Honeypot    │   │  Skill Taxonomy   │   │
│  │           │   │  Detector    │   │  (150+ skills     │   │
│  │ Structured│   │  (7 heuristic│   │   mapped to       │   │
│  │ JD reqs   │   │   checks)   │   │   categories)     │   │
│  └─────┬─────┘   └──────┬──────┘   └────────┬──────────┘   │
│        │                │                    │              │
│        ▼                ▼                    ▼              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Multi-Dimensional Scorer                │   │
│  │                                                      │   │
│  │  Role Fit (30%) │ Skills (25%) │ Career (15%)        │   │
│  │  Education (5%) │ Behavioral (15%) │ Red Flags (10%) │   │
│  └──────────────────────────┬───────────────────────────┘   │
│                             │                               │
│                             ▼                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           Reasoning Generator                        │   │
│  │  (Specific, honest, per-candidate explanations)      │   │
│  └──────────────────────────┬───────────────────────────┘   │
│                             │                               │
│                             ▼                               │
│                      submission.csv                         │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 Key Design Decisions

### 1. **Beyond Keywords: Semantic Role Understanding**
The JD explicitly warns that "keyword matching is a trap." Our system:
- Uses a **hand-crafted skill taxonomy** (150+ skills → 11 categories) for semantic matching
- Checks **title-career coherence** (a "Marketing Manager" with 10 AI skills is a keyword-stuffer)
- Analyzes **career descriptions** for actual AI/ML work signals
- Weighs **proficiency × duration × endorsements** as a trust multiplier

### 2. **Honeypot Detection (7 Heuristic Checks)**
- Experience duration vs. career history mismatch
- Expert proficiency with zero duration or endorsements
- Career timeline impossibilities (end date before start date)
- Education impossibilities (PhD in 1 year)
- Signal anomalies (high completeness, zero verifications)
- Title-description mismatch
- Suspicious skill accumulation patterns

### 3. **Behavioral Signals as Availability Multiplier**
A perfect-on-paper candidate who hasn't logged in for 6 months isn't actually available:
- `recruiter_response_rate` as primary availability signal
- `last_active_date` recency
- `open_to_work_flag`, `notice_period_days`
- `interview_completion_rate`, `github_activity_score`

### 4. **Red Flag Penalty System (Multiplicative)**
Red flags don't just reduce score — they multiply against it:
- Entire career at consulting firms (TCS, Infosys, Wipro, etc.)
- Title-chasing pattern (new company every 1-1.5 years)
- Domain mismatch (CV/speech without NLP/IR experience)
- Keyword-stuffer detection (non-tech title + AI keyword dump)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- No external packages required for the ranking pipeline

### Run the Ranker
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/redrob-ai-ranker.git
cd redrob-ai-ranker

# Run ranking (no dependencies needed!)
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# With verbose output
python rank.py --candidates ./candidates.jsonl --out ./submission.csv --verbose
```

### Validate Submission
```bash
python validate_submission.py submission.csv
```

### Run the Web Dashboard
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

```
redrob-ai-ranker/
├── rank.py                  # Main pipeline entry point
├── jd_parser.py             # JD → structured requirements
├── skill_taxonomy.py        # Semantic skill mapping (150+ skills)
├── honeypot_detector.py     # Trap candidate detection (7 checks)
├── scorer.py                # Multi-dimensional scoring engine
├── reasoning_generator.py   # Per-candidate reasoning strings
├── app.py                   # Streamlit web dashboard
├── requirements.txt         # Dependencies
├── submission.csv           # Generated output
├── submission_metadata.yaml # Hackathon metadata
└── README.md                # This file
```

---

## 📊 Scoring Dimensions

| Dimension | Weight | What It Captures |
|---|---|---|
| **Role Fit** | 30% | Title relevance, AI/ML career trajectory, YoE fit, product-company ratio |
| **Skills Match** | 25% | Semantic skill coverage, proficiency × duration trust, must-have cluster coverage |
| **Career Quality** | 15% | Product vs. consulting, tenure stability, progression, hands-on coding signals |
| **Education** | 5% | Institution tier, field relevance, degree level |
| **Behavioral** | 15% | Availability (open_to_work, response rate, recency, notice, GitHub, location) |
| **Red Flags** | 10% | Multiplicative penalty for keyword-stuffers, consulting-only, title-chasers |

---

## 🔬 Methodology

### Why No LLM / Neural Model?
The compute constraints (5 min, CPU-only, 16 GB, no network) make LLM-per-candidate scoring infeasible for 100K candidates. Instead, we use:

1. **Feature engineering** on structured candidate data
2. **Hand-crafted skill taxonomy** for semantic matching
3. **Rule-based career analysis** from descriptions
4. **Weighted scoring** with multiplicative red-flag penalties

This approach runs the full 100K pipeline in **~60 seconds** on a standard CPU — well within the 5-minute budget — while being fully interpretable and producing specific reasoning for every candidate.

### The Philosophy
> "The right answer involves reasoning about the gap between what the JD says and what the JD means."

Our system doesn't just check if skills match — it asks:
1. Is this person *actually* an AI/ML engineer? (title + career coherence)
2. Do they have *real* depth? (proficiency × duration trust)
3. Have they shipped *production* systems? (product-company, description analysis)
4. Are they *actually available*? (behavioral signals)
5. Are there *red flags*? (consulting-only, keyword-stuffing, domain mismatch)

---

## 🛠️ AI Tools Declaration

- **Antigravity IDE (Claude)** — Used for architecture discussion, code generation, and iterative development
- All candidate scoring logic is original engineering work
- No candidate data was fed to any external LLM during ranking

---

## 📝 License

MIT License — built for the Redrob India Runs Hackathon.
