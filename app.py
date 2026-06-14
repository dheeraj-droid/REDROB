"""
Redrob AI — Intelligent Candidate Ranking Dashboard

A Streamlit web application for recruiters to visualize
candidate rankings and explore the scoring breakdown.

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import json
import io
import time
import plotly.graph_objects as go
import plotly.express as px

from jd_parser import get_jd_requirements
from honeypot_detector import detect_honeypot
from scorer import score_candidate, DIMENSION_WEIGHTS
from reasoning_generator import generate_reasoning


# ── Page Config ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Redrob AI - Candidate Ranker",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .main .block-container {
        padding-top: 2rem;
        max-width: 1400px;
    }

    /* Hero header */
    .hero-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2.5rem 3rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.05);
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    .hero-header h1 {
        color: #e94560;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .hero-header p {
        color: #a8b2d1;
        font-size: 1.1rem;
        font-weight: 300;
        margin: 0;
    }

    /* Stat cards */
    .stat-card {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(233,69,96,0.2);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .stat-card:hover {
        border-color: rgba(233,69,96,0.5);
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(233,69,96,0.1);
    }
    .stat-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #e94560, #f97316);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .stat-label {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.3rem;
    }

    /* Candidate card */
    .candidate-card {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1.8rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .candidate-card:hover {
        border-color: rgba(233,69,96,0.3);
        box-shadow: 0 8px 30px rgba(0,0,0,0.2);
    }

    .rank-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 40px;
        height: 40px;
        border-radius: 10px;
        font-weight: 800;
        font-size: 1.1rem;
        margin-right: 1rem;
    }
    .rank-top {
        background: linear-gradient(135deg, #e94560, #f97316);
        color: white;
    }
    .rank-mid {
        background: rgba(233,69,96,0.15);
        color: #e94560;
        border: 1px solid rgba(233,69,96,0.3);
    }
    .rank-low {
        background: rgba(100,116,139,0.15);
        color: #94a3b8;
        border: 1px solid rgba(100,116,139,0.2);
    }

    .score-bar {
        height: 6px;
        border-radius: 3px;
        background: rgba(255,255,255,0.06);
        overflow: hidden;
        margin-top: 0.5rem;
    }
    .score-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.8s ease;
    }

    /* Dimension labels */
    .dim-label {
        font-size: 0.75rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .dim-score {
        font-size: 0.95rem;
        font-weight: 700;
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #0f172a;
    }

    .stButton>button {
        background: linear-gradient(135deg, #e94560, #d63384);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(233,69,96,0.3);
    }

    /* Tags */
    .skill-tag {
        display: inline-block;
        background: rgba(233,69,96,0.1);
        color: #e94560;
        border: 1px solid rgba(233,69,96,0.2);
        border-radius: 6px;
        padding: 0.2rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 500;
        margin: 0.15rem;
    }
    .skill-tag-green {
        background: rgba(16,185,129,0.1);
        color: #10b981;
        border-color: rgba(16,185,129,0.2);
    }
    .skill-tag-yellow {
        background: rgba(245,158,11,0.1);
        color: #f59e0b;
        border-color: rgba(245,158,11,0.2);
    }
    .skill-tag-gray {
        background: rgba(100,116,139,0.1);
        color: #94a3b8;
        border-color: rgba(100,116,139,0.2);
    }

    /* Warning badge */
    .warning-badge {
        background: rgba(245,158,11,0.1);
        color: #f59e0b;
        border: 1px solid rgba(245,158,11,0.2);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }

    /* Honeypot badge */
    .honeypot-badge {
        background: rgba(239,68,68,0.1);
        color: #ef4444;
        border: 1px solid rgba(239,68,68,0.3);
        border-radius: 8px;
        padding: 0.4rem 0.8rem;
        font-size: 0.8rem;
        font-weight: 600;
    }

    div[data-testid="stExpander"] {
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        background: rgba(15,23,42,0.5);
    }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ──────────────────────────────────────────────────────

def create_radar_chart(dim_scores: dict) -> go.Figure:
    """Create a radar chart for dimension scores."""
    labels = {
        "role_fit": "Role Fit",
        "skills": "Skills",
        "career": "Career",
        "education": "Education",
        "behavioral": "Behavioral",
        "red_flags": "Red Flags",
    }

    cats = list(labels.values())
    vals = [dim_scores.get(k, 0) for k in labels.keys()]
    vals.append(vals[0])  # Close the polygon
    cats.append(cats[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals,
        theta=cats,
        fill='toself',
        fillcolor='rgba(233,69,96,0.15)',
        line=dict(color='#e94560', width=2),
        marker=dict(size=6, color='#e94560'),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor='rgba(255,255,255,0.08)',
                linecolor='rgba(255,255,255,0.08)',
                tickfont=dict(color='#64748b', size=10),
            ),
            angularaxis=dict(
                gridcolor='rgba(255,255,255,0.08)',
                linecolor='rgba(255,255,255,0.08)',
                tickfont=dict(color='#a8b2d1', size=12, family='Inter'),
            ),
        ),
        showlegend=False,
        margin=dict(t=30, b=30, l=60, r=60),
        height=320,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def create_score_distribution(ranked_data: list) -> go.Figure:
    """Create a score distribution chart."""
    scores = [r["final_score"] for r in ranked_data]
    ranks = list(range(1, len(scores) + 1))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ranks,
        y=scores,
        marker=dict(
            color=scores,
            colorscale=[[0, '#64748b'], [0.5, '#e94560'], [1, '#f97316']],
            line=dict(width=0),
        ),
        hovertemplate="Rank %{x}<br>Score: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(
            title="Rank",
            gridcolor='rgba(255,255,255,0.05)',
            tickfont=dict(color='#64748b'),
            titlefont=dict(color='#94a3b8'),
        ),
        yaxis=dict(
            title="Score",
            gridcolor='rgba(255,255,255,0.05)',
            tickfont=dict(color='#64748b'),
            titlefont=dict(color='#94a3b8'),
        ),
        margin=dict(t=20, b=50, l=60, r=20),
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def get_rank_class(rank: int) -> str:
    if rank <= 10:
        return "rank-top"
    elif rank <= 50:
        return "rank-mid"
    return "rank-low"


def get_score_color(score: float) -> str:
    if score >= 0.7:
        return "#10b981"
    elif score >= 0.4:
        return "#f59e0b"
    return "#ef4444"


def render_candidate_card(entry: dict, rank: int):
    """Render a single candidate card."""
    candidate = entry["candidate"]
    score_result = entry["score_result"]
    dim_scores = score_result["dimension_scores"]
    signals = score_result["signals"]
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    redrob = candidate.get("redrob_signals", {})

    rank_class = get_rank_class(rank)
    score_color = get_score_color(entry["final_score"])

    with st.expander(
        f"#{rank}  |  {profile.get('current_title', 'N/A')} at "
        f"{profile.get('current_company', 'N/A')}  |  "
        f"Score: {entry['final_score']:.4f}  |  "
        f"{profile.get('anonymized_name', 'N/A')}",
        expanded=rank <= 3,
    ):
        col1, col2 = st.columns([2, 1])

        with col1:
            # Profile summary
            st.markdown(f"**{profile.get('anonymized_name', 'N/A')}** - "
                       f"{profile.get('headline', '')}")
            st.markdown(f"*{profile.get('summary', '')[:200]}...*"
                       if len(profile.get('summary', '')) > 200
                       else f"*{profile.get('summary', '')}*")

            # Key metrics
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Experience", f"{profile.get('years_of_experience', 0):.1f} yrs")
            mcol2.metric("Location", f"{profile.get('location', 'N/A')}")
            mcol3.metric("Response Rate", f"{redrob.get('recruiter_response_rate', 0):.0%}")
            mcol4.metric("Notice", f"{redrob.get('notice_period_days', 'N/A')} days")

            # Skills
            st.markdown("**Skills:**")
            from skill_taxonomy import classify_skill, get_category_relevance
            skill_html = ""
            for s in skills:
                cat = classify_skill(s.get("name", ""))
                rel = get_category_relevance(cat)
                if rel >= 0.8:
                    tag_class = "skill-tag"
                elif rel >= 0.4:
                    tag_class = "skill-tag skill-tag-green"
                elif rel >= 0.2:
                    tag_class = "skill-tag skill-tag-yellow"
                else:
                    tag_class = "skill-tag skill-tag-gray"
                skill_html += f'<span class="{tag_class}">{s.get("name", "")} ({s.get("proficiency", "")[0].upper()})</span> '
            st.markdown(skill_html, unsafe_allow_html=True)

            # Reasoning
            reasoning = generate_reasoning(candidate, score_result, rank)
            st.info(f"**Reasoning:** {reasoning}")

            # Honeypot check
            if entry.get("is_honeypot"):
                st.markdown('<div class="honeypot-badge">HONEYPOT DETECTED - Profile flagged as potentially impossible</div>', unsafe_allow_html=True)

        with col2:
            # Radar chart
            fig = create_radar_chart(dim_scores)
            st.plotly_chart(fig, use_container_width=True, key=f"radar_{rank}")

            # Score breakdown
            for dim_name, dim_label in [
                ("role_fit", "Role Fit"),
                ("skills", "Skills"),
                ("career", "Career"),
                ("behavioral", "Behavioral"),
                ("red_flags", "Red Flags"),
            ]:
                val = dim_scores.get(dim_name, 0)
                color = get_score_color(val)
                weight = DIMENSION_WEIGHTS.get(dim_name, 0)
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                    f'<span class="dim-label">{dim_label} ({weight:.0%})</span>'
                    f'<span class="dim-score" style="color:{color}">{val:.2f}</span>'
                    f'</div>'
                    f'<div class="score-bar"><div class="score-fill" style="width:{val*100}%;background:{color}"></div></div>',
                    unsafe_allow_html=True,
                )


# ── Main App ──────────────────────────────────────────────────────────────

def main():
    # Hero header
    st.markdown("""
    <div class="hero-header">
        <h1>Redrob AI Candidate Ranker</h1>
        <p>Intelligent candidate discovery beyond keyword matching - analyzing career fit, skill depth, behavioral signals, and more.</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### Configuration")

        jd = get_jd_requirements()
        st.markdown(f"**Target Role:** {jd.title}")
        st.markdown(f"**Company:** {jd.company} ({jd.stage})")
        st.markdown(f"**Experience:** {jd.min_years}-{jd.max_years} years")
        st.markdown("---")

        # File upload
        uploaded_file = st.file_uploader(
            "Upload candidates (JSONL or JSON)",
            type=["jsonl", "json"],
            help="Upload a JSONL file with candidate profiles"
        )

        use_sample = st.checkbox("Use sample data (10 candidates)", value=True)

        top_n = st.slider("Top N candidates", 5, 100, 20)

        run_btn = st.button("Run Ranking", type="primary")

        st.markdown("---")
        st.markdown("### Scoring Weights")
        for dim, weight in DIMENSION_WEIGHTS.items():
            st.markdown(f"**{dim.replace('_', ' ').title()}:** {weight:.0%}")

    # Main content
    if run_btn or "ranked_data" in st.session_state:
        if run_btn:
            candidates = []

            if uploaded_file is not None:
                content = uploaded_file.read().decode("utf-8")
                if uploaded_file.name.endswith(".jsonl"):
                    for line in content.strip().split("\n"):
                        if line.strip():
                            candidates.append(json.loads(line))
                else:
                    data = json.loads(content)
                    if isinstance(data, list):
                        candidates = data
                    else:
                        candidates = [data]
            elif use_sample:
                # Load sample candidates
                import os
                sample_paths = [
                    "sample_candidates.json",
                    "[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/sample_candidates.json",
                ]
                for sp in sample_paths:
                    if os.path.exists(sp):
                        with open(sp, "r", encoding="utf-8") as f:
                            candidates = json.load(f)
                        break

            if not candidates:
                st.warning("No candidates loaded. Upload a file or use sample data.")
                return

            # Score candidates
            progress = st.progress(0, text="Scoring candidates...")
            ranked_data = []

            for i, cand in enumerate(candidates):
                is_hp, hp_reasons = detect_honeypot(cand)
                result = score_candidate(cand, jd)
                final = result["final_score"]
                if is_hp:
                    final *= 0.05

                ranked_data.append({
                    "candidate_id": cand.get("candidate_id", f"CAND_{i:07d}"),
                    "candidate": cand,
                    "score_result": result,
                    "final_score": final,
                    "is_honeypot": is_hp,
                })
                progress.progress((i + 1) / len(candidates),
                                text=f"Scoring {i+1}/{len(candidates)}...")

            ranked_data.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
            ranked_data = ranked_data[:top_n]
            st.session_state["ranked_data"] = ranked_data
            progress.empty()

        ranked_data = st.session_state.get("ranked_data", [])

        if ranked_data:
            # Stats row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{len(ranked_data)}</div>
                    <div class="stat-label">Candidates Ranked</div>
                </div>""", unsafe_allow_html=True)
            with col2:
                avg_score = sum(r["final_score"] for r in ranked_data) / len(ranked_data)
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{avg_score:.3f}</div>
                    <div class="stat-label">Avg Score</div>
                </div>""", unsafe_allow_html=True)
            with col3:
                hp_count = sum(1 for r in ranked_data if r["is_honeypot"])
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{hp_count}</div>
                    <div class="stat-label">Honeypots Caught</div>
                </div>""", unsafe_allow_html=True)
            with col4:
                top_score = ranked_data[0]["final_score"] if ranked_data else 0
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{top_score:.3f}</div>
                    <div class="stat-label">Top Score</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Score distribution chart
            st.markdown("### Score Distribution")
            fig = create_score_distribution(ranked_data)
            st.plotly_chart(fig, use_container_width=True)

            # Candidate cards
            st.markdown("### Ranked Candidates")

            for i, entry in enumerate(ranked_data):
                render_candidate_card(entry, i + 1)

            # Export
            st.markdown("---")
            st.markdown("### Export")

            csv_data = "candidate_id,rank,score,reasoning\n"
            for i, entry in enumerate(ranked_data):
                reasoning = generate_reasoning(
                    entry["candidate"], entry["score_result"], i + 1
                )
                score = 0.999 - i * (0.799 / max(len(ranked_data) - 1, 1))
                csv_data += f'{entry["candidate_id"]},{i+1},{score:.4f},"{reasoning}"\n'

            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="submission.csv",
                mime="text/csv",
            )
    else:
        # Landing state
        st.markdown("""
        <div style="text-align:center; padding: 4rem 2rem;">
            <h2 style="color:#e94560; font-weight:700;">Ready to Rank</h2>
            <p style="color:#64748b; font-size:1.1rem; max-width:600px; margin:1rem auto;">
                Upload candidate profiles or use the sample data, then click
                <strong>Run Ranking</strong> to see AI-powered candidate scoring in action.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Feature cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="stat-card" style="text-align:left; padding:1.5rem;">
                <div style="color:#e94560; font-weight:700; font-size:1.1rem; margin-bottom:0.5rem;">Semantic Understanding</div>
                <div style="color:#94a3b8; font-size:0.9rem;">Goes beyond keywords to understand career trajectories, skill depth, and genuine role fit.</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="stat-card" style="text-align:left; padding:1.5rem;">
                <div style="color:#f97316; font-weight:700; font-size:1.1rem; margin-bottom:0.5rem;">Trap Detection</div>
                <div style="color:#94a3b8; font-size:0.9rem;">7 heuristic checks catch honeypot candidates with impossible profiles and keyword-stuffing.</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="stat-card" style="text-align:left; padding:1.5rem;">
                <div style="color:#10b981; font-weight:700; font-size:1.1rem; margin-bottom:0.5rem;">Behavioral Signals</div>
                <div style="color:#94a3b8; font-size:0.9rem;">Weighs platform engagement, response rates, and availability for recruiter-ready shortlists.</div>
            </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
