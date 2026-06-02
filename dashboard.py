import os
import sys
import json
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Make sure our modules are importable
sys.path.insert(0, os.path.dirname(__file__))

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Sleep Scorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Colours ────────────────────────────────────────────────────────────────
STAGE_COLOURS = {
    "Wake": "#E74C3C",
    "N1":   "#F39C12",
    "N2":   "#3498DB",
    "N3":   "#2C3E50",
    "REM":  "#9B59B6",
}
STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
STAGE_CODES = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "outputs")
DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1F4E79;
        margin-bottom: 0;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #666;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1F4E79, #2E75B6);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.85;
    }
    .stage-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        color: white;
    }
    .override-box {
        background: #FFF3CD;
        border-left: 4px solid #F39C12;
        padding: 10px 14px;
        border-radius: 4px;
        margin: 4px 0;
        font-size: 0.85rem;
    }
    .accepted-box {
        background: #D4EDDA;
        border-left: 4px solid #28A745;
        padding: 10px 14px;
        border-radius: 4px;
        margin: 4px 0;
        font-size: 0.85rem;
    }
    .clinical-box {
        background: #F8F9FA;
        border: 1px solid #DEE2E6;
        border-radius: 8px;
        padding: 20px;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        line-height: 1.6;
    }
    div[data-testid="stProgress"] > div > div {
        background-color: #2E75B6;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ────────────────────────────────────────────────────────

@st.cache_data
def load_features():
    path = os.path.join(DATA_DIR, "features.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_agent_predictions():
    path = os.path.join(OUTPUT_DIR, "agent_predictions.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_ml_metrics():
    path = os.path.join(OUTPUT_DIR, "ml_metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

@st.cache_data
def load_comparison():
    path = os.path.join(OUTPUT_DIR, "comparison_table.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_clinical_summary():
    path = os.path.join(OUTPUT_DIR, "clinical_summary.txt")
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return None

@st.cache_data
def load_run_config():
    path = os.path.join(OUTPUT_DIR, "run_config.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def stage_badge(stage):
    colour = STAGE_COLOURS.get(stage, "#888")
    return f'<span class="stage-badge" style="background:{colour}">{stage}</span>'

def run_pipeline_fresh(n_epochs, subject_idx=0, custom_prompt=None):
    """Run the pipeline and stream progress back."""
    from env.feature_extractor  import load_and_extract
    from rl.ml_baseline         import train_and_evaluate
    from agent.llm_scorer       import score_epochs
    from agent.agentic_reviewer import SleepTools, review_epochs, generate_clinical_summary
    from agent.evaluator        import evaluate_three_way, qualitative_analysis

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR,   exist_ok=True)

    yield "step", "step1", "⚙️ Step 1/5 — Extracting EEG features..."
    df = load_and_extract(data_dir=DATA_DIR, subject_idx=subject_idx)
    yield "data", "features", df
    yield "done", "step1", f"✅ Feature extraction complete — {len(df)} epochs"

    yield "step", "step2", "🌲 Step 2/5 — Training Random Forest baseline..."
    clf, ml_metrics, ml_pred_df = train_and_evaluate(df, test_size=0.4,
                                                      random_state=42,
                                                      output_dir=OUTPUT_DIR)
    yield "data", "ml_metrics", ml_metrics
    yield "done", "step2", (f"✅ ML Baseline — Accuracy: {ml_metrics['accuracy']:.1%}  "
                             f"Macro F1: {ml_metrics['macro_f1']:.3f}")

    yield "step", "step3", "🤖 Step 3/5 — Running LLM scorer..."
    # Apply custom prompt if provided
    if custom_prompt:
        import agent.llm_scorer as _scorer
        _scorer.SYSTEM_PROMPT = custom_prompt
    llm_df, parse_rate = score_epochs(df, max_epochs=n_epochs)
    yield "data", "llm_df", llm_df
    yield "done", "step3", f"✅ LLM scorer complete — Parse rate: {parse_rate:.1%}"

    yield "step", "step4", "🔍 Step 4/5 — Agentic reviewer..."
    tools    = SleepTools(df, llm_df)
    agent_df = review_epochs(df, llm_df, tools, ollama_available=False)
    tools.scored_df = agent_df
    summary  = generate_clinical_summary(agent_df, tools)
    yield "data", "agent_df", agent_df
    yield "data", "summary", summary
    n_ov = int(agent_df["was_overridden"].sum())
    yield "done", "step4", f"✅ Agent review complete — {n_ov} overrides ({n_ov/max(len(agent_df),1):.1%})"

    yield "step", "step5", "📊 Step 5/5 — 3-way comparison..."
    comparison_df, _ = evaluate_three_way(df, clf, agent_df, output_dir=OUTPUT_DIR)
    qualitative_analysis(agent_df, n=10, output_dir=OUTPUT_DIR)
    yield "data", "comparison", comparison_df
    yield "done", "step5", "✅ Evaluation complete"

    yield "finished", "", ""


# ── SIDEBAR ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 LLM Sleep Scorer")
    st.markdown("*Special Topic in AI — GUtech*")
    st.divider()

    st.markdown("### ⚙️ Settings")

    subject_idx = st.selectbox(
        "Patient (subject index):",
        options=[0, 1, 2, 3, 4, 5],
        index=0,
        help="Each number is a different patient from Sleep-EDF Expanded"
    )

    n_epochs = st.slider(
        "Epochs to score",
        min_value=20, max_value=150, value=120, step=10,
        help="How many 30-second epochs to send to the LLM scorer (80-150 recommended)"
    )

    st.divider()

    run_button = st.button(
        "▶\u25b6\ufe0f Run Full Pipeline",
        type="primary",
        use_container_width=True
    )

    st.divider()
    st.markdown("### 📁 Project Layers")
    st.markdown("""
    **Layer 1** — Feature Extractor  
    `env/feature_extractor.py`
    
    **Layer 2** — ML Baseline  
    `rl/ml_baseline.py`
    
    **Layer 3a** — LLM Scorer  
    `agent/llm_scorer.py`
    
    **Layer 3b** — Agentic Reviewer  
    `agent/agentic_reviewer.py`
    """)

    st.divider()
    cfg = load_run_config()
    if cfg:
        st.markdown("### 📋 Last Run")
        st.markdown(f"**Epochs scored:** {cfg.get('epochs_scored', '—')}")
        st.markdown(f"**ML Accuracy:** {cfg.get('ml_accuracy', 0):.1%}")
        st.markdown(f"**Parse rate:** {cfg.get('llm_parse_rate', 0):.1%}")
        st.markdown(f"**Agent overrides:** {cfg.get('n_agent_overrides', '—')}")
        st.markdown(f"**Runtime:** {cfg.get('elapsed_seconds', '—')}s")


# ── MAIN AREA ───────────────────────────────────────────────────────────────

st.markdown('<p class="main-title">🧠 LLM Sleep Scorer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Automated EEG sleep staging — comparing ML Baseline vs Raw LLM vs Agent-Reviewed</p>', unsafe_allow_html=True)

# ── RUN PIPELINE ────────────────────────────────────────────────────────────

if run_button:
    st.markdown("---")
    st.markdown("## 🚀 Running Pipeline")

    progress_bar  = st.progress(0)
    status_text   = st.empty()
    step_statuses = {}

    col1, col2, col3, col4, col5 = st.columns(5)
    step_cols = [col1, col2, col3, col4, col5]
    step_labels = ["Features", "ML Baseline", "LLM Scorer", "Agent", "Evaluation"]
    step_placeholders = [c.empty() for c in step_cols]

    for i, (ph, label) in enumerate(zip(step_placeholders, step_labels)):
        ph.markdown(f"⬜ {label}")

    collected = {}
    step_num  = 0

    for kind, key, value in run_pipeline_fresh(n_epochs, subject_idx, custom_prompt=st.session_state.get("active_prompt", None)):
        if kind == "step":
            status_text.markdown(f"**{value}**")
            if step_num < 5:
                step_placeholders[step_num].markdown(f"🔄 {step_labels[step_num]}")
            progress_bar.progress(step_num * 20)

        elif kind == "data":
            collected[key] = value

        elif kind == "done":
            status_text.markdown(f"**{value}**")
            if step_num < 5:
                step_placeholders[step_num].markdown(f"✅ {step_labels[step_num]}")
            step_num += 1
            progress_bar.progress(step_num * 20)

        elif kind == "finished":
            progress_bar.progress(100)
            status_text.markdown("### ✅ Pipeline complete!")
            # clear the cache so tabs reload fresh data
            st.cache_data.clear()
            time.sleep(0.5)
            st.rerun()

st.markdown("---")

# ── TABS ────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "🌲 ML Baseline",
    "🤖 LLM + Agent",
    "📈 Comparison",
    "🏥 Clinical Report",
    "✏️ Prompt Lab"
])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("## System Overview")

    features_df = load_features()
    cfg         = load_run_config()
    ml_metrics  = load_ml_metrics()

    if features_df is None:
        st.info("👈 Click **Run Full Pipeline** in the sidebar to get started.")
    else:
        # Top metrics row
        c1, c2, c3, c4, c5 = st.columns(5)
        total = len(features_df)
        counts = features_df["stage_name"].value_counts()

        with c1:
            st.metric("Total Epochs", total)
        with c2:
            st.metric("Epochs Scored", cfg.get("epochs_scored", "—"))
        with c3:
            acc = cfg.get("ml_accuracy", None)
            st.metric("ML Accuracy", f"{acc:.1%}" if acc else "—")
        with c4:
            pr = cfg.get("llm_parse_rate", None)
            st.metric("LLM Parse Rate", f"{pr:.1%}" if pr else "—")
        with c5:
            ov = cfg.get("n_agent_overrides", None)
            st.metric("Agent Overrides", ov if ov is not None else "—")

        st.markdown("---")

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("### Stage Distribution in Dataset")
            stage_order = ["Wake", "N1", "N2", "N3", "REM"]
            stage_counts = [counts.get(s, 0) for s in stage_order]
            colours      = [STAGE_COLOURS[s] for s in stage_order]

            fig = go.Figure(go.Bar(
                x=stage_order,
                y=stage_counts,
                marker_color=colours,
                text=stage_counts,
                textposition="outside",
            ))
            fig.update_layout(
                title="Number of Epochs per Stage",
                xaxis_title="Sleep Stage",
                yaxis_title="Epoch Count",
                plot_bgcolor="white",
                height=350,
                showlegend=False,
                margin=dict(t=50, b=40, l=40, r=20)
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridcolor="#EEE")
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown("### Stage Proportions")
            fig2 = go.Figure(go.Pie(
                labels=stage_order,
                values=stage_counts,
                marker_colors=colours,
                hole=0.4,
                textinfo="label+percent",
            ))
            fig2.update_layout(
                height=350,
                margin=dict(t=30, b=10, l=10, r=10),
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.markdown("### Feature Value Distributions")

        feat_cols = ["delta_power", "theta_power", "alpha_power", "beta_power",
                     "spindle_ratio", "eog_variance", "emg_variance", "signal_entropy"]

        selected_feat = st.selectbox("Select feature to visualise:", feat_cols)

        fig3 = go.Figure()
        for stage in stage_order:
            subset = features_df[features_df["stage_name"] == stage][selected_feat]
            fig3.add_trace(go.Box(
                y=subset,
                name=stage,
                marker_color=STAGE_COLOURS[stage],
                boxmean=True,
            ))
        fig3.update_layout(
            title=f"{selected_feat} distribution per sleep stage",
            yaxis_title=selected_feat,
            plot_bgcolor="white",
            height=380,
            margin=dict(t=50, b=40, l=50, r=20)
        )
        fig3.update_yaxes(showgrid=True, gridcolor="#EEE")
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown(
            "*Box plots show the median, interquartile range, and outliers for each stage. "
            "Features that separate cleanly between stages are the most informative.*",
            unsafe_allow_html=False
        )


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — ML BASELINE
# ══════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("## 🌲 ML Baseline — Random Forest")

    ml_metrics = load_ml_metrics()

    if ml_metrics is None:
        st.info("Run the pipeline first to see ML baseline results.")
    else:
        st.markdown("### Performance Metrics")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Overall Accuracy",
                      f"{ml_metrics['accuracy']:.1%}")
        with c2:
            st.metric("Macro F1",
                      f"{ml_metrics['macro_f1']:.3f}")
        with c3:
            st.metric("Training Epochs",
                      ml_metrics.get("n_train", "—"))
        with c4:
            st.metric("Test Epochs",
                      ml_metrics.get("n_test", "—"))

        st.markdown("---")

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### Per-Class F1 Scores")
            pcf1 = ml_metrics.get("per_class_f1", {})
            stages = [s for s in ["Wake", "N1", "N2", "N3", "REM"] if s in pcf1]
            f1_vals = [pcf1[s] for s in stages]
            colours  = [STAGE_COLOURS[s] for s in stages]

            fig = go.Figure(go.Bar(
                x=stages, y=f1_vals,
                marker_color=colours,
                text=[f"{v:.3f}" for v in f1_vals],
                textposition="outside",
            ))
            fig.add_hline(y=0.7, line_dash="dash", line_color="red",
                          annotation_text="0.7 target")
            fig.update_layout(
                yaxis=dict(range=[0, 1.1], title="F1 Score"),
                xaxis_title="Sleep Stage",
                plot_bgcolor="white",
                height=360,
                showlegend=False,
                margin=dict(t=30, b=40, l=50, r=20)
            )
            fig.update_yaxes(showgrid=True, gridcolor="#EEE")
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown("### Saved Confusion Matrix")
            cm_path = os.path.join(OUTPUT_DIR, "ml_confusion_matrix.png")
            if os.path.exists(cm_path):
                st.image(cm_path, use_container_width=True)
            else:
                st.info("Run pipeline to generate confusion matrix image.")

        st.markdown("---")
        st.markdown("### Feature Importance")
        fi_path = os.path.join(OUTPUT_DIR, "ml_feature_importance.png")
        if os.path.exists(fi_path):
            st.image(fi_path, use_container_width=True)

        with st.expander("ℹ️ About the Random Forest Baseline"):
            st.markdown("""
            **Model:** RandomForestClassifier — 200 trees, max depth 12, balanced class weights

            **Why balanced class weights?**
            N1 epochs are rare (~5% of a night). Without balancing, the model would learn to
            ignore N1 entirely and still get high accuracy. Balanced weights force it to treat
            each stage equally during training.

            **Why 40% test split?**
            The spec requires 80–150 held-out epochs for the LLM comparison. 40% of 200 epochs
            gives approximately 80 test epochs, meeting this requirement.

            **Evaluation:** Accuracy, per-class F1, macro F1, confusion matrix.
            F1 is more meaningful than accuracy when classes are imbalanced.
            """)


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — LLM + AGENT
# ══════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("## 🤖 LLM Scorer + Agentic Reviewer")

    agent_df = load_agent_predictions()

    if agent_df is None:
        st.info("Run the pipeline first to see epoch-by-epoch results.")
    else:
        # Summary row
        n_total    = len(agent_df)
        n_correct  = int((agent_df["agent_stage"] == agent_df["true_label"].map(STAGE_NAMES)).sum())
        n_override = int(agent_df["was_overridden"].sum())
        n_llm_correct = int((agent_df["llm_stage"] == agent_df["true_label"].map(STAGE_NAMES)).sum())

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Epochs Reviewed", n_total)
        with c2:
            st.metric("Agent Correct",
                      f"{n_correct/max(n_total,1):.1%}",
                      delta=f"{(n_correct-n_llm_correct)/max(n_total,1):+.1%} vs LLM")
        with c3:
            st.metric("Overrides Made", n_override,
                      delta=f"{n_override/max(n_total,1):.1%} of epochs")
        with c4:
            # override accuracy
            ov_df = agent_df[agent_df["was_overridden"]]
            if len(ov_df) > 0:
                ov_correct = int((ov_df["agent_stage"] == ov_df["true_label"].map(STAGE_NAMES)).sum())
                st.metric("Override Accuracy",
                          f"{ov_correct/len(ov_df):.1%}",
                          help="% of overrides that were correct")
            else:
                st.metric("Override Accuracy", "—")

        st.markdown("---")

        # Hypnogram
        st.markdown("### Sleep Stage Timeline (Hypnogram)")

        fig_hyp = go.Figure()
        stage_to_y = {"Wake": 4, "REM": 3, "N1": 2, "N2": 1, "N3": 0}

        # True stages
        true_stages  = agent_df["true_label"].map(STAGE_NAMES).tolist()
        true_y       = [stage_to_y.get(s, 1) for s in true_stages]
        agent_stages = agent_df["agent_stage"].tolist()
        agent_y      = [stage_to_y.get(s, 1) for s in agent_stages]

        fig_hyp.add_trace(go.Scatter(
            x=list(range(len(true_stages))), y=true_y,
            mode="lines", name="True Stage",
            line=dict(color="#1F4E79", width=2),
            hovertemplate="Epoch %{x}<br>True: %{text}<extra></extra>",
            text=true_stages
        ))
        fig_hyp.add_trace(go.Scatter(
            x=list(range(len(agent_stages))), y=agent_y,
            mode="lines", name="Agent-Reviewed",
            line=dict(color="#E74C3C", width=1.5, dash="dot"),
            hovertemplate="Epoch %{x}<br>Agent: %{text}<extra></extra>",
            text=agent_stages
        ))

        # Mark overrides
        ov_idx = agent_df[agent_df["was_overridden"]].index.tolist()
        ov_x   = [i for i in range(len(agent_df)) if agent_df.iloc[i]["was_overridden"]]
        ov_y   = [stage_to_y.get(agent_df.iloc[i]["agent_stage"], 1) for i in ov_x]

        if ov_x:
            fig_hyp.add_trace(go.Scatter(
                x=ov_x, y=ov_y,
                mode="markers", name="Override",
                marker=dict(color="#F39C12", size=10, symbol="star"),
                hovertemplate="Epoch %{x}<br>OVERRIDE<extra></extra>",
            ))

        fig_hyp.update_layout(
            yaxis=dict(
                tickvals=[0, 1, 2, 3, 4],
                ticktext=["N3", "N2", "N1", "REM", "Wake"],
                title="Sleep Stage"
            ),
            xaxis_title="Epoch Number",
            height=320,
            plot_bgcolor="white",
            legend=dict(orientation="h", y=1.12),
            margin=dict(t=40, b=40, l=80, r=20)
        )
        fig_hyp.update_yaxes(showgrid=True, gridcolor="#EEE")
        fig_hyp.update_xaxes(showgrid=False)
        st.plotly_chart(fig_hyp, use_container_width=True)
        st.caption("Blue = true stage | Red dotted = agent prediction | ⭐ = override event")

        st.markdown("---")

        # Epoch-by-epoch table with filters
        st.markdown("### Epoch-by-Epoch Decisions")

        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            show_only = st.selectbox(
                "Filter by:", ["All epochs", "Overrides only", "Correct only", "Incorrect only"]
            )
        with col_f2:
            stage_filter = st.multiselect(
                "Filter by true stage:",
                ["Wake", "N1", "N2", "N3", "REM"],
                default=["Wake", "N1", "N2", "N3", "REM"]
            )
        with col_f3:
            n_show = st.slider("Show N epochs:", 10, min(150, len(agent_df)), 30)

        display_df = agent_df.copy()
        display_df["true_stage_name"] = display_df["true_label"].map(STAGE_NAMES)
        display_df = display_df[display_df["true_stage_name"].isin(stage_filter)]

        if show_only == "Overrides only":
            display_df = display_df[display_df["was_overridden"]]
        elif show_only == "Correct only":
            display_df = display_df[display_df["agent_stage"] == display_df["true_stage_name"]]
        elif show_only == "Incorrect only":
            display_df = display_df[display_df["agent_stage"] != display_df["true_stage_name"]]

        display_df = display_df.head(n_show)

        for _, row in display_df.iterrows():
            true_s  = STAGE_NAMES.get(int(row["true_label"]), "?")
            llm_s   = str(row["llm_stage"])
            agent_s = str(row["agent_stage"])
            correct = agent_s == true_s
            overridden = bool(row["was_overridden"])

            correct_icon = "✅" if correct else "❌"
            override_str = f"🔄 OVERRIDDEN ({llm_s} → {agent_s})" if overridden else f"✔ ACCEPTED ({agent_s})"
            box_class    = "override-box" if overridden else "accepted-box"

            just = str(row.get("agent_just", ""))[:200]

            st.markdown(f"""
            <div class="{box_class}">
              <strong>Epoch {int(row['epoch_id'])}</strong> &nbsp;|&nbsp;
              True: <strong>{true_s}</strong> &nbsp;|&nbsp;
              {override_str} &nbsp;|&nbsp; {correct_icon}
              <br><small>📝 {just}</small>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Agent tools explanation
        with st.expander("🔧 About the 6 Agent Tools"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("""
                **Per-Epoch Tools (called once per epoch):**

                1. `get_features(epoch_id)`
                   Retrieves the 8 raw feature values so the agent
                   can verify the base scorer's justification.

                2. `compare_to_neighbors(epoch_id, n=3)`
                   Checks the 3 epochs before and after for temporal
                   context. Sleep stages follow predictable sequences.

                3. `lookup_stage_definition(stage)`
                   Retrieves the AASM definition to check whether
                   features are actually consistent with the claimed stage.
                """)
            with col_b:
                st.markdown("""
                **End-of-Night Tools (called once at end):**

                4. `count_stage_total(stage)`
                   Counts epochs per stage and estimates minutes.
                   Compares to normative healthy adult values.

                5. `find_transitions()`
                   Counts stage changes and finds unusual sequences
                   (e.g. N3→REM). Computes fragmentation index.

                6. `compare_to_baseline_night()`
                   Flags any stage that deviates more than 10%
                   from a typical healthy adult night.
                """)


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — 3-WAY COMPARISON
# ══════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("## 📈 3-Way Comparison")

    comp_df = load_comparison()

    if comp_df is None:
        st.info("Run the pipeline first to see comparison results.")
    else:
        st.markdown("### Overall Performance Table")

        # Highlight best in each column
        display_comp = comp_df.copy()
        st.dataframe(
            display_comp.style
                .highlight_max(subset=[c for c in display_comp.columns if c != "method"],
                               color="#D4EDDA")
                .format({c: "{:.3f}" for c in display_comp.columns if c != "method"}),
            use_container_width=True,
            hide_index=True
        )
        st.caption("Green highlight = best value in each column.")

        st.markdown("---")

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### Accuracy Comparison")
            methods = comp_df["method"].tolist()
            accs    = comp_df["accuracy"].tolist()
            fig_acc = go.Figure(go.Bar(
                x=methods, y=accs,
                marker_color=["#2196F3", "#FF9800", "#4CAF50"],
                text=[f"{a:.1%}" for a in accs],
                textposition="outside",
            ))
            fig_acc.update_layout(
                yaxis=dict(range=[0, 1.1], title="Accuracy"),
                plot_bgcolor="white", height=320,
                showlegend=False,
                margin=dict(t=30, b=80, l=50, r=20)
            )
            fig_acc.update_yaxes(showgrid=True, gridcolor="#EEE")
            fig_acc.update_xaxes(tickangle=-15)
            st.plotly_chart(fig_acc, use_container_width=True)

        with col_right:
            st.markdown("### Macro F1 Comparison")
            mf1s = comp_df["macro_f1"].tolist()
            fig_f1 = go.Figure(go.Bar(
                x=methods, y=mf1s,
                marker_color=["#2196F3", "#FF9800", "#4CAF50"],
                text=[f"{f:.3f}" for f in mf1s],
                textposition="outside",
            ))
            fig_f1.update_layout(
                yaxis=dict(range=[0, 1.1], title="Macro F1"),
                plot_bgcolor="white", height=320,
                showlegend=False,
                margin=dict(t=30, b=80, l=50, r=20)
            )
            fig_f1.update_yaxes(showgrid=True, gridcolor="#EEE")
            fig_f1.update_xaxes(tickangle=-15)
            st.plotly_chart(fig_f1, use_container_width=True)

        st.markdown("---")
        st.markdown("### Per-Class F1 by Method")

        f1_cols  = [c for c in comp_df.columns if c.startswith("f1_")]
        stages_f = [c.replace("f1_", "") for c in f1_cols]
        colours_m = ["#2196F3", "#FF9800", "#4CAF50"]

        fig_pc = go.Figure()
        for i, row in comp_df.iterrows():
            vals = [row[c] for c in f1_cols]
            fig_pc.add_trace(go.Bar(
                name=row["method"],
                x=stages_f,
                y=vals,
                marker_color=colours_m[i],
                text=[f"{v:.2f}" for v in vals],
                textposition="outside",
            ))

        fig_pc.update_layout(
            barmode="group",
            yaxis=dict(range=[0, 1.2], title="F1 Score"),
            xaxis_title="Sleep Stage",
            plot_bgcolor="white",
            height=380,
            legend=dict(orientation="h", y=1.1),
            margin=dict(t=60, b=40, l=50, r=20)
        )
        fig_pc.update_yaxes(showgrid=True, gridcolor="#EEE")
        st.plotly_chart(fig_pc, use_container_width=True)

        st.markdown("---")
        st.markdown("### Saved Comparison Images")
        cc1, cc2 = st.columns(2)
        with cc1:
            p1 = os.path.join(OUTPUT_DIR, "comparison_confusion_matrices.png")
            if os.path.exists(p1):
                st.image(p1, caption="Confusion matrices — all 3 methods",
                         use_container_width=True)
        with cc2:
            p2 = os.path.join(OUTPUT_DIR, "comparison_f1_per_class.png")
            if os.path.exists(p2):
                st.image(p2, caption="Per-class F1 bar chart",
                         use_container_width=True)

        with st.expander("ℹ️ Why the agent may not always have the highest accuracy"):
            st.markdown("""
            The agent's primary value is **grounded reasoning**, not maximum accuracy.

            When the agent overrides the base LLM scorer, it must cite a specific
            numeric reason from a tool result (e.g., "delta_power=8.73 dominates at 71%").
            It cannot override based on generic knowledge.

            This means:
            - Sometimes the base LLM was accidentally right and the agent corrects it wrongly
              — but for the *right clinical reason*
            - A grounded wrong answer is more useful clinically than an ungrounded right one,
              because a clinician can inspect and verify the reasoning
            - The comparison table shows raw accuracy, but the qualitative analysis in the
              Clinical Report tab shows reasoning quality
            """)


# ══════════════════════════════════════════════════════════════════════════
# TAB 5 — CLINICAL REPORT
# ══════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("## 🏥 Clinical Report")

    agent_df = load_agent_predictions()
    summary  = load_clinical_summary()

    if agent_df is None:
        st.info("Run the pipeline first to see the clinical report.")
    else:
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("### Sleep Architecture")

            stages = ["Wake", "N1", "N2", "N3", "REM"]
            obs_counts = [(agent_df["agent_stage"] == s).sum() for s in stages]
            norm_pcts  = [5, 5, 45, 15, 25]
            obs_pcts   = [c / max(len(agent_df), 1) * 100 for c in obs_counts]

            fig_arch = go.Figure()
            fig_arch.add_trace(go.Bar(
                name="Observed",
                x=stages, y=obs_pcts,
                marker_color=[STAGE_COLOURS[s] for s in stages],
                text=[f"{p:.1f}%" for p in obs_pcts],
                textposition="outside",
            ))
            fig_arch.add_trace(go.Scatter(
                name="Normative",
                x=stages, y=norm_pcts,
                mode="markers+lines",
                marker=dict(color="black", size=8, symbol="diamond"),
                line=dict(color="black", dash="dash"),
            ))
            fig_arch.update_layout(
                yaxis=dict(range=[0, 65], title="% of Night"),
                plot_bgcolor="white",
                height=350,
                legend=dict(orientation="h", y=1.1),
                margin=dict(t=50, b=40, l=50, r=20)
            )
            fig_arch.update_yaxes(showgrid=True, gridcolor="#EEE")
            st.plotly_chart(fig_arch, use_container_width=True)
            st.caption("Bars = observed | Dashed line = normative healthy adult")

        with col_right:
            st.markdown("### Stage Summary Table")

            norm_dict = {"Wake": 5, "N1": 5, "N2": 45, "N3": 15, "REM": 25}
            rows = []
            for s in stages:
                cnt  = int((agent_df["agent_stage"] == s).sum())
                mins = round(cnt * 0.5, 1)
                pct  = round(cnt / max(len(agent_df), 1) * 100, 1)
                norm = norm_dict[s]
                diff = round(pct - norm, 1)
                flag = "🔴 HIGH" if diff > 10 else ("🔵 LOW" if diff < -10 else "✅ NORMAL")
                rows.append({
                    "Stage": s,
                    "Epochs": cnt,
                    "Minutes": mins,
                    "Observed %": f"{pct:.1f}%",
                    "Normative %": f"{norm}%",
                    "Deviation": f"{diff:+.1f}%",
                    "Status": flag
                })

            sum_df = pd.DataFrame(rows)
            st.dataframe(sum_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### 📋 Full Clinical Summary")

        if summary:
            st.markdown(
                f'<div class="clinical-box">{summary.replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True
            )
        else:
            st.info("Clinical summary not found. Run the pipeline to generate it.")

        st.markdown("---")
        st.markdown("### Grounded vs Hallucinated Justifications")

        qa_path = os.path.join(OUTPUT_DIR, "qualitative_analysis.md")
        if os.path.exists(qa_path):
            with open(qa_path) as f:
                qa_content = f.read()
            import re
            grounded    = len(re.findall(r'GROUNDED', qa_content))
            hallucinated = len(re.findall(r'HALLUCINATED', qa_content))
            total_qa    = grounded + hallucinated

            qc1, qc2, qc3 = st.columns(3)
            with qc1:
                st.metric("Grounded", grounded,
                          help="Justification cites specific numeric feature values")
            with qc2:
                st.metric("Hallucinated", hallucinated,
                          help="Generic medical statement not tied to actual data")
            with qc3:
                if total_qa > 0:
                    st.metric("Grounding Rate",
                              f"{grounded/total_qa:.0%}")

            with st.expander("📄 Full Qualitative Analysis Report"):
                st.markdown(qa_content)

        st.markdown("---")
        with st.expander("ℹ️ What makes a justification grounded?"):
            st.markdown("""
            **Grounded example:**
            > *"get_features(42): delta_power=8.73 dominates at 71% of total spectral power.
            AASM N3 threshold requires delta dominance. Overriding N2 → N3."*

            This is grounded because it cites the exact tool result (get_features),
            the exact value (8.73), the exact percentage (71%), and the exact rule applied.
            A clinician can verify every part of this claim.

            **Hallucinated example:**
            > *"Delta waves are characteristic of slow-wave sleep, indicating N3."*

            This is hallucinated because it makes a general medical statement
            without referencing the actual epoch data. It could be true or false
            for this specific epoch — there is no way to verify it from the output alone.
            """)


# ══════════════════════════════════════════════════════════════════════════
# TAB 6 — PROMPT LAB
# ══════════════════════════════════════════════════════════════════════════

# The 5 preset prompt iterations — editable text for each version
PROMPT_PRESETS = {
    "Iteration 1 — Baseline (22% parse rate)": """\
You are a sleep scoring assistant.
Given these EEG features, classify the epoch into one of these stages:
Wake, N1, N2, N3, or REM.
Features will be provided as JSON.""",

    "Iteration 2 — JSON format added (65% parse rate)": """\
You are a sleep scoring assistant. Given EEG features, classify the epoch.
Respond ONLY with JSON in this exact format:
{"stage": "Wake or N1 or N2 or N3 or REM", "confidence": 0.0 to 1.0, "justification": "your reason"}
No extra text. No markdown.""",

    "Iteration 3 — AASM definitions added (82% parse rate)": """\
You are an expert sleep scorer. Score EEG epochs using AASM rules.

STAGE DEFINITIONS:
- Wake: high alpha and beta power, high muscle activity (emg_variance), eye movements
- N1:   theta waves dominant, slow eye movements, transitional sleep
- N2:   sleep spindles present (spindle_ratio elevated), K-complexes
- N3:   delta waves dominate more than 20 percent of the epoch
- REM:  low amplitude EEG, muscle paralysis (very low emg_variance), rapid eye movements

Respond with JSON only. No markdown.
{"stage": "...", "confidence": ..., "justification": "cite the feature values"}""",

    "Iteration 4 — Quantitative thresholds (91% parse rate)": """\
You are an expert polysomnographer scoring EEG epochs using AASM rules.

SCORING RULES:
1. If delta_power is above 4.0 and is the strongest band, classify as N3
2. If spindle_ratio is above 0.15 and delta is not dominant, classify as N2
3. If theta_power is the dominant band, classify as N1
4. If alpha_power plus beta_power is dominant and emg_variance is above 0.5, classify as Wake
5. If eog_variance is above 1.0 and emg_variance is below 0.2, classify as REM

Respond with valid JSON only. Cite the actual numbers from the features.
{"stage": "...", "confidence": ..., "justification": "cite actual values"}""",

    "Iteration 5 — Final version (≥95% parse rate)": """\
You are an expert polysomnographer scoring EEG epochs according to AASM 2023 rules.

SLEEP STAGE DEFINITIONS (AASM):
- Wake (W):  Dominant alpha rhythm (8-13 Hz), high EMG tone, frequent eye movements.
             Features: high alpha_power, high beta_power, high emg_variance.
- N1:        Transition from wake. Theta waves dominant, alpha drops.
             Features: theta_power dominant, alpha_power reduced, low spindle_ratio.
- N2:        Sleep spindles (11-16 Hz bursts) on mixed-frequency background.
             Features: spindle_ratio > 0.15, moderate delta_power, low emg_variance.
- N3:        Slow-wave sleep. Delta waves dominate the epoch.
             Features: very high delta_power (dominant over all other bands), low signal_entropy.
- REM:       Rapid eye movement sleep. Low-amplitude EEG, muscle atonia, eye movements.
             Features: high eog_variance, very low emg_variance, moderate theta_power.

SCORING RULES (apply in this priority order):
1. If delta_power is dominant AND very high (>3x other bands) → N3
2. If spindle_ratio > 0.15 AND delta is NOT dominant → N2
3. If theta_power is dominant AND delta is moderate → N1
4. If alpha_power + beta_power dominant AND emg_variance > 0.5 → Wake
5. If eog_variance > 0.8 AND emg_variance < 0.2 → REM

IMPORTANT:
- Respond ONLY with valid JSON. No markdown. No explanation outside the JSON.
- Your justification MUST cite the specific numeric values from the input.
- Format: {"stage": "...", "confidence": 0.0-1.0, "justification": "cite actual numbers"}""",
}

with tab6:
    st.markdown("## ✏️ Prompt Lab")
    st.markdown(
        "Select a prompt iteration, edit it if you want, then click "
        "**Activate & Run** to test it through the full pipeline."
    )

    st.markdown("---")

    # ── Left: preset picker + editor ──────────────────────────────────────
    col_editor, col_results = st.columns([1, 1])

    with col_editor:
        st.markdown("### 1. Choose a Preset")
        selected_preset = st.selectbox(
            "Prompt iteration:",
            list(PROMPT_PRESETS.keys()),
            index=4,
            label_visibility="collapsed"
        )

        st.markdown("### 2. Edit the Prompt (optional)")
        st.caption("You can modify the text below before running.")

        edited_prompt = st.text_area(
            "Prompt text:",
            value=PROMPT_PRESETS[selected_preset],
            height=380,
            label_visibility="collapsed",
            key="prompt_editor"
        )

        st.markdown("### 3. Quick-test on one epoch")
        st.caption("Test the prompt on a single epoch before running the full pipeline.")

        features_df = load_features()
        if features_df is not None and len(features_df) > 0:
            test_epoch_id = st.slider(
                "Pick epoch to test:",
                min_value=0,
                max_value=min(50, len(features_df) - 1),
                value=0
            )

            if st.button("🧪 Quick Test (1 epoch)", use_container_width=True):
                from agent.llm_scorer import (
                    _make_epoch_prompt, _call_ollama,
                    _parse_llm_output, _rule_based_fallback,
                    SYSTEM_PROMPT as DEFAULT_PROMPT
                )
                import agent.llm_scorer as _scorer
                _scorer.SYSTEM_PROMPT = edited_prompt

                row = features_df[features_df["epoch_id"] == test_epoch_id].iloc[0]
                prompt = _make_epoch_prompt(row)

                with st.spinner("Sending to LLM..."):
                    raw = _call_ollama(prompt, system=edited_prompt, timeout=20)

                if raw is None:
                    parsed = _rule_based_fallback(row)
                    st.warning("Ollama not running — showing rule-based fallback result.")
                else:
                    parsed = _parse_llm_output(raw)
                    if parsed is None:
                        st.error("❌ Parse FAILED — the LLM response was not valid JSON.")
                        st.code(raw, language="text")
                        parsed = None
                    else:
                        st.success("✅ Parse SUCCESS")

                if parsed:
                    true_stage = {0:"Wake",1:"N1",2:"N2",3:"N3",4:"REM"}.get(
                        int(row["true_label"]), "?"
                    )
                    correct = parsed.get("stage") == true_stage
                    icon = "✅" if correct else "❌"

                    st.markdown(f"""
                    **Epoch {test_epoch_id}** — True stage: `{true_stage}`

                    | Field | Value |
                    |---|---|
                    | Predicted stage | `{parsed.get('stage','?')}` {icon} |
                    | Confidence | `{parsed.get('confidence', '?')}` |

                    **Justification:**
                    > {parsed.get('justification', '—')}
                    """)

                # Restore default prompt
                _scorer.SYSTEM_PROMPT = DEFAULT_PROMPT

        else:
            st.info("Run the pipeline once first so feature data is available.")

        st.markdown("---")
        st.markdown("### 4. Run Full Pipeline with This Prompt")

        n_test_epochs = st.slider(
            "Epochs to score:",
            min_value=20, max_value=150, value=40, step=10,
            key="prompt_lab_epochs"
        )

        activate_btn = st.button(
            "🚀 Activate & Run Full Pipeline",
            type="primary",
            use_container_width=True,
            key="activate_prompt"
        )

    # ── Right: results panel ───────────────────────────────────────────────
    with col_results:
        st.markdown("### Live Results")

        # Show which iteration is currently active
        active = st.session_state.get("active_prompt_name", "None — using default")
        st.info(f"**Active prompt:** {active}")

        # Iteration comparison table (from session state history)
        if "iteration_history" not in st.session_state:
            st.session_state["iteration_history"] = []

        if st.session_state["iteration_history"]:
            st.markdown("#### Parse Rate History")
            hist_df = pd.DataFrame(st.session_state["iteration_history"])
            fig_hist = go.Figure(go.Bar(
                x=hist_df["iteration"],
                y=hist_df["parse_rate"],
                marker_color=[
                    "#E74C3C" if p < 0.7 else
                    "#F39C12" if p < 0.9 else
                    "#2ECC71"
                    for p in hist_df["parse_rate"]
                ],
                text=[f"{p:.0%}" for p in hist_df["parse_rate"]],
                textposition="outside",
            ))
            fig_hist.add_hline(
                y=0.95, line_dash="dash", line_color="green",
                annotation_text="95% target"
            )
            fig_hist.update_layout(
                yaxis=dict(range=[0, 1.15], title="Parse Rate"),
                xaxis_title="Iteration",
                plot_bgcolor="white",
                height=280,
                showlegend=False,
                margin=dict(t=30, b=60, l=50, r=20)
            )
            fig_hist.update_yaxes(showgrid=True, gridcolor="#EEE")
            fig_hist.update_xaxes(tickangle=-20)
            st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("#### Accuracy History")
            fig_acc2 = go.Figure(go.Scatter(
                x=hist_df["iteration"],
                y=hist_df["accuracy"],
                mode="lines+markers+text",
                marker=dict(color="#2E75B6", size=10),
                line=dict(color="#2E75B6", width=2),
                text=[f"{a:.1%}" for a in hist_df["accuracy"]],
                textposition="top center",
            ))
            fig_acc2.update_layout(
                yaxis=dict(range=[0, 1.1], title="Accuracy"),
                xaxis_title="Iteration",
                plot_bgcolor="white",
                height=240,
                margin=dict(t=30, b=60, l=50, r=20)
            )
            fig_acc2.update_yaxes(showgrid=True, gridcolor="#EEE")
            st.plotly_chart(fig_acc2, use_container_width=True)

            st.markdown("#### Full History Table")
            display_hist = hist_df.copy()
            display_hist["parse_rate"] = display_hist["parse_rate"].apply(lambda x: f"{x:.1%}")
            display_hist["accuracy"]   = display_hist["accuracy"].apply(lambda x: f"{x:.1%}")
            display_hist["macro_f1"]   = display_hist["macro_f1"].apply(lambda x: f"{x:.3f}")
            st.dataframe(display_hist, use_container_width=True, hide_index=True)
        else:
            st.markdown("""
            <div style="background:#F8F9FA;border-radius:8px;padding:30px;text-align:center;color:#888;">
            <div style="font-size:2rem">📊</div>
            <div>Run a prompt iteration to see results here.</div>
            <div style="font-size:0.8rem;margin-top:8px;">
            Each run adds a row to the history table above<br>
            so you can compare parse rates and accuracy across iterations.
            </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Sample Justifications from Last Run")

        last_justifications = st.session_state.get("last_justifications", [])
        if last_justifications:
            for item in last_justifications[:8]:
                true_s  = item["true"]
                pred_s  = item["pred"]
                just    = item["just"]
                correct = true_s == pred_s
                icon    = "✅" if correct else "❌"
                box_cls = "accepted-box" if correct else "override-box"
                st.markdown(f"""
                <div class="{box_cls}">
                  <strong>Epoch {item['epoch_id']}</strong> &nbsp;|&nbsp;
                  True: <strong>{true_s}</strong> &nbsp;|&nbsp;
                  Pred: <strong>{pred_s}</strong> &nbsp;{icon}
                  <br><small>📝 {str(just)[:180]}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("Justifications will appear here after running a prompt iteration.")

    # ── Handle Activate button ─────────────────────────────────────────────
    if activate_btn:
        st.session_state["active_prompt"]      = edited_prompt
        st.session_state["active_prompt_name"] = selected_preset

        import agent.llm_scorer as _scorer
        _scorer.SYSTEM_PROMPT = edited_prompt

        st.markdown("---")
        st.markdown(f"### Running: *{selected_preset}*")

        progress  = st.progress(0)
        status    = st.empty()

        status.markdown("**⚙️ Extracting features...**")
        progress.progress(10)
        features_df_lab = load_features()
        if features_df_lab is None:
            from env.feature_extractor import load_and_extract
            features_df_lab = load_and_extract(data_dir=DATA_DIR)

        status.markdown("**🤖 Running LLM scorer...**")
        progress.progress(40)
        from agent.llm_scorer import score_epochs
        llm_df_lab, parse_rate_lab = score_epochs(
            features_df_lab,
            max_epochs=n_test_epochs,
            use_fallback_if_no_ollama=True
        )

        status.markdown("**🔍 Running agent reviewer...**")
        progress.progress(70)
        from agent.agentic_reviewer import SleepTools, review_epochs
        tools_lab    = SleepTools(features_df_lab, llm_df_lab)
        agent_df_lab = review_epochs(features_df_lab, llm_df_lab, tools_lab,
                                     ollama_available=False)

        status.markdown("**📊 Computing metrics...**")
        progress.progress(90)
        from sklearn.metrics import accuracy_score, f1_score
        from rl.ml_baseline import FEATURE_COLS, train_and_evaluate

        y_true  = agent_df_lab["true_label"].values
        y_pred  = agent_df_lab["agent_pred"].values
        acc_lab = accuracy_score(y_true, y_pred)
        mf1_lab = f1_score(y_true, y_pred, average="macro",
                           labels=sorted(set(y_true)), zero_division=0)

        # Save to history
        short_name = selected_preset.split("—")[0].strip()
        st.session_state["iteration_history"].append({
            "iteration": short_name,
            "parse_rate": parse_rate_lab,
            "accuracy":   acc_lab,
            "macro_f1":   mf1_lab,
            "n_epochs":   n_test_epochs,
        })

        # Save sample justifications
        st.session_state["last_justifications"] = [
            {
                "epoch_id": int(r["epoch_id"]),
                "true":     {0:"Wake",1:"N1",2:"N2",3:"N3",4:"REM"}.get(int(r["true_label"]),"?"),
                "pred":     str(r["agent_stage"]),
                "just":     str(r["agent_just"]),
            }
            for _, r in agent_df_lab.head(8).iterrows()
        ]

        progress.progress(100)
        status.markdown(
            f"### ✅ Done — Parse rate: **{parse_rate_lab:.1%}** | "
            f"Accuracy: **{acc_lab:.1%}** | Macro F1: **{mf1_lab:.3f}**"
        )

        # Restore default prompt
        from agent.llm_scorer import SYSTEM_PROMPT as _DEFAULT
        _scorer.SYSTEM_PROMPT = _DEFAULT

        time.sleep(0.5)
        st.rerun()
