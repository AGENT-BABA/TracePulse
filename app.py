"""TracePulse: Hybrid ML & LLM Log Anomaly Root-Cause Analyzer.

Interactive Streamlit web dashboard providing:
- Log ingestion & synthetic log generation
- Real-time feature engineering & Isolation Forest anomaly detection
- Interactive telemetry timeline visualizations
- Automated Incident Grouping & Pydantic-structured LLM Root-Cause Analysis (RCA)
- Filterable telemetry log explorer
"""

from __future__ import annotations

import io
import time
from datetime import datetime
from pathlib import Path
from typing import List

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Import TracePulse pipeline components
from src.detector import LogAnomalyDetector
from src.log_generator import SyntheticLogGenerator
from src.parser import LogParser
from src.rca_engine import IncidentGrouper, IncidentReport, RCAEngine

load_dotenv()

# Streamlit Page Config
st.set_page_config(
    page_title="TracePulse — Hybrid ML & LLM Log Anomaly RCA",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished modern design
st.markdown(
    """
    <style>
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #1a1e29;
        border: 1px solid #2e384d;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    div[data-testid="stMetricLabel"] p {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetricValue"] div {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
    }

    /* Incident Card Badges */
    .severity-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-right: 8px;
    }
    .severity-critical { background-color: #ef4444; color: #ffffff; }
    .severity-high { background-color: #f97316; color: #ffffff; }
    .severity-medium { background-color: #eab308; color: #1e293b; }
    .severity-low { background-color: #3b82f6; color: #ffffff; }

    .incident-card {
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 16px;
        border: 1px solid #334155;
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
    }

    .badge-pill {
        display: inline-block;
        background-color: #334155;
        color: #e2e8f0;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        margin-right: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_default_log_file() -> Path:
    """Return path to data/sample_logs.log, ensuring directory exists."""
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "sample_logs.log"


def load_or_generate_logs(force_generate: bool = False) -> str:
    """Retrieve raw log content from disk or generate a fresh set."""
    log_file = get_default_log_file()
    if force_generate or not log_file.exists():
        generator = SyntheticLogGenerator(seed=int(time.time()) if force_generate else 42)
        generator.save_to_file(log_file, normal_count=1500)

    return log_file.read_text(encoding="utf-8")


# ---------------- Sidebar Controls ----------------
st.sidebar.image(
    "https://img.icons8.com/fluency/96/shield.png",
    width=64,
)
st.sidebar.title("TracePulse")
st.sidebar.caption("Two-Stage Unsupervised ML & LLM RCA Pipeline")
st.sidebar.divider()

# 1. Data Source Selection
st.sidebar.subheader("1. Ingestion Source")
data_source = st.sidebar.radio(
    "Select Log Source:",
    ["Use Generated Demo Logs", "Upload .log / .txt File"],
    index=0,
)

raw_log_content = ""
if data_source == "Use Generated Demo Logs":
    col_reg1, col_reg2 = st.sidebar.columns([1, 1])
    if col_reg1.button("🔄 Regenerate", use_container_width=True, help="Generate fresh baseline logs and randomized anomaly clusters"):
        with st.spinner("Generating fresh logs..."):
            raw_log_content = load_or_generate_logs(force_generate=True)
            st.sidebar.success("New logs generated!")
    else:
        raw_log_content = load_or_generate_logs(force_generate=False)
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload raw log file:",
        type=["log", "txt"],
        help="Upload standard structured application logs",
    )
    if uploaded_file is not None:
        raw_log_content = uploaded_file.getvalue().decode("utf-8", errors="replace")
    else:
        st.sidebar.info("Upload a log file or switch to demo logs.")

# 2. Isolation Forest Hyperparameters
st.sidebar.subheader("2. ML Anomaly Detector")
contamination = st.sidebar.slider(
    "Contamination Threshold",
    min_value=0.01,
    max_value=0.10,
    value=0.03,
    step=0.005,
    help="Expected proportion of anomalies in dataset (Isolation Forest contamination parameter)",
)
n_estimators = st.sidebar.slider(
    "Estimator Trees",
    min_value=50,
    max_value=250,
    value=150,
    step=25,
    help="Number of isolation trees in ensemble",
)

# 3. LLM RCA Configuration
st.sidebar.subheader("3. RCA Engine Configuration")
rca_engine = RCAEngine()
has_api_key = rca_engine.is_api_configured()

use_llm_mode = st.sidebar.toggle(
    "Enable Live LLM RCA",
    value=has_api_key,
    help="Send grouped incident telemetry to OpenAI/OpenRouter LLM endpoint for structured Pydantic analysis",
)

if use_llm_mode:
    if not has_api_key:
        st.sidebar.warning("⚠️ No valid OPENAI_API_KEY found in environment. Heuristic fallback will be used.")
    else:
        st.sidebar.success(f"🟢 Active LLM: `{rca_engine.model_name}`")
else:
    st.sidebar.info("⚡ Fast Deterministic Heuristic Engine Active")

st.sidebar.divider()
st.sidebar.markdown(
    """
    <div style='font-size: 0.78rem; color: #94a3b8;'>
    <b>Architecture:</b><br/>
    Stage 1: Regex Parser & Rolling Features<br/>
    Stage 2: Isolation Forest Anomaly Detection<br/>
    Stage 3: 60s Sliding Window Incident Clustering<br/>
    Stage 4: Pydantic Structured LLM Root-Cause Analysis
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- Main Dashboard View ----------------

st.title("TracePulse: Hybrid ML & LLM Log Anomaly Root-Cause Analyzer")
st.markdown(
    "Ingests application telemetry, identifies statistical outliers via unsupervised **Isolation Forest**, "
    "clusters alerts into temporal incident windows, and generates actionable **Root-Cause Diagnostics**."
)

if not raw_log_content or not raw_log_content.strip():
    st.warning("Please upload a log file or select 'Use Generated Demo Logs' from the sidebar to start analysis.")
    st.stop()

# Parse and process logs
log_lines = raw_log_content.splitlines()

with st.spinner("Parsing log lines and engineering statistical features..."):
    parser = LogParser()
    try:
        df_parsed = parser.parse_lines(log_lines)
        if df_parsed.empty:
            st.info("No valid structured log records found in the provided input.")
            st.stop()
        df_featured, feature_matrix = parser.engineer_features(df_parsed)
    except Exception as e:
        st.error(f"Failed to parse logs: {e}")
        st.stop()

# Run Isolation Forest anomaly detection
with st.spinner("Training Isolation Forest and scoring anomalies..."):
    detector = LogAnomalyDetector(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=42,
    )
    df_results, metrics = detector.fit_predict(df_featured, feature_matrix)

# Group anomalies into incidents
anomalies_df = df_results[df_results["is_anomaly"]].copy()
incidents = IncidentGrouper.group_anomalies(anomalies_df, window_seconds=60)

# Top KPI Metric Row
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Logs Ingested", f"{metrics['total_logs']:,}")
kpi2.metric(
    "Anomalies Flagged",
    f"{metrics['anomalies_detected']:,}",
    f"{metrics['anomaly_rate_pct']}% rate",
    delta_color="inverse",
)
kpi3.metric("Grouped Incidents", len(incidents))
kpi4.metric("ML Inference Time", f"{metrics['inference_time_ms']} ms")

st.markdown("<br/>", unsafe_allow_html=True)

# ---------------- Interactive Telemetry Visualization ----------------
st.subheader("📈 Telemetry Timeline & Anomaly Detection")

viz_df = df_results.copy()
viz_df["time_display"] = viz_df["timestamp"].dt.strftime("%H:%M:%S")
viz_df["anomaly_status"] = np.where(viz_df["is_anomaly"], "Anomaly (Isolation Forest)", "Normal Baseline")

base_chart = (
    alt.Chart(viz_df)
    .mark_circle()
    .encode(
        x=alt.X("timestamp:T", title="Timestamp (Time)"),
        y=alt.Y("latency_ms:Q", title="Latency (ms)", scale=alt.Scale(type="log", padding=10)),
        color=alt.Color(
            "anomaly_status:N",
            scale=alt.Scale(
                domain=["Normal Baseline", "Anomaly (Isolation Forest)"],
                range=["#64748b", "#ef4444"],
            ),
            legend=alt.Legend(title="Detection Status", orient="top-left"),
        ),
        size=alt.condition(
            alt.datum.is_anomaly,
            alt.value(75),
            alt.value(25),
        ),
        opacity=alt.condition(
            alt.datum.is_anomaly,
            alt.value(0.95),
            alt.value(0.4),
        ),
        tooltip=[
            alt.Tooltip("timestamp_str:N", title="Timestamp"),
            alt.Tooltip("service:N", title="Microservice"),
            alt.Tooltip("endpoint:N", title="Endpoint"),
            alt.Tooltip("latency_ms:Q", title="Latency (ms)"),
            alt.Tooltip("status_code:Q", title="HTTP Status"),
            alt.Tooltip("anomaly_severity:Q", title="Anomaly Severity (0-1)"),
            alt.Tooltip("message:N", title="Log Message"),
        ],
    )
    .properties(height=360)
    .interactive()
)

st.altair_chart(base_chart, use_container_width=True)

# ---------------- Incident Root-Cause Analysis (RCA) Cards ----------------
st.subheader(f"🔍 Root-Cause Analysis Incidents ({len(incidents)} Discovered)")

if not incidents:
    st.info("No anomaly clusters detected with current contamination threshold. Try adjusting the slider in the sidebar.")
else:
    force_heuristic = not use_llm_mode
    rca_reports: List[IncidentReport] = []

    with st.spinner("Synthesizing Root-Cause Diagnostics..."):
        for inc in incidents:
            report = rca_engine.diagnose_incident(inc, force_heuristic=force_heuristic)
            rca_reports.append(report)

    for inc, report in zip(incidents, rca_reports):
        sev_class = f"severity-{report.severity.lower()}"
        with st.expander(
            f"🚨 [{report.severity}] {inc.incident_id} — {report.incident_title} ({inc.service})",
            expanded=True,
        ):
            c_meta1, c_meta2, c_meta3, c_meta4 = st.columns(4)
            c_meta1.markdown(f"**Target Service:** `{inc.service}`")
            c_meta2.markdown(f"**Anomalous Events:** `{inc.log_count}` logs")
            c_meta3.markdown(f"**Duration:** `{inc.duration_seconds():.1f}s`")
            c_meta4.markdown(f"**Dominant Status:** `{inc.dominant_status}` (Avg {inc.avg_latency}ms)")

            st.markdown(
                f"""
                <div style="margin-top: 8px; margin-bottom: 8px;">
                    <span class="severity-badge {sev_class}">{report.severity}</span>
                    <span class="badge-pill">Confidence: {int(report.confidence_score * 100)}%</span>
                    <span class="badge-pill">Cluster ID: {inc.incident_id}</span>
                    <span class="badge-pill">Peak Latency: {inc.max_latency}ms</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("**🔬 Root-Cause Diagnostic:**")
            st.info(report.root_cause)

            st.markdown("**📦 Impacted Components & Dependencies:**")
            badges_html = " ".join([f'<span class="badge-pill">{c}</span>' for c in report.affected_components])
            st.markdown(badges_html, unsafe_allow_html=True)

            st.markdown("**🛠️ Actionable Remediation Steps:**")
            for idx, step in enumerate(report.remediation_steps, start=1):
                st.markdown(f"**{idx}.** {step}")

            with st.expander("📄 View Triggering Log Snippets"):
                snippet_df = inc.logs[
                    ["timestamp_str", "level", "service", "endpoint", "latency_ms", "status_code", "message"]
                ]
                st.dataframe(snippet_df, use_container_width=True, hide_index=True)

# ---------------- Raw Data Explorer Table ----------------
st.subheader("📋 Filterable Telemetry Explorer")

f_col1, f_col2, f_col3 = st.columns([1, 1, 2])
filter_status = f_col1.selectbox("Filter by Status", ["All Logs", "Anomalies Only", "Normal Only"])
available_services = ["All Services"] + sorted(df_results["service"].unique().tolist())
filter_service = f_col2.selectbox("Filter by Service", available_services)
search_query = f_col3.text_input("Search Log Messages", "")

filtered_df = df_results.copy()
if filter_status == "Anomalies Only":
    filtered_df = filtered_df[filtered_df["is_anomaly"]]
elif filter_status == "Normal Only":
    filtered_df = filtered_df[~filtered_df["is_anomaly"]]

if filter_service != "All Services":
    filtered_df = filtered_df[filtered_df["service"] == filter_service]

if search_query.strip():
    filtered_df = filtered_df[filtered_df["message"].str.contains(search_query, case=False, na=False)]

display_cols = [
    "timestamp_str",
    "level",
    "service",
    "endpoint",
    "latency_ms",
    "status_code",
    "is_anomaly",
    "anomaly_severity",
    "message",
]

st.dataframe(
    filtered_df[display_cols],
    use_container_width=True,
    hide_index=True,
)

# CSV Export Button
csv_buffer = io.StringIO()
filtered_df[display_cols].to_csv(csv_buffer, index=False)
st.download_button(
    label="📥 Download Filtered Telemetry (CSV)",
    data=csv_buffer.getvalue(),
    file_name=f"tracepulse_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv",
)
