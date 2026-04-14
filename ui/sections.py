from __future__ import annotations

from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from ui.state import select_patient


def render_sidebar(
    *,
    device_label: str,
    patients: dict[str, dict[str, Any]],
    current_patient_id: str | None,
) -> None:
    with st.sidebar:
        st.markdown('<p class="sidebar-heading">MedScan AI</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="sidebar-copy">CT brain scan triage workspace for clinician-led review.</p>',
            unsafe_allow_html=True,
        )
        st.caption(f"Inference device: `{device_label}`")

        if st.button("New patient intake", use_container_width=True, type="primary"):
            select_patient(None)
            st.rerun()

        if current_patient_id:
            st.success(f"Active workspace: {current_patient_id}")
        else:
            st.info("No active patient workspace.")

        if patients:
            st.markdown("### Patient workspaces")
            for patient_id, patient in patients.items():
                label = f"{patient_id} - {patient['name']}"
                is_active = current_patient_id == patient_id
                if st.button(
                    label,
                    key=f"patient-select-{patient_id}",
                    use_container_width=True,
                    type="secondary" if not is_active else "primary",
                ):
                    select_patient(patient_id)
                    st.rerun()
        else:
            st.caption("Patient workspaces created in this session will appear here.")


def render_page_header(*, model_path: Path, model_available: bool) -> None:
    model_status = "Model ready" if model_available else "Model missing"
    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-header-kicker">Clinical review workspace</div>
            <h1 class="page-header-title">CT hemorrhage triage dashboard</h1>
            <p class="page-header-copy">
                Register the active patient, upload a brain CT scan, run AI-assisted review,
                and inspect triage, confidence, Grad-CAM attention, and stored scan history
                in a single workflow designed for rapid interpretation.
            </p>
            <div class="page-header-meta">
                <span class="meta-pill">{model_status}</span>
                <span class="meta-pill">Canonical model path: {model_path.as_posix()}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_shell(title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="shell-card">
            <p class="section-title">{title}</p>
            <p class="section-copy">{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, subtext: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-subtext">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_readiness_banner(readiness: dict[str, str]) -> None:
    st.markdown(
        f"""
        <div class="banner-card banner-{readiness['tone']}">
            <h3>{readiness['title']}</h3>
            <p>{readiness['message']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_patient_registration_form() -> tuple[bool, str, str, int]:
    render_section_shell(
        "Patient intake",
        "Create a lightweight patient workspace for the current review session. This stores session context and links future scan events in the local history table.",
    )

    with st.form("patient-registration-form", clear_on_submit=False):
        col1, col2, col3 = st.columns([1.05, 1.45, 0.75])
        patient_id = col1.text_input("Patient ID", placeholder="MS-001").strip()
        name = col2.text_input("Patient name", placeholder="Aarav Patel").strip()
        age = col3.number_input("Age", min_value=0, max_value=120, value=45)
        submitted = st.form_submit_button("Open patient workspace", use_container_width=True)

    st.markdown(
        '<div class="intake-note">Clinical note: patient records remain session-scoped in the UI. Persisted storage currently covers scan events only.</div>',
        unsafe_allow_html=True,
    )
    return submitted, patient_id, name, int(age)


def render_patient_summary(patient_id: str, patient: dict[str, Any], history: pd.DataFrame) -> None:
    total_scans = len(history.index)
    last_scan_time = history.iloc[0]["timestamp"] if total_scans else "No scans recorded"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Active patient", patient_id, "Current review workspace")
    with col2:
        render_metric_card("Patient name", patient["name"], f"Age {patient['age']}")
    with col3:
        render_metric_card("Scan events", str(total_scans), "Reverse chronological local history")
    with col4:
        render_metric_card("Latest event", str(last_scan_time), "Most recent stored scan timestamp")


def render_scan_upload_panel(*, model_available: bool) -> tuple[Any | None, bool]:
    render_section_shell(
        "Scan upload",
        "Upload a JPG, JPEG, or PNG CT brain image for local AI-assisted review. The uploaded scan is not persisted until analysis is completed.",
    )
    uploaded_file = st.file_uploader(
        "Upload CT brain scan",
        type=["jpg", "jpeg", "png"],
        help="Accepted formats: JPG, JPEG, PNG",
        disabled=not model_available,
    )
    analyze_clicked = st.button(
        "Run AI analysis",
        type="primary",
        use_container_width=True,
        disabled=(uploaded_file is None or not model_available),
    )
    return uploaded_file, analyze_clicked


def render_review_notes(*, model_path: Path, last_uploaded_filename: str | None) -> None:
    render_section_shell(
        "Clinical guidance",
        "Triage output is an assistive signal only. Final interpretation, escalation, and treatment decisions remain with qualified clinical staff.",
    )
    st.caption(f"Configured model path: `{model_path.as_posix()}`")
    if last_uploaded_filename:
        st.caption(f"Most recent uploaded scan: `{last_uploaded_filename}`")


def render_triage_card(title: str, value: str, copy: str, tone: str) -> None:
    st.markdown(
        f"""
        <div class="triage-card triage-{tone}">
            <div class="triage-label">{title}</div>
            <div class="triage-value">{value}</div>
            <div class="triage-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def classify_tone(triage: str) -> str:
    triage_lower = triage.lower()
    if "emergency" in triage_lower:
        return "critical"
    if "review" in triage_lower or "low confidence" in triage_lower:
        return "warning"
    return "safe"


def build_probability_chart(probabilities) -> alt.Chart:
    probability_frame = pd.DataFrame(
        {
            "Class": ["Hemorrhage", "Normal"],
            "Probability": [float(probabilities[0]), float(probabilities[1])],
        }
    )
    return (
        alt.Chart(probability_frame)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
        .encode(
            x=alt.X("Class:N", axis=alt.Axis(title=None, labelAngle=0)),
            y=alt.Y("Probability:Q", axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "Class:N",
                scale=alt.Scale(domain=["Hemorrhage", "Normal"], range=["#a7343f", "#206f52"]),
                legend=None,
            ),
            tooltip=[alt.Tooltip("Class:N"), alt.Tooltip("Probability:Q", format=".2%")],
        )
        .properties(height=280)
    )


def render_results(result: dict[str, Any]) -> None:
    render_section_shell(
        "Analysis results",
        "Decision-first review order: operational triage, predicted class, model confidence, then explainability and class probabilities.",
    )
    tone = classify_tone(result["triage"])
    prediction_copy = (
        "Pattern is concerning for intracranial hemorrhage and should be prioritized for clinician attention."
        if result["prediction"] == "hemorrhage"
        else "Model output favors a normal study, but clinician review remains required."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        render_triage_card(
            "Triage decision",
            result["triage"],
            "Operational urgency label derived from predicted class and confidence.",
            tone,
        )
    with col2:
        render_triage_card("Predicted class", result["prediction"].title(), prediction_copy, tone)
    with col3:
        render_triage_card(
            "Confidence",
            f"{result['confidence']:.1%}",
            "Probability assigned to the predicted class.",
            tone,
        )

    visual_col, chart_col = st.columns([1.15, 0.95])
    with visual_col:
        render_section_shell(
            "Explainability overlay",
            "Grad-CAM highlights regions that influenced the model output. Use this as contextual support, not as a standalone diagnosis.",
        )
        st.image(result["overlay"], caption="Grad-CAM attention overlay", use_container_width=True)
    with chart_col:
        render_section_shell(
            "Class probabilities",
            "Relative model confidence across the hemorrhage and normal classes.",
        )
        st.altair_chart(build_probability_chart(result["probabilities"]), use_container_width=True)


def render_scan_history(history: pd.DataFrame) -> None:
    render_section_shell(
        "Scan event history",
        "Stored scan events for the active patient appear below in reverse chronological order.",
    )
    if history.empty:
        st.markdown(
            '<div class="empty-state">No stored scan events yet. Complete an analysis to save the first event for this patient.</div>',
            unsafe_allow_html=True,
        )
        return

    display_frame = history.rename(
        columns={
            "image_name": "Scan file",
            "prediction": "Prediction",
            "confidence": "Confidence",
            "triage": "Triage",
            "timestamp": "Timestamp",
        }
    ).drop(columns=["patient_name", "patient_id", "patient_age"], errors="ignore")

    if "Confidence" in display_frame.columns:
        display_frame["Confidence"] = display_frame["Confidence"].map(lambda value: f"{float(value):.1%}")

    def triage_style(value: Any) -> str:
        text = str(value).lower()
        if "emergency" in text:
            return "background-color: #fae9eb; color: #7f2230; font-weight: 600;"
        if "review" in text or "low confidence" in text:
            return "background-color: #fff2dc; color: #7b5311; font-weight: 600;"
        return "background-color: #e6f5ed; color: #195a43; font-weight: 600;"

    styled_frame = display_frame.style.map(triage_style, subset=["Triage"]) if "Triage" in display_frame.columns else display_frame
    st.dataframe(styled_frame, use_container_width=True, hide_index=True)
