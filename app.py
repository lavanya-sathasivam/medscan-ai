from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import uuid

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

from database.db import fetch_scans, init_db, save_scan
from models.model_loader import load_model
from models.predict import predict_image
from utils.gradcam import GradCAMPlusPlus
from utils.processing import DatasetSplitConfig, split_dataset
from utils.preprocess import preprocess_image
from utils.triage import get_triage

ROOT_DIR = Path(__file__).resolve().parent
ICON_PATH = ROOT_DIR / "MedScan AI.png"
MODEL_PATH = ROOT_DIR / "hemorrhage_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


st.set_page_config(
    page_title="MedScan AI",
    page_icon=str(ICON_PATH) if ICON_PATH.exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_model() -> torch.nn.Module:
    return load_model(MODEL_PATH, DEVICE)


def initialize_app() -> torch.nn.Module:
    init_db()
    initialize_session_state()
    return get_model()


def initialize_session_state() -> None:
    st.session_state.setdefault("patients", {})
    st.session_state.setdefault("current_patient_id", None)
    st.session_state.setdefault("analysis_result", None)
    st.session_state.setdefault("last_uploaded_filename", None)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --page-bg: #f4f7fb;
            --panel-bg: rgba(255, 255, 255, 0.92);
            --panel-border: rgba(16, 34, 57, 0.08);
            --text-main: #112033;
            --text-muted: #607086;
            --accent: #0f5c7a;
            --accent-soft: #dff3f8;
            --critical: #a52232;
            --warning: #b26a00;
            --safe: #18794e;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(15, 92, 122, 0.12), transparent 28%),
                radial-gradient(circle at top right, rgba(20, 64, 115, 0.10), transparent 22%),
                linear-gradient(180deg, #f7fafc 0%, var(--page-bg) 100%);
            color: var(--text-main);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #102239 0%, #173552 100%);
            color: #f5f8fb;
        }

        [data-testid="stSidebar"] * {
            color: inherit;
        }

        .hero-card,
        .panel-card,
        .metric-card,
        .status-card {
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 22px;
            box-shadow: 0 18px 50px rgba(16, 34, 57, 0.08);
        }

        .hero-card {
            padding: 2rem;
            background:
                linear-gradient(135deg, rgba(16, 34, 57, 0.95), rgba(15, 92, 122, 0.90)),
                var(--panel-bg);
            color: #f7fbff;
            min-height: 220px;
        }

        .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.78rem;
            opacity: 0.74;
            margin-bottom: 0.75rem;
        }

        .hero-title {
            font-size: 2.65rem;
            font-weight: 700;
            line-height: 1.1;
            margin: 0;
        }

        .hero-copy {
            margin-top: 0.9rem;
            max-width: 44rem;
            font-size: 1.02rem;
            line-height: 1.7;
            color: rgba(247, 251, 255, 0.86);
        }

        .panel-card {
            padding: 1.35rem 1.4rem;
            margin-top: 0.9rem;
        }

        .section-title {
            margin: 0;
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--text-main);
        }

        .section-copy {
            margin-top: 0.35rem;
            color: var(--text-muted);
            font-size: 0.95rem;
            line-height: 1.55;
        }

        .metric-card {
            padding: 1.2rem 1.15rem;
            height: 100%;
        }

        .metric-label {
            color: var(--text-muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .metric-value {
            margin-top: 0.55rem;
            font-size: 1.65rem;
            font-weight: 700;
            color: var(--text-main);
        }

        .metric-subtext {
            margin-top: 0.5rem;
            color: var(--text-muted);
            font-size: 0.9rem;
        }

        .status-card {
            padding: 1.25rem 1.35rem;
            height: 100%;
        }

        .status-critical {
            border-left: 6px solid var(--critical);
        }

        .status-warning {
            border-left: 6px solid var(--warning);
        }

        .status-safe {
            border-left: 6px solid var(--safe);
        }

        .status-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            color: var(--text-muted);
            letter-spacing: 0.08em;
        }

        .status-value {
            margin-top: 0.45rem;
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text-main);
        }

        .status-caption {
            margin-top: 0.55rem;
            color: var(--text-muted);
            line-height: 1.5;
        }

        .empty-state {
            border: 1px dashed rgba(16, 34, 57, 0.18);
            border-radius: 18px;
            padding: 1.35rem;
            background: rgba(255, 255, 255, 0.65);
            color: var(--text-muted);
        }

        .sidebar-title {
            margin: 0 0 0.4rem 0;
            font-size: 1.35rem;
            font-weight: 700;
        }

        .sidebar-copy {
            color: rgba(245, 248, 251, 0.8);
            line-height: 1.6;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown('<p class="sidebar-title">MedScan AI</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="sidebar-copy">Clinical decision support for CT brain scan review.</p>',
            unsafe_allow_html=True,
        )
        st.caption(f"Inference device: `{DEVICE.type.upper()}`")

        if st.session_state.current_patient_id:
            st.success(f"Active patient: {st.session_state.current_patient_id}")
        else:
            st.info("Register a patient to start a new review.")

        if st.button("Start new patient intake", use_container_width=True):
            st.session_state.current_patient_id = None
            st.session_state.analysis_result = None
            st.rerun()

        patients = st.session_state.patients
        if patients:
            st.markdown("### Session patients")
            for patient_id, patient in patients.items():
                label = f"{patient_id} - {patient['name']}"
                is_active = st.session_state.current_patient_id == patient_id
                if st.button(
                    label,
                    key=f"patient-select-{patient_id}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state.current_patient_id = patient_id
                    st.session_state.analysis_result = None
                    st.rerun()
        else:
            st.caption("No patients registered in this session.")


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="eyebrow">Radiology workflow</div>
            <h1 class="hero-title">Hemorrhage triage with a cleaner clinical dashboard.</h1>
            <p class="hero-copy">
                Upload a CT brain scan, run AI-assisted review, inspect class probabilities and
                Grad-CAM attention, then retain each scan event in local history for the active patient.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_panel(title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="panel-card">
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


def render_patient_registration() -> None:
    render_panel(
        "Patient intake",
        "Capture a lightweight patient record before starting scan review. Session entries stay available in the sidebar for quick switching.",
    )

    with st.form("patient-registration-form", clear_on_submit=False):
        col1, col2, col3 = st.columns([1.1, 1.5, 0.8])
        patient_id = col1.text_input("Patient ID", placeholder="MS-001").strip()
        name = col2.text_input("Patient name", placeholder="Aarav Patel").strip()
        age = col3.number_input("Age", min_value=0, max_value=120, value=45)
        submitted = st.form_submit_button("Save and open dashboard", use_container_width=True)

    if submitted:
        if not patient_id or not name:
            st.error("Patient ID and patient name are required.")
            return

        st.session_state.patients[patient_id] = {
            "name": name,
            "age": int(age),
            "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        st.session_state.current_patient_id = patient_id
        st.session_state.analysis_result = None
        st.success(f"Patient {patient_id} is ready for scan review.")
        st.rerun()


def get_active_patient() -> dict[str, Any] | None:
    patient_id = st.session_state.current_patient_id
    if not patient_id:
        return None
    return st.session_state.patients.get(patient_id)


def render_patient_summary(patient_id: str, patient: dict[str, Any], history: pd.DataFrame) -> None:
    total_scans = len(history.index)
    last_scan_time = history.iloc[0]["timestamp"] if total_scans else "No scans yet"

    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card("Patient ID", patient_id, "Active review context")
    with col2:
        render_metric_card("Patient", patient["name"], f"Age {patient['age']}")
    with col3:
        render_metric_card("Scan history", str(total_scans), f"Latest: {last_scan_time}")


def decode_uploaded_image(uploaded_file: Any) -> np.ndarray:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file could not be decoded as an image.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def classify_status(triage: str) -> str:
    triage_lower = triage.lower()
    if "emergency" in triage_lower:
        return "critical"
    if "review" in triage_lower or "low confidence" in triage_lower:
        return "warning"
    return "safe"


def render_status_card(title: str, value: str, copy: str, tone: str) -> None:
    tone_class = {
        "critical": "status-critical",
        "warning": "status-warning",
        "safe": "status-safe",
    }[tone]
    st.markdown(
        f"""
        <div class="status-card {tone_class}">
            <div class="status-label">{title}</div>
            <div class="status-value">{value}</div>
            <div class="status-caption">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_probability_chart(probabilities: np.ndarray) -> alt.Chart:
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
            y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
            color=alt.Color(
                "Class:N",
                scale=alt.Scale(domain=["Hemorrhage", "Normal"], range=["#a52232", "#18794e"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Class:N"),
                alt.Tooltip("Probability:Q", format=".2%"),
            ],
        )
        .properties(height=280)
    )


def create_gradcam_overlay(model, input_tensor, image):
    # Select correct layer
    if hasattr(model, "layer4"):
        target_layer = model.layer4[-1]
    elif hasattr(model, "features"):
        target_layer = model.features[-1]
    else:
        raise ValueError("Unsupported model")

    gradcam = GradCAMPlusPlus(model, target_layer)

    try:
        heatmap = gradcam.generate(input_tensor)

        # 🔥 Keep only strong regions
        heatmap = np.clip(heatmap, 0.5, 1)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() + 1e-8)

        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

        # Ensure image format
        if image.max() <= 1.0:
            image = image * 255
        image = image.astype(np.uint8)

        # 🔥 Create RED mask manually (NO JET)
        red_mask = np.zeros_like(image)
        red_mask[:, :, 0] = (heatmap * 255).astype(np.uint8)  # Red channel only

        # 🔥 Blend (strong red highlight)
        overlay = cv2.addWeighted(image, 0.7, red_mask, 0.6, 0)

        return overlay

    finally:
        gradcam.close()

def analyze_scan(
    model: torch.nn.Module,
    patient_id: str,
    patient: dict[str, Any],
    uploaded_file: Any,
    image: np.ndarray,
) -> dict[str, Any]:
    status_placeholder = st.empty()
    progress = st.progress(0, text="Initializing review pipeline")
    stages = [
        (18, "Preparing scan and standardizing image dimensions"),
        (44, "Running hemorrhage classifier"),
        (72, "Scoring clinical triage confidence"),
        (100, "Generating Grad-CAM attention map"),
    ]
    for value, message in stages:
        progress.progress(value, text=message)

    input_tensor = preprocess_image(image).to(DEVICE)
    prediction, confidence, probabilities = predict_image(model, input_tensor)
    triage = get_triage(prediction, confidence)
    overlay = create_gradcam_overlay(model, input_tensor, image)

    record = {
        "scan_id": str(uuid.uuid4()),
        "patient_id": patient_id,
        "patient_name": patient["name"],
        "patient_age": patient["age"],
        "image_name": uploaded_file.name,
        "prediction": prediction,
        "confidence": float(confidence),
        "triage": triage,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "probabilities": probabilities,
        "overlay": overlay,
    }

    save_scan(
        patient_id=patient_id,
        patient_name=patient["name"],
        patient_age=int(patient["age"]),
        image_name=uploaded_file.name,
        prediction=prediction,
        confidence=float(confidence),
        triage=triage,
    )

    progress.empty()
    status_placeholder.success("Analysis complete. Results updated below.")
    return record


def render_results(result: dict[str, Any]) -> None:
    render_panel(
        "Analysis results",
        "Review model output, estimated confidence, triage severity, and the visual explanation generated from the final convolutional block.",
    )

    tone = classify_status(result["triage"])
    prediction_copy = (
        "Potential intracranial hemorrhage pattern detected."
        if result["prediction"] == "hemorrhage"
        else "Model review favors a normal scan."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        render_status_card("Prediction", result["prediction"].title(), prediction_copy, tone)
    with col2:
        render_status_card(
            "Confidence",
            f"{result['confidence']:.1%}",
            "Probability assigned to the predicted class.",
            tone,
        )
    with col3:
        render_status_card(
            "Triage",
            result["triage"],
            "Operational urgency label derived from prediction confidence.",
            tone,
        )

    visual_col, chart_col = st.columns([1.1, 1])
    with visual_col:
        st.image(
            result["overlay"],
            caption="Grad-CAM attention overlay",
            use_container_width=True,
        )
    with chart_col:
        st.altair_chart(build_probability_chart(result["probabilities"]), use_container_width=True)


def render_scan_history(history: pd.DataFrame) -> None:
    render_panel(
        "Scan history",
        "Stored scan events for the active patient are listed below in reverse chronological order.",
    )

    if history.empty:
        st.markdown(
            '<div class="empty-state">No stored scan events yet. Run an analysis to populate patient history.</div>',
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
    )
    if "patient_name" in display_frame.columns:
        display_frame = display_frame.drop(columns=["patient_name", "patient_id", "patient_age"], errors="ignore")

    if "Confidence" in display_frame.columns:
        display_frame["Confidence"] = display_frame["Confidence"].map(lambda value: f"{float(value):.1%}")

    st.dataframe(display_frame, use_container_width=True, hide_index=True)


def render_active_patient_dashboard(model: torch.nn.Module, patient_id: str, patient: dict[str, Any]) -> None:
    history = fetch_scans(patient_id)
    render_patient_summary(patient_id, patient, history)

    upload_col, insight_col = st.columns([1.2, 0.8])
    with upload_col:
        render_panel(
            "Scan upload",
            "Use JPG, JPEG, or PNG images. The uploaded scan is reviewed locally with the configured ResNet18 model weights.",
        )
        uploaded_file = st.file_uploader(
            "Upload CT brain scan",
            type=["jpg", "jpeg", "png"],
            help="Accepted formats: JPG, JPEG, PNG",
        )
        if uploaded_file is not None:
            try:
                image = decode_uploaded_image(uploaded_file)
            except ValueError as exc:
                st.error(str(exc))
                return

            st.image(image, caption=uploaded_file.name, use_container_width=True)
            st.session_state.last_uploaded_filename = uploaded_file.name

            if st.button("Run AI analysis", type="primary", use_container_width=True):
                st.session_state.analysis_result = analyze_scan(
                    model=model,
                    patient_id=patient_id,
                    patient=patient,
                    uploaded_file=uploaded_file,
                    image=image,
                )
                st.rerun()

    with insight_col:
        render_panel(
            "Review notes",
            "Triage labels are supportive signals only. Clinical interpretation and patient management remain the responsibility of qualified professionals.",
        )
        st.caption(
            "Model artifact expected at "
            f"`{MODEL_PATH.name}`. Current file status: {'available' if MODEL_PATH.exists() else 'missing'}."
        )
        if st.session_state.last_uploaded_filename:
            st.caption(f"Most recent uploaded file: `{st.session_state.last_uploaded_filename}`")

    if st.session_state.analysis_result:
        render_results(st.session_state.analysis_result)

    render_scan_history(fetch_scans(patient_id))


def render_missing_model_message() -> None:
    st.error(
        "Model weights were not found. Place `hemorrhage_model.pth` in the project root before running the app."
    )
    st.stop()


def main() -> None:
    initialize_session_state()
    inject_styles()
    render_sidebar()
    render_hero()

    if not MODEL_PATH.exists():
        render_missing_model_message()

    model = initialize_app()
    patient = get_active_patient()
    if patient is None:
        render_patient_registration()
        return

    render_active_patient_dashboard(model, st.session_state.current_patient_id, patient)


if __name__ == "__main__":
    main()
