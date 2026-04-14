from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import uuid

import cv2
import numpy as np
import streamlit as st
import torch

from database.db import fetch_scans, init_db, save_scan
from models.model_loader import load_model
from models.predict import predict_image
from ui.sections import (
    render_page_header,
    render_patient_registration_form,
    render_patient_summary,
    render_readiness_banner,
    render_results,
    render_review_notes,
    render_scan_history,
    render_scan_upload_panel,
    render_sidebar,
)
from ui.state import derive_readiness_state, get_active_patient, initialize_session_state
from ui.theme import inject_styles
from utils.gradcam import generate_gradcam_overlay
from utils.preprocess import preprocess_image
from utils.triage import get_triage

ROOT_DIR = Path(__file__).resolve().parent
ICON_PATH = ROOT_DIR / "MedScan AI.png"
MODEL_PATH = ROOT_DIR / "models" / "best_model.pth"
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
    return get_model()


def decode_uploaded_image(uploaded_file: Any) -> np.ndarray:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file could not be decoded as an image.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


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
        (20, "Preparing scan and standardizing image dimensions"),
        (46, "Running hemorrhage classifier"),
        (72, "Scoring triage urgency"),
        (100, "Generating Grad-CAM attention map"),
    ]
    for value, message in stages:
        progress.progress(value, text=message)

    input_tensor = preprocess_image(image).to(DEVICE)
    prediction, confidence, probabilities = predict_image(model, input_tensor)
    triage = get_triage(prediction, confidence)
    overlay = generate_gradcam_overlay(model, input_tensor, image)

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
    status_placeholder.success("Analysis complete. Results have been refreshed below.")
    return record


def render_patient_intake() -> None:
    submitted, patient_id, name, age = render_patient_registration_form()
    if not submitted:
        return

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
    st.session_state.last_uploaded_filename = None
    st.success(f"Patient workspace {patient_id} is ready for scan review.")
    st.rerun()


def render_active_patient_dashboard(model: torch.nn.Module, patient_id: str, patient: dict[str, Any]) -> None:
    history = fetch_scans(patient_id)
    render_patient_summary(patient_id, patient, history)

    upload_col, notes_col = st.columns([1.15, 0.85])
    with upload_col:
        uploaded_file, analyze_clicked = render_scan_upload_panel(model_available=True)
        uploaded_scan_name = uploaded_file.name if uploaded_file is not None else None

        readiness = derive_readiness_state(
            model_available=True,
            active_patient_id=patient_id,
            uploaded_scan_name=uploaded_scan_name,
            analysis_result=st.session_state.analysis_result,
        )
        render_readiness_banner(readiness)

        if uploaded_file is not None:
            try:
                image = decode_uploaded_image(uploaded_file)
            except ValueError as exc:
                st.error(str(exc))
                return

            st.image(image, caption=uploaded_file.name, use_container_width=True)
            st.session_state.last_uploaded_filename = uploaded_file.name

            if analyze_clicked:
                st.session_state.analysis_result = analyze_scan(
                    model=model,
                    patient_id=patient_id,
                    patient=patient,
                    uploaded_file=uploaded_file,
                    image=image,
                )
                st.rerun()

    with notes_col:
        render_review_notes(
            model_path=MODEL_PATH,
            last_uploaded_filename=st.session_state.last_uploaded_filename,
        )

    if st.session_state.analysis_result:
        render_results(st.session_state.analysis_result)

    render_scan_history(fetch_scans(patient_id))


def render_blocking_missing_model_state() -> None:
    readiness = derive_readiness_state(
        model_available=False,
        active_patient_id=st.session_state.current_patient_id,
        uploaded_scan_name=None,
        analysis_result=st.session_state.analysis_result,
    )
    render_readiness_banner(readiness)
    st.error(
        "Model weights are missing. Place the trained file at "
        f"`{MODEL_PATH.as_posix()}` before running AI-assisted review."
    )
    st.stop()


def main() -> None:
    initialize_session_state()
    inject_styles()

    model_available = MODEL_PATH.exists()
    render_sidebar(
        device_label=DEVICE.type.upper(),
        patients=st.session_state.patients,
        current_patient_id=st.session_state.current_patient_id,
    )
    render_page_header(model_path=MODEL_PATH, model_available=model_available)

    if not model_available:
        render_blocking_missing_model_state()

    model = initialize_app()
    patient = get_active_patient()

    if patient is None:
        readiness = derive_readiness_state(
            model_available=True,
            active_patient_id=None,
            uploaded_scan_name=None,
            analysis_result=st.session_state.analysis_result,
        )
        render_readiness_banner(readiness)
        render_patient_intake()
        return

    render_active_patient_dashboard(model, st.session_state.current_patient_id, patient)


if __name__ == "__main__":
    main()
