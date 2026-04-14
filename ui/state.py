from __future__ import annotations

from typing import Any

SESSION_DEFAULTS = {
    "patients": {},
    "current_patient_id": None,
    "analysis_result": None,
    "last_uploaded_filename": None,
}


def initialize_session_state() -> None:
    import streamlit as st

    for key, value in SESSION_DEFAULTS.items():
        st.session_state.setdefault(key, value)


def reset_active_review() -> None:
    import streamlit as st

    st.session_state.analysis_result = None
    st.session_state.last_uploaded_filename = None


def select_patient(patient_id: str | None) -> None:
    import streamlit as st

    st.session_state.current_patient_id = patient_id
    reset_active_review()


def get_active_patient() -> dict[str, Any] | None:
    import streamlit as st

    patient_id = st.session_state.current_patient_id
    if not patient_id:
        return None
    return st.session_state.patients.get(patient_id)


def derive_readiness_state(
    *,
    model_available: bool,
    active_patient_id: str | None,
    uploaded_scan_name: str | None,
    analysis_result: dict[str, Any] | None,
) -> dict[str, str]:
    if not model_available:
        return {
            "state": "model_missing",
            "tone": "critical",
            "title": "Model unavailable",
            "message": "Place the trained model in the configured path before running clinical review.",
        }

    if not active_patient_id:
        return {
            "state": "no_patient",
            "tone": "warning",
            "title": "No active patient",
            "message": "Complete patient intake to open a scan review workspace.",
        }

    analyzed_scan_name = None
    if analysis_result:
        analyzed_scan_name = str(analysis_result.get("image_name") or "")

    if uploaded_scan_name and analyzed_scan_name and uploaded_scan_name != analyzed_scan_name:
        return {
            "state": "scan_ready",
            "tone": "warning",
            "title": "New scan ready for analysis",
            "message": "A different scan is loaded than the currently displayed result. Run AI analysis to refresh the review.",
        }

    if analysis_result:
        return {
            "state": "analysis_complete",
            "tone": "safe",
            "title": "Analysis complete",
            "message": "Results, Grad-CAM explanation, and scan history are ready for review.",
        }

    if uploaded_scan_name:
        return {
            "state": "scan_ready",
            "tone": "warning",
            "title": "Scan ready for analysis",
            "message": "A scan is loaded. Run AI analysis to generate triage and explainability outputs.",
        }

    return {
        "state": "patient_ready",
        "tone": "safe",
        "title": "Patient workspace ready",
        "message": "Upload a CT brain scan to begin AI-assisted review.",
    }
