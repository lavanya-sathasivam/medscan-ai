from __future__ import annotations


def get_triage(prediction: str, confidence: float) -> str:
    if prediction == "hemorrhage":
        if confidence >= 0.8:
            return "Emergency"
        if confidence >= 0.6:
            return "Needs Review"
        return "Low Confidence"
    return "Normal"
