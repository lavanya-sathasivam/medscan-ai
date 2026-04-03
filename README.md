# MedScan-AI

MedScan-AI is a Streamlit-based clinical decision support prototype for CT brain scan triage. The app loads a locally supplied ResNet18 hemorrhage classifier, analyzes an uploaded scan, presents the predicted class with confidence and Grad-CAM attention, and stores scan events in a local SQLite history.

## Features

- Professional dashboard-style Streamlit interface for patient intake and scan review
- AI-assisted hemorrhage vs normal classification
- Confidence scoring with triage labels
- Grad-CAM attention overlay for visual explanation
- Local SQLite history per patient
- Clean project structure and runtime-focused dependencies

## Project Structure

```text
MedScan-AI/
|-- app.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- hemorrhage_model.pth        # local runtime asset, not intended for git tracking
|-- MedScan AI.png
|-- database/
|   |-- db.py
|   `-- database.db            # generated locally when the app runs
|-- models/
|   |-- __init__.py
|   |-- model_loader.py
|   `-- predict.py
`-- utils/
    |-- __init__.py
    |-- gradcam.py
    |-- preprocess.py
    |-- processing.py
    `-- triage.py
```

## Requirements

- Python 3.10 or newer
- A local trained weights file named `hemorrhage_model.pth` placed in the project root

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure the model file is available at:

```text
./hemorrhage_model.pth
```

## Run the App

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Workflow

1. Register a patient from the intake form.
2. Upload a CT brain scan image in JPG, JPEG, or PNG format.
3. Run AI analysis to generate:
   - predicted class
   - confidence score
   - triage label
   - Grad-CAM attention overlay
4. Review stored scan history for the active patient.

## Notes

- The model output is a supportive signal only and should not replace clinical judgment.
- Local artifacts such as virtual environments, databases, caches, and model weights are intentionally excluded from git tracking through `.gitignore`.
- `utils/processing.py` is now a reusable dataset-splitting helper rather than a hard-coded machine-specific script.

## Suggested Next Steps

- Add automated tests for preprocessing, database persistence, and prediction helpers
- Introduce configurable model and database paths via environment variables
- Add patient search and persisted patient records if the app needs multi-session usage
