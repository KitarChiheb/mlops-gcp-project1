# MLOps GCP Project 1

End-to-end MLOps pipeline on Google Cloud Platform.
Data ingestion → Processing → Training → MLFlow tracking → Flask app → Jenkins CI/CD → Cloud Run.

## Tech Stack
- Python 3.10+
- GCP (Cloud Storage, Cloud Run)
- MLFlow
- Flask
- Jenkins
- Docker

## Setup

1. Clone the repo
2. Copy `.env.example` to `.env` and fill in your values
3. Create virtual environment: `python -m venv venv`
4. Activate: `source venv/bin/activate`
5. Install: `pip install -e . && pip install -r requirements.txt`

## Project Structure

mlops-gcp-project1/
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml          ← GitHub Actions (if needed), Jenkins reads this later
│
├── src/
│   ├── __init__.py
│   ├── data_ingestion/
│   │   ├── __init__.py
│   │   └── ingest.py          ← downloads data from GCS
│   ├── data_processing/
│   │   ├── __init__.py
│   │   └── process.py         ← cleans, transforms, splits data
│   ├── model_training/
│   │   ├── __init__.py
│   │   └── train.py           ← trains model, logs to MLFlow
│   └── pipeline/
│       ├── __init__.py
│       └── training_pipeline.py ← orchestrates all steps end-to-end
│
├── app/
│   ├── __init__.py
│   ├── app.py                 ← Flask web application
│   └── templates/
│       └── index.html         ← HTML frontend
│
├── tests/
│   ├── __init__.py
│   ├── test_ingestion.py
│   ├── test_processing.py
│   └── test_training.py
│
├── notebooks/
│   └── exploration.ipynb      ← Jupyter EDA (never imported by src/)
│
├── config/
│   └── config.yaml            ← all configurable values (paths, params)
│
├── mlruns/                    ← auto-created by MLFlow (add to .gitignore)
│
├── Dockerfile                 ← containerizes the Flask app
├── requirements.txt           ← all Python dependencies
├── setup.py                   ← makes src/ importable as a package
├── .gitignore
├── .env.example               ← shows what env vars are needed (no real values)
└── README.md