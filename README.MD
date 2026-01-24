# ML Models Playground (FastAPI + Streamlit)

## Repo structure

- `streamlit/` — application code
  - `ml_core.py` — shared utils (data prep, split, metrics, safe custom metric eval)
  - `01_data_analysis.py` — EDA + schema generation (`reports/schema.json`)
  - `02_train_models.py` — train + save models to `models/` + holdout dump
  - `api.py` — FastAPI inference service (predict, metrics, feature importance)
  - `app.py` — Streamlit UI (model selection, holdout eval, custom rows/file upload)
- `scraped_data/` — raw scraped datasets (place your `videos.jsonl`, optional `channels.jsonl`)
- `reports/` — analysis outputs (`analysis.log`, `schema.json`, etc.)
- `models/` — trained artifacts (`*.joblib`, `*.cbm`, `*.pt`, `holdout.parquet`, `split.json`)
- `eda/` - data analysis code v2_cosmetics.py is the final version
- `catboost_info/` — optional outputs/logs
- `scraper/` — scraping code

## Setup

```bash
python -m venv .venv
```
Windows PowerShell
```bash
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install pandas numpy scikit-learn scipy joblib torch streamlit fastapi uvicorn catboost pyarrow requests
```

Run (from repo root)
1) Data analysis (writes reports/schema.json)

python streamlit/01_data_analysis.py --channels scraped_data\v1_vids_and_channels_100k\channels.ndjson --videos scraped_data\v1_vids_and_channels_100k\videos.ndjson

1) Train + save models (writes to models/)

python streamlit/02_train_models.py --channels scraped_data\v1_vids_and_channels_100k\channels.ndjson --videos scraped_data\v1_vids_and_channels_100k\videos.ndjson

3) Start API (FastAPI)

uvicorn api:app --app-dir streamlit --host 0.0.0.0 --port 8000


4) Start UI (Streamlit)

In a second terminal:

streamlit run streamlit/app.py

Notes

    Default target is hits (change via --target in analysis/train scripts if needed).

    Holdout split is time-based (by publication_ts else created_ts), and is saved as models/holdout.parquet + models/split.json.

