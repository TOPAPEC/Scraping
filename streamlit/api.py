import json
import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from catboost import CatBoostRegressor

from ml_core import is_standardized_numeric, mse, mae, r2, safe_numpy_metric


class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]
    scale_numeric: bool = True
    y_true: Optional[List[float]] = None
    custom_metric: Optional[str] = None


class PredictResponse(BaseModel):
    n: int
    y_pred: List[float]
    metrics: Optional[Dict[str, float]] = None
    custom_metric: Optional[Dict[str, Any]] = None


class EmbMLP(nn.Module):
    def __init__(self, num_in: int, cat_card_dims: List[List[int]]):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(int(card), int(dim)) for card, dim in cat_card_dims])
        emb_out = int(sum(int(dim) for _, dim in cat_card_dims))
        self.fc1 = nn.Linear(int(num_in + emb_out), 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, x_num: torch.Tensor, x_cat: List[torch.Tensor]) -> torch.Tensor:
        embs = []
        for emb, xc in zip(self.embs, x_cat):
            embs.append(emb(xc))
        x = torch.cat([x_num] + embs, dim=1) if embs else x_num
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)


def clean_bool(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = False
        s = out[c]
        if s.dtype == bool:
            out[c] = s
        else:
            v = s.astype(str).str.strip().str.lower()
            out[c] = v.isin(["1", "true", "t", "yes", "y", "да"])
    return out


def clean_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out.get(c, np.nan), errors="coerce").fillna(0.0).astype(float)
    return out


def clean_categorical(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = out.get(c, "").astype(str).fillna("")
    return out


def linear_predict(df: pd.DataFrame, scale_numeric: bool, meta: dict, scaler, ohe, model) -> np.ndarray:
    num_cols = meta["numeric"]
    bool_cols = meta["bool"]
    cat_cols = meta["categorical"]
    d = clean_categorical(clean_numeric(clean_bool(df, bool_cols), num_cols), cat_cols)
    X_num = d[num_cols].to_numpy(dtype=np.float32) if len(num_cols) else np.zeros((len(d), 0), dtype=np.float32)
    if meta["expects_scaled_numeric"] and (not scale_numeric) and X_num.shape[1]:
        if not is_standardized_numeric(X_num):
            raise HTTPException(status_code=400, detail="numeric features look non-standardized, but scale_numeric=false")
    X_num_s = scaler.transform(X_num).astype(np.float32) if (scale_numeric and X_num.shape[1]) else X_num
    X_bool = d[bool_cols].astype(int).to_numpy(dtype=np.float32) if len(bool_cols) else np.zeros((len(d), 0), dtype=np.float32)
    X_cat = ohe.transform(d[cat_cols]) if len(cat_cols) else sp.csr_matrix((len(d), 0))
    X = sp.hstack([sp.csr_matrix(X_num_s), sp.csr_matrix(X_bool), X_cat], format="csr")
    return model.predict(X).astype(float)


def catboost_predict(df: pd.DataFrame, meta: dict, model: CatBoostRegressor) -> np.ndarray:
    num_cols = meta["numeric"]
    bool_cols = meta["bool"]
    cat_cols = meta["categorical"]
    d = clean_categorical(clean_numeric(clean_bool(df, bool_cols), num_cols), cat_cols)
    X = pd.concat([d[num_cols], d[bool_cols].astype(int), d[cat_cols].astype(str)], axis=1)
    return model.predict(X).astype(float)


def nn_predict(df: pd.DataFrame, scale_numeric: bool, meta: dict, scaler, state_path: str) -> np.ndarray:
    num_cols = meta["numeric"]
    bool_cols = meta["bool"]
    cat_cols = meta["categorical"]
    d = clean_categorical(clean_numeric(clean_bool(df, bool_cols), num_cols), cat_cols)
    X_num = d[num_cols].to_numpy(dtype=np.float32) if len(num_cols) else np.zeros((len(d), 0), dtype=np.float32)
    if meta["expects_scaled_numeric"] and (not scale_numeric) and X_num.shape[1]:
        if not is_standardized_numeric(X_num):
            raise HTTPException(status_code=400, detail="numeric features look non-standardized, but scale_numeric=false")
    X_num_s = scaler.transform(X_num).astype(np.float32) if (scale_numeric and X_num.shape[1]) else X_num
    X_bool = d[bool_cols].astype(int).to_numpy(dtype=np.float32) if len(bool_cols) else np.zeros((len(d), 0), dtype=np.float32)
    X_num_all = np.concatenate([X_num_s, X_bool], axis=1).astype(np.float32)

    vocabs = meta["vocabs"]
    x_cat = []
    for c in cat_cols:
        vocab = vocabs[c]
        arr = d[c].astype(str).fillna("").map(lambda v: vocab.get(v, 0)).astype(np.int64).to_numpy()
        x_cat.append(arr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmbMLP(num_in=X_num_all.shape[1], cat_card_dims=meta["cat_card_dims"]).to(device)
    sd = torch.load(state_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    with torch.no_grad():
        xb_num = torch.from_numpy(X_num_all).to(device)
        xb_cat = [torch.from_numpy(a).to(device) for a in x_cat]
        pred = model(xb_num, xb_cat).detach().cpu().numpy().reshape(-1)
    return pred.astype(float)


def permutation_importance(df: pd.DataFrame, y_true: np.ndarray, predict_fn, features: List[str], metric_name: str, n_rows: int, seed: int) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    if len(df) > n_rows:
        idx = rng.choice(len(df), size=n_rows, replace=False)
        d = df.iloc[idx].copy()
        y = y_true[idx].copy()
    else:
        d = df.copy()
        y = y_true.copy()

    base_pred = predict_fn(d)
    base = mse(y, base_pred) if metric_name == "mse" else (mae(y, base_pred) if metric_name == "mae" else -r2(y, base_pred))

    out = []
    for f in features:
        dd = d.copy()
        if f not in dd.columns:
            continue
        col = dd[f].to_numpy()
        perm = col.copy()
        rng.shuffle(perm)
        dd[f] = perm
        p = predict_fn(dd)
        val = mse(y, p) if metric_name == "mse" else (mae(y, p) if metric_name == "mae" else -r2(y, p))
        out.append({"feature": f, "importance": float(val - base)})
    out.sort(key=lambda x: x["importance"], reverse=True)
    return out


MODELS_DIR = os.getenv("MODELS_DIR", "models")

with open(f"{MODELS_DIR}/schema_used.json", "r", encoding="utf-8") as f:
    SCHEMA = json.load(f)

with open(f"{MODELS_DIR}/linear_meta.json", "r", encoding="utf-8") as f:
    LINEAR_META = json.load(f)
LINEAR_SCALER = joblib.load(f"{MODELS_DIR}/linear_scaler.joblib")
LINEAR_OHE = joblib.load(f"{MODELS_DIR}/linear_ohe.joblib")
LINEAR_MODEL = joblib.load(f"{MODELS_DIR}/linear_model.joblib")

with open(f"{MODELS_DIR}/catboost_meta.json", "r", encoding="utf-8") as f:
    CB_META = json.load(f)
CB_MODEL = CatBoostRegressor()
CB_MODEL.load_model(f"{MODELS_DIR}/catboost.cbm")

with open(f"{MODELS_DIR}/nn_meta.json", "r", encoding="utf-8") as f:
    NN_META = json.load(f)
NN_SCALER = joblib.load(f"{MODELS_DIR}/nn_scaler.joblib")
NN_STATE = f"{MODELS_DIR}/nn_state.pt"

HOLDOUT = pd.read_parquet(f"{MODELS_DIR}/holdout.parquet")
TARGET = SCHEMA["target"]

app = FastAPI()


@app.get("/health")
def health():
    return {"ok": True, "models": ["linear", "catboost", "nn"]}


@app.get("/models")
def list_models():
    return {"models": ["linear", "catboost", "nn"], "target": TARGET}


@app.get("/holdout_metrics")
def holdout_metrics():
    df = HOLDOUT.copy()
    y = pd.to_numeric(df.get(TARGET, 0.0), errors="coerce").fillna(0.0).astype(float).values
    res = {}
    lp = linear_predict(df, True, LINEAR_META, LINEAR_SCALER, LINEAR_OHE, LINEAR_MODEL)
    cp = catboost_predict(df, CB_META, CB_MODEL)
    npred = nn_predict(df, True, NN_META, NN_SCALER, NN_STATE)
    res["linear"] = {"mse": mse(y, lp), "mae": mae(y, lp), "r2": r2(y, lp)}
    res["catboost"] = {"mse": mse(y, cp), "mae": mae(y, cp), "r2": r2(y, cp)}
    res["nn"] = {"mse": mse(y, npred), "mae": mae(y, npred), "r2": r2(y, npred)}
    return res


@app.post("/predict/{model_name}", response_model=PredictResponse)
def predict(model_name: str, req: PredictRequest):
    if model_name not in {"linear", "catboost", "nn"}:
        raise HTTPException(status_code=404, detail="unknown model")
    df = pd.DataFrame(req.records)
    if model_name == "linear":
        pred = linear_predict(df, req.scale_numeric, LINEAR_META, LINEAR_SCALER, LINEAR_OHE, LINEAR_MODEL)
    elif model_name == "catboost":
        pred = catboost_predict(df, CB_META, CB_MODEL)
    else:
        pred = nn_predict(df, req.scale_numeric, NN_META, NN_SCALER, NN_STATE)

    out_metrics = None
    out_custom = None
    if req.y_true is not None:
        y = np.asarray(req.y_true, dtype=float).reshape(-1)
        p = np.asarray(pred, dtype=float).reshape(-1)
        if len(y) != len(p):
            raise HTTPException(status_code=400, detail="y_true length mismatch")
        out_metrics = {"mse": mse(y, p), "mae": mae(y, p), "r2": r2(y, p)}
        if req.custom_metric:
            v, err = safe_numpy_metric(req.custom_metric, y, p)
            out_custom = {"expr": req.custom_metric, "value": v, "error": err}
    return PredictResponse(n=int(len(pred)), y_pred=[float(x) for x in pred], metrics=out_metrics, custom_metric=out_custom)


@app.get("/importance/{model_name}")
def importance(model_name: str, metric: str = "mse", n_rows: int = 2000, seed: int = 42):
    if model_name not in {"linear", "catboost", "nn"}:
        raise HTTPException(status_code=404, detail="unknown model")
    if metric not in {"mse", "mae", "r2"}:
        raise HTTPException(status_code=400, detail="metric must be mse|mae|r2")

    df = HOLDOUT.copy()
    y = pd.to_numeric(df.get(TARGET, 0.0), errors="coerce").fillna(0.0).astype(float).values

    if model_name == "linear":
        feats = LINEAR_META["numeric"] + LINEAR_META["bool"] + LINEAR_META["categorical"]
        def pf(dfx): return linear_predict(dfx, True, LINEAR_META, LINEAR_SCALER, LINEAR_OHE, LINEAR_MODEL)
        perm = permutation_importance(df, y, pf, feats, metric, n_rows, seed)
        coefs = np.asarray(LINEAR_MODEL.coef_).reshape(-1)
        names = LINEAR_META["feature_names"]
        w = [{"feature": names[i], "weight": float(coefs[i]), "abs_weight": float(abs(coefs[i]))} for i in range(min(len(names), len(coefs)))]
        w.sort(key=lambda x: x["abs_weight"], reverse=True)
        return {"linear_weights": w[:200], "permutation_importance": perm[:200]}
    if model_name == "catboost":
        feats = CB_META["numeric"] + CB_META["bool"] + CB_META["categorical"]
        def pf(dfx): return catboost_predict(dfx, CB_META, CB_MODEL)
        perm = permutation_importance(df, y, pf, feats, metric, n_rows, seed)
        X = pd.concat(
            [clean_numeric(clean_bool(df, CB_META["bool"]), CB_META["numeric"])[CB_META["numeric"]],
             clean_bool(df, CB_META["bool"])[CB_META["bool"]].astype(int),
             clean_categorical(df, CB_META["categorical"])[CB_META["categorical"]].astype(str)],
            axis=1
        )
        builtin = CB_MODEL.get_feature_importance(type="PredictionValuesChange")
        builtin = [{"feature": X.columns[i], "importance": float(builtin[i])} for i in range(len(X.columns))]
        builtin.sort(key=lambda x: x["importance"], reverse=True)
        return {"catboost_builtin_importance": builtin[:200], "permutation_importance": perm[:200]}
    feats = NN_META["numeric"] + NN_META["bool"] + NN_META["categorical"]
    def pf(dfx): return nn_predict(dfx, True, NN_META, NN_SCALER, NN_STATE)
    perm = permutation_importance(df, y, pf, feats, metric, n_rows, seed)
    return {"permutation_importance": perm[:200]}
