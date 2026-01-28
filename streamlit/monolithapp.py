import json
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from catboost import CatBoostRegressor

from ml_core import is_standardized_numeric, mse, mae, r2, safe_numpy_metric

MODELS_DIR = os.getenv("MODELS_DIR", "models")


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
        embs = [emb(xc) for emb, xc in zip(self.embs, x_cat)]
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
            raise RuntimeError("numeric features look non-standardized, but scale_numeric=false")
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


def nn_predict(df: pd.DataFrame, scale_numeric: bool, meta: dict, scaler, model: EmbMLP, device: torch.device) -> np.ndarray:
    num_cols = meta["numeric"]
    bool_cols = meta["bool"]
    cat_cols = meta["categorical"]
    d = clean_categorical(clean_numeric(clean_bool(df, bool_cols), num_cols), cat_cols)
    X_num = d[num_cols].to_numpy(dtype=np.float32) if len(num_cols) else np.zeros((len(d), 0), dtype=np.float32)
    if meta["expects_scaled_numeric"] and (not scale_numeric) and X_num.shape[1]:
        if not is_standardized_numeric(X_num):
            raise RuntimeError("numeric features look non-standardized, but scale_numeric=false")
    X_num_s = scaler.transform(X_num).astype(np.float32) if (scale_numeric and X_num.shape[1]) else X_num
    X_bool = d[bool_cols].astype(int).to_numpy(dtype=np.float32) if len(bool_cols) else np.zeros((len(d), 0), dtype=np.float32)
    X_num_all = np.concatenate([X_num_s, X_bool], axis=1).astype(np.float32)

    vocabs = meta["vocabs"]
    x_cat = []
    for c in cat_cols:
        vocab = vocabs[c]
        arr = d[c].astype(str).fillna("").map(lambda v: vocab.get(v, 0)).astype(np.int64).to_numpy()
        x_cat.append(arr)

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
        if f not in d.columns:
            continue
        dd = d.copy()
        col = dd[f].to_numpy()
        perm = col.copy()
        rng.shuffle(perm)
        dd[f] = perm
        p = predict_fn(dd)
        val = mse(y, p) if metric_name == "mse" else (mae(y, p) if metric_name == "mae" else -r2(y, p))
        out.append({"feature": f, "importance": float(val - base)})
    out.sort(key=lambda x: x["importance"], reverse=True)
    return out


@st.cache_data
def load_schema():
    p = f"{MODELS_DIR}/schema_used.json"
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_holdout():
    p = f"{MODELS_DIR}/holdout.parquet"
    if os.path.exists(p):
        return pd.read_parquet(p)
    return None


@st.cache_resource
def load_models():
    with open(f"{MODELS_DIR}/linear_meta.json", "r", encoding="utf-8") as f:
        linear_meta = json.load(f)
    linear_scaler = joblib.load(f"{MODELS_DIR}/linear_scaler.joblib")
    linear_ohe = joblib.load(f"{MODELS_DIR}/linear_ohe.joblib")
    linear_model = joblib.load(f"{MODELS_DIR}/linear_model.joblib")

    with open(f"{MODELS_DIR}/catboost_meta.json", "r", encoding="utf-8") as f:
        cb_meta = json.load(f)
    cb_model = CatBoostRegressor()
    cb_model.load_model(f"{MODELS_DIR}/catboost.cbm")

    with open(f"{MODELS_DIR}/nn_meta.json", "r", encoding="utf-8") as f:
        nn_meta = json.load(f)
    nn_scaler = joblib.load(f"{MODELS_DIR}/nn_scaler.joblib")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_in = int(len(nn_meta["numeric"]) + len(nn_meta["bool"]))
    nn_model = EmbMLP(num_in=num_in, cat_card_dims=nn_meta["cat_card_dims"]).to(device)
    sd = torch.load(f"{MODELS_DIR}/nn_state.pt", map_location=device)
    nn_model.load_state_dict(sd)

    return {
        "linear": {"meta": linear_meta, "scaler": linear_scaler, "ohe": linear_ohe, "model": linear_model},
        "catboost": {"meta": cb_meta, "model": cb_model},
        "nn": {"meta": nn_meta, "scaler": nn_scaler, "model": nn_model, "device": device},
    }


def _norm_cell(x):
    if isinstance(x, (list, dict, set, tuple)):
        try:
            return json.dumps(x, ensure_ascii=False, sort_keys=isinstance(x, dict))
        except Exception:
            return str(x)
    return x


def build_editor_df(n_rows: int, features: list, holdout: pd.DataFrame | None) -> pd.DataFrame:
    base = pd.DataFrame({c: [None] * n_rows for c in features})
    if holdout is None or len(holdout) == 0 or n_rows <= 0:
        return base
    pool = holdout.reindex(columns=list(set(holdout.columns) | set(features))).copy()
    acc = []
    need = n_rows
    for seed in [42, 123, 777, 999, 2026, 2027, 2028]:
        take = min(max(n_rows * 10, 200), len(pool))
        samp = pool.sample(n=take, random_state=seed) if take < len(pool) else pool
        samp = samp[features].copy()
        tmp = samp.applymap(_norm_cell)
        keep_idx = tmp.drop_duplicates().index
        samp = samp.loc[keep_idx]
        acc.append(samp)
        merged = pd.concat(acc, axis=0, ignore_index=True)
        tmpm = merged.applymap(_norm_cell)
        merged = merged.loc[tmpm.drop_duplicates().index].reset_index(drop=True)
        if len(merged) >= need:
            merged = merged.head(need)
            return merged.reindex(columns=features)
    merged = pd.concat(acc, axis=0, ignore_index=True) if acc else base
    if len(merged) == 0:
        return base
    tmpm = merged.applymap(_norm_cell)
    merged = merged.loc[tmpm.drop_duplicates().index].reset_index(drop=True)
    merged = merged.head(min(len(merged), n_rows)).reindex(columns=features)
    if len(merged) < n_rows:
        pad = pd.DataFrame({c: [None] * (n_rows - len(merged)) for c in features})
        merged = pd.concat([merged, pad], axis=0, ignore_index=True)
    return merged.reindex(columns=features)


def predict_local(model_name: str, records: List[Dict[str, Any]], scale_numeric: bool, models: dict) -> np.ndarray:
    df = pd.DataFrame(records)
    if model_name == "linear":
        m = models["linear"]
        return linear_predict(df, scale_numeric, m["meta"], m["scaler"], m["ohe"], m["model"])
    if model_name == "catboost":
        m = models["catboost"]
        return catboost_predict(df, m["meta"], m["model"])
    m = models["nn"]
    return nn_predict(df, scale_numeric, m["meta"], m["scaler"], m["model"], m["device"])


def main():
    st.set_page_config(page_title="ML Models Playground", layout="wide")
    st.title("Интерактивное ML-приложение: выбор модели, проверка данных, метрики, важности")

    schema = load_schema()
    holdout = load_holdout()
    if schema is None or holdout is None:
        st.error("Нет models/schema_used.json или models/holdout.parquet. Сначала запусти 01_data_analysis.py и 02_train_models.py.")
        return

    models = load_models()
    target = schema["target"]
    model_list = ["linear", "catboost", "nn"]

    cols = st.columns([2, 2, 2, 3])
    with cols[0]:
        model_name = st.selectbox("Модель", model_list, index=1)
    with cols[1]:
        scale_numeric = st.toggle("Скейлить числовые признаки (если выключить — для стандартизованных моделей будет ошибка на сыром вводе)", value=True)
    with cols[2]:
        metric_for_importance = st.selectbox("Метрика для permutation importance", ["mse", "mae", "r2"], index=0)
    with cols[3]:
        n_rows_pi = st.slider("Строк для permutation importance (ускоряет вычисления)", min_value=200, max_value=5000, value=2000, step=200)

    fs = schema["feature_sets"][model_name]
    features = fs["numeric"] + fs["bool"] + fs["categorical"]

    st.subheader("Holdout: качество и быстрый sanity-check")
    max_h = int(min(50000, len(holdout)))
    default_h = int(min(5000, max_h))
    holdout_n = st.slider("Holdout rows for custom metric", min_value=200, max_value=max_h, value=default_h, step=200)

    custom_on_holdout = st.toggle("Also compute custom metric on holdout", value=False)
    custom_expr_holdout = st.text_input("Custom metric for holdout (vars: np, y_true, y_pred)", value="np.mean(np.abs(y_true - y_pred))")

    if st.button("Compute holdout metrics"):
        try:
            dfh = holdout.copy()
            y = pd.to_numeric(dfh.get(target, 0.0), errors="coerce").fillna(0.0).astype(float).values
            res = {}
            for mn in model_list:
                fs_m = schema["feature_sets"][mn]
                feats_m = fs_m["numeric"] + fs_m["bool"] + fs_m["categorical"]
                rec = dfh.reindex(columns=list(set(dfh.columns) | set(feats_m))).copy()
                records = rec[feats_m].to_dict(orient="records")
                pred = predict_local(mn, records, True, models)
                res[mn] = {"mse": mse(y, pred), "mae": mae(y, pred), "r2": r2(y, pred)}
            if not custom_on_holdout:
                st.json(res.get(model_name, res))
            else:
                dfhs = dfh.sample(n=int(holdout_n), random_state=42) if int(holdout_n) < len(dfh) else dfh
                y_true = pd.to_numeric(dfhs.get(target, 0.0), errors="coerce").fillna(0.0).astype(float).values
                recs = dfhs.reindex(columns=list(set(dfhs.columns) | set(features))).copy()[features].to_dict(orient="records")
                y_pred = predict_local(model_name, recs, bool(scale_numeric), models)
                st.write("Base metrics")
                st.json({"mse": mse(y_true, y_pred), "mae": mae(y_true, y_pred), "r2": r2(y_true, y_pred)})
                v, err = safe_numpy_metric(custom_expr_holdout, y_true, y_pred)
                st.write("Custom metric")
                st.json({"expr": custom_expr_holdout, "value": v, "error": err})
        except Exception as e:
            st.error(str(e))

    st.subheader("Новые данные: загрузка файла")
    up = st.file_uploader("CSV / Parquet / JSONL (json-lines)", type=["csv", "parquet", "jsonl"])
    df_new = None
    if up is not None:
        name = up.name.lower()
        try:
            if name.endswith(".csv"):
                df_new = pd.read_csv(up)
            elif name.endswith(".parquet"):
                df_new = pd.read_parquet(up)
            elif name.endswith(".jsonl"):
                lines = up.getvalue().decode("utf-8").splitlines()
                recs = [json.loads(x) for x in lines if x.strip()]
                df_new = pd.json_normalize(recs, sep=".")
        except Exception as e:
            st.error(str(e))

    if df_new is not None:
        st.write("Превью загруженных данных")
        st.dataframe(df_new.head(50), use_container_width=True)

        has_y = target in df_new.columns
        y_true = pd.to_numeric(df_new[target], errors="coerce").fillna(0.0).astype(float).values if has_y else None

        rec = df_new.reindex(columns=list(set(df_new.columns) | set(features))).copy()
        records = rec[features].to_dict(orient="records")

        custom_expr = st.text_input("Кастомная метрика (numpy-выражение). Доступно: np, y_true, y_pred", value="np.mean(np.abs(y_true - y_pred))")
        if st.button("Predict + метрики на загруженных данных"):
            try:
                y_pred = predict_local(model_name, records, bool(scale_numeric), models)
                st.write("Предсказания (первые 50)")
                st.dataframe(pd.DataFrame({"y_pred": y_pred}).head(50), use_container_width=True)
                if y_true is not None:
                    st.write("Базовые метрики")
                    st.json({"mse": mse(y_true, y_pred), "mae": mae(y_true, y_pred), "r2": r2(y_true, y_pred)})
                    v, err = safe_numpy_metric(custom_expr, y_true, y_pred)
                    st.write("Кастомная метрика")
                    st.json({"expr": custom_expr, "value": v, "error": err})
            except Exception as e:
                st.error(str(e))

    st.subheader("Новые данные: ручной ввод строк")
    n_new = st.number_input("Сколько строк добавить", min_value=1, max_value=200, value=5, step=1)
    cat_options = schema.get("cat_options", {})
    col_cfg = {}
    nonneg = set(schema.get("nonneg_numeric", []))

    for c in fs["numeric"]:
        col_cfg[c] = st.column_config.NumberColumn(c, min_value=0.0) if c in nonneg else st.column_config.NumberColumn(c)
    for c in fs["bool"]:
        col_cfg[c] = st.column_config.CheckboxColumn(c)
    for c in fs["categorical"]:
        opts = cat_options.get(c, [""])
        col_cfg[c] = st.column_config.SelectboxColumn(c, options=opts, required=False)

    df_editor = build_editor_df(int(n_new), features, holdout)
    df_filled = st.data_editor(df_editor, column_config=col_cfg, num_rows="dynamic", use_container_width=True)

    y_true_input = None
    if st.toggle("У меня есть y_true для этих строк (для оценки метрик)", value=False):
        y_true_input = st.text_area("Вставь y_true построчно или через запятую", value="")
    custom_expr2 = st.text_input("Кастомная метрика (numpy-выражение) для ручного ввода. Доступно: np, y_true, y_pred", value="np.mean((y_true - y_pred)**2)")

    if st.button("Predict на введённых строках"):
        try:
            df_use = df_filled.copy()
            records = df_use[features].to_dict(orient="records")
            y_pred = predict_local(model_name, records, bool(scale_numeric), models)
            st.dataframe(pd.DataFrame({"y_pred": y_pred}), use_container_width=True)
            if y_true_input is not None and y_true_input.strip():
                parts = [p.strip() for p in y_true_input.replace("\n", ",").split(",") if p.strip()]
                y_true = np.asarray([float(x) for x in parts], dtype=float).reshape(-1)
                if len(y_true) == len(y_pred):
                    st.json({"mse": mse(y_true, y_pred), "mae": mae(y_true, y_pred), "r2": r2(y_true, y_pred)})
                    v, err = safe_numpy_metric(custom_expr2, y_true, y_pred)
                    st.json({"expr": custom_expr2, "value": v, "error": err})
                else:
                    st.error("y_true length mismatch")
        except Exception as e:
            st.error(str(e))

    st.subheader("Feature importance")
    if st.button("Показать важности (встроенная + permutation где доступно)"):
        try:
            dfh = holdout.copy()
            y = pd.to_numeric(dfh.get(target, 0.0), errors="coerce").fillna(0.0).astype(float).values

            if model_name == "linear":
                m = models["linear"]
                feats = m["meta"]["numeric"] + m["meta"]["bool"] + m["meta"]["categorical"]
                def pf(dfx):
                    recs = dfx.reindex(columns=list(set(dfx.columns) | set(feats))).copy()[feats].to_dict(orient="records")
                    return predict_local("linear", recs, True, models)
                perm = permutation_importance(dfh, y, pf, feats, metric_for_importance, int(n_rows_pi), 42)
                coefs = np.asarray(m["model"].coef_).reshape(-1)
                names = m["meta"]["feature_names"]
                w = [{"feature": names[i], "weight": float(coefs[i]), "abs_weight": float(abs(coefs[i]))} for i in range(min(len(names), len(coefs)))]
                w.sort(key=lambda x: x["abs_weight"], reverse=True)
                st.json({"linear_weights": w[:200], "permutation_importance": perm[:200]})
            elif model_name == "catboost":
                m = models["catboost"]
                feats = m["meta"]["numeric"] + m["meta"]["bool"] + m["meta"]["categorical"]
                def pf(dfx):
                    recs = dfx.reindex(columns=list(set(dfx.columns) | set(feats))).copy()[feats].to_dict(orient="records")
                    return predict_local("catboost", recs, True, models)
                perm = permutation_importance(dfh, y, pf, feats, metric_for_importance, int(n_rows_pi), 42)
                X = pd.concat(
                    [
                        clean_numeric(clean_bool(dfh, m["meta"]["bool"]), m["meta"]["numeric"])[m["meta"]["numeric"]],
                        clean_bool(dfh, m["meta"]["bool"])[m["meta"]["bool"]].astype(int),
                        clean_categorical(dfh, m["meta"]["categorical"])[m["meta"]["categorical"]].astype(str),
                    ],
                    axis=1,
                )
                builtin = m["model"].get_feature_importance(type="PredictionValuesChange")
                builtin = [{"feature": X.columns[i], "importance": float(builtin[i])} for i in range(len(X.columns))]
                builtin.sort(key=lambda x: x["importance"], reverse=True)
                st.json({"catboost_builtin_importance": builtin[:200], "permutation_importance": perm[:200]})
            else:
                m = models["nn"]
                feats = m["meta"]["numeric"] + m["meta"]["bool"] + m["meta"]["categorical"]
                def pf(dfx):
                    recs = dfx.reindex(columns=list(set(dfx.columns) | set(feats))).copy()[feats].to_dict(orient="records")
                    return predict_local("nn", recs, True, models)
                perm = permutation_importance(dfh, y, pf, feats, metric_for_importance, int(n_rows_pi), 42)
                st.json({"permutation_importance": perm[:200]})
        except Exception as e:
            st.error(str(e))


main()
