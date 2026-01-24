import os
import json
import math
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def setup_logging(log_path: str) -> logging.Logger:
    ensure_dir(os.path.dirname(log_path) or ".")
    logger = logging.getLogger("ml_app")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            records.append(json.loads(s))
    return records


def read_any_table(path: str) -> pd.DataFrame:
    p = path.lower()
    if p.endswith(".ndjson") or p.endswith(".jsonl"):
        return pd.json_normalize(read_jsonl(path), sep=".")
    if p.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return pd.json_normalize(obj, sep=".")
        return pd.json_normalize([obj], sep=".")
    if p.endswith(".parquet"):
        return pd.read_parquet(path)
    if p.endswith(".csv"):
        return pd.read_csv(path)
    if p.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported file: {path}")


def normalize_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(int).astype(bool)
    v = s.astype(str).str.strip().str.lower()
    true_set = {"1", "true", "t", "yes", "y", "да"}
    false_set = {"0", "false", "f", "no", "n", "нет", "nan", "none", ""}
    out = v.map(lambda x: True if x in true_set else (False if x in false_set else False))
    return out.astype(bool)


def prepare_dataframe(videos_path: str, channels_path: Optional[str] = None) -> pd.DataFrame:
    dfv = read_any_table(videos_path)
    if channels_path and os.path.exists(channels_path):
        dfc = read_any_table(channels_path)
        if "channel_id" in dfc.columns:
            dfc = dfc.rename(columns={"channel_id": "author.id"})
        df = dfv.merge(dfc, on="author.id", how="left", suffixes=("", "_ch"))
    else:
        df = dfv
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "common_subscription_product_codes" in d.columns:
        def _len(x: Any) -> int:
            if isinstance(x, list):
                return int(len(x))
            return 0
        d["common_subscription_product_codes_len"] = d["common_subscription_product_codes"].map(_len).astype(int)
    if "created_ts" in d.columns:
        dt = pd.to_datetime(d["created_ts"], errors="coerce", utc=False)
        d["created_year"] = dt.dt.year.fillna(0).astype(int)
        d["created_month"] = dt.dt.month.fillna(0).astype(int)
        d["created_dow"] = dt.dt.dayofweek.fillna(0).astype(int)
        d["created_hour"] = dt.dt.hour.fillna(0).astype(int)
    if "publication_ts" in d.columns:
        dt = pd.to_datetime(d["publication_ts"], errors="coerce", utc=False)
        d["publication_year"] = dt.dt.year.fillna(0).astype(int)
        d["publication_month"] = dt.dt.month.fillna(0).astype(int)
        d["publication_dow"] = dt.dt.dayofweek.fillna(0).astype(int)
        d["publication_hour"] = dt.dt.hour.fillna(0).astype(int)
    if "title" in d.columns:
        d["title_len"] = d["title"].fillna("").astype(str).str.len().astype(int)
    if "description" in d.columns:
        d["description_len"] = d["description"].fillna("").astype(str).str.len().astype(int)
    return d


def infer_feature_types(df: pd.DataFrame, target: str) -> Dict[str, List[str]]:
    cols = [c for c in df.columns if c != target]
    bool_cols = []
    num_cols = []
    cat_cols = []

    def _norm(x: Any) -> Any:
        if x is None:
            return ""
        if isinstance(x, float) and np.isnan(x):
            return ""
        if isinstance(x, np.ndarray):
            x = x.tolist()
        if isinstance(x, set):
            try:
                x = sorted(list(x))
            except Exception:
                x = list(x)
        if isinstance(x, (list, tuple, dict)):
            try:
                return json.dumps(x, ensure_ascii=False, sort_keys=isinstance(x, dict))
            except Exception:
                return str(x)
        return x

    for c in cols:
        s = df[c]

        if s.dtype == bool:
            bool_cols.append(c)
            continue

        if pd.api.types.is_numeric_dtype(s):
            num_cols.append(c)
            continue

        s_nonnull = s.dropna()
        if s_nonnull.empty:
            cat_cols.append(c)
            continue

        if len(s_nonnull) > 50000:
            sample = s_nonnull.sample(n=50000, random_state=42)
        else:
            sample = s_nonnull

        try:
            nunique = sample.nunique(dropna=True)
        except TypeError:
            nunique = sample.map(_norm).nunique(dropna=True)

        if nunique <= 2:
            bool_cols.append(c)
        else:
            cat_cols.append(c)

    return {"bool": sorted(bool_cols), "numeric": sorted(num_cols), "categorical": sorted(cat_cols)}


def pick_default_feature_sets(df: pd.DataFrame, target: str) -> Dict[str, Dict[str, List[str]]]:
    base_bool = [
        "is_audio", "is_adult", "is_livestream", "is_on_air", "is_official", "is_licensed", "is_paid",
        "is_hidden", "is_original_content", "is_original_sticker_2x2", "is_reborn_channel",
        "kind_sign_for_user", "is_classic", "is_club", "author.is_allowed_offline"
    ]
    base_num = [
        "duration", "pg_rating.age", "action_reason.id", "common_subscription_product_codes_len",
        "title_len", "description_len", "created_month", "created_dow", "created_hour",
        "publication_month", "publication_dow", "publication_hour"
    ]
    base_cat = [
        "origin_type", "category.name", "category.id", "author.id", "feed_name", "stream_type", "action_reason.name"
    ]
    present = set(df.columns)
    base_bool = [c for c in base_bool if c in present]
    base_num = [c for c in base_num if c in present]
    base_cat = [c for c in base_cat if c in present]
    linear = {
        "bool": base_bool,
        "numeric": [c for c in base_num if c not in {"created_month", "created_dow", "created_hour", "publication_month", "publication_dow", "publication_hour"}],
        "categorical": [c for c in base_cat if c in {"origin_type", "category.name", "action_reason.name"}]
    }
    catboost = {
        "bool": base_bool,
        "numeric": base_num,
        "categorical": [c for c in base_cat if c not in {"category.id"}] + ([ "category.id" ] if "category.id" in present else [])
    }
    nn = {
        "bool": base_bool,
        "numeric": [c for c in base_num if c not in {"created_month", "created_dow", "created_hour", "publication_month", "publication_dow", "publication_hour"}],
        "categorical": [c for c in base_cat if c in {"origin_type", "category.id", "author.id"}]
    }
    for k in (linear, catboost, nn):
        k["bool"] = sorted(list(dict.fromkeys(k["bool"])))
        k["numeric"] = sorted(list(dict.fromkeys(k["numeric"])))
        k["categorical"] = sorted(list(dict.fromkeys(k["categorical"])))
    return {"linear": linear, "catboost": catboost, "nn": nn}


def compute_nonneg_numeric(df: pd.DataFrame, numeric_cols: List[str]) -> List[str]:
    out = []
    for c in numeric_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.dropna().empty:
            continue
        if float(s.min()) >= 0.0:
            out.append(c)
    return sorted(out)


def most_frequent_values(df: pd.DataFrame, col: str, max_values: int) -> List[str]:
    s = df[col].astype(str).fillna("")
    vc = s.value_counts(dropna=False)
    vals = vc.index.astype(str).tolist()[:max_values]
    if "" not in vals:
        vals = [""] + vals
    return vals


def is_standardized_numeric(x: np.ndarray, mean_tol: float = 0.25, std_tol: float = 0.25) -> bool:
    if x.size == 0:
        return True
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    mu_ok = np.nanmean(np.abs(mu)) <= mean_tol
    sd_ok = np.nanmean(np.abs(sd - 1.0)) <= std_tol
    return bool(mu_ok and sd_ok)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def safe_numpy_metric(expr: str, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[Optional[float], Optional[str]]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    env_g = {"__builtins__": {}, "np": np}
    env_l = {"y_true": y_true, "y_pred": y_pred}
    try:
        val = eval(expr, env_g, env_l)
        if isinstance(val, (np.ndarray, list, tuple)):
            val = float(np.asarray(val).reshape(-1)[0])
        return float(val), None
    except Exception as e:
        return None, str(e)


def time_holdout_split(df: pd.DataFrame, time_col: str, test_size: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    dt = pd.to_datetime(df[time_col], errors="coerce")
    mask = dt.notna()
    if mask.sum() == 0:
        idx = np.arange(len(df))
        rng = np.random.default_rng(42)
        rng.shuffle(idx)
        n_test = int(math.ceil(len(df) * test_size))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        info = {"method": "random", "seed": 42, "test_size": test_size}
        return train_idx, test_idx, info
    order = np.argsort(dt.fillna(pd.Timestamp.min).values)
    n_test = int(math.ceil(len(df) * test_size))
    test_idx = order[-n_test:]
    train_idx = order[:-n_test]
    cutoff = dt.iloc[test_idx].min()
    info = {"method": "time", "time_col": time_col, "test_size": test_size, "cutoff_min_in_test": str(cutoff)}
    return train_idx, test_idx, info
