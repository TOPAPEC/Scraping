import argparse
from typing import Dict, List, Tuple
import json
import os
import numpy as np
import pandas as pd
import joblib
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge

from catboost import CatBoostRegressor

from ml_core import (
    setup_logging, prepare_dataframe, add_derived_features, time_holdout_split,
    ensure_dir
)


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def df_select(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols]


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


def linear_fit_save(df_train: pd.DataFrame, y_train: np.ndarray, meta: dict, out_dir: str) -> None:
    num_cols = meta["numeric"]
    bool_cols = meta["bool"]
    cat_cols = meta["categorical"]

    d = clean_categorical(clean_numeric(clean_bool(df_train, bool_cols), num_cols), cat_cols)
    X_num = d[num_cols].to_numpy(dtype=np.float32) if len(num_cols) else np.zeros((len(d), 0), dtype=np.float32)
    X_bool = d[bool_cols].astype(int).to_numpy(dtype=np.float32) if len(bool_cols) else np.zeros((len(d), 0), dtype=np.float32)

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_num_s = scaler.fit_transform(X_num) if X_num.shape[1] else X_num

    ohe = make_ohe()
    X_cat = ohe.fit_transform(d[cat_cols]) if len(cat_cols) else sp.csr_matrix((len(d), 0))

    X = sp.hstack([sp.csr_matrix(X_num_s), sp.csr_matrix(X_bool), X_cat], format="csr")
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X, y_train)

    ensure_dir(out_dir)
    joblib.dump(scaler, f"{out_dir}/linear_scaler.joblib")
    joblib.dump(ohe, f"{out_dir}/linear_ohe.joblib")
    joblib.dump(model, f"{out_dir}/linear_model.joblib")

    feat_names = []
    feat_names += [f"NUM::{c}" for c in num_cols]
    feat_names += [f"BOOL::{c}" for c in bool_cols]
    if len(cat_cols):
        ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
        feat_names += [f"CAT::{x}" for x in ohe_names]

    lm = {
        "expects_scaled_numeric": True,
        "numeric": num_cols,
        "bool": bool_cols,
        "categorical": cat_cols,
        "feature_names": feat_names
    }
    with open(f"{out_dir}/linear_meta.json", "w", encoding="utf-8") as f:
        json.dump(lm, f, ensure_ascii=False, indent=2)


def catboost_fit_save(df_train: pd.DataFrame, y_train: np.ndarray, meta: dict, out_dir: str) -> None:
    num_cols = meta["numeric"]
    bool_cols = meta["bool"]
    cat_cols = meta["categorical"]

    d = clean_categorical(clean_numeric(clean_bool(df_train, bool_cols), num_cols), cat_cols)
    X = pd.concat(
        [d[num_cols], d[bool_cols].astype(int), d[cat_cols].astype(str)],
        axis=1
    )
    cat_idx = list(range(len(num_cols) + len(bool_cols), len(num_cols) + len(bool_cols) + len(cat_cols)))

    model = CatBoostRegressor(
        loss_function="RMSE",
        random_seed=42,
        iterations=2000,
        depth=8,
        learning_rate=0.05,
        l2_leaf_reg=5.0,
        eval_metric="RMSE",
        verbose=True,
        task_type="GPU",
    )
    model.fit(X, y_train, cat_features=cat_idx)

    ensure_dir(out_dir)
    model.save_model(f"{out_dir}/catboost.cbm")

    cm = {
        "expects_scaled_numeric": False,
        "numeric": num_cols,
        "bool": bool_cols,
        "categorical": cat_cols,
        "cat_feature_indices": cat_idx
    }
    with open(f"{out_dir}/catboost_meta.json", "w", encoding="utf-8") as f:
        json.dump(cm, f, ensure_ascii=False, indent=2)


def build_vocab(series: pd.Series, min_freq: int = 2, max_vocab: int = 200000) -> Dict[str, int]:
    s = series.astype(str).fillna("")
    vc = s.value_counts()
    items = vc[vc >= min_freq].index.astype(str).tolist()
    items = items[:max_vocab]
    vocab = {"": 0, "__UNK__": 0}
    idx = 1
    for v in items:
        if v in vocab:
            continue
        vocab[v] = idx
        idx += 1
    return vocab


class NNDataset(Dataset):
    def __init__(self, df: pd.DataFrame, y: np.ndarray, num_cols: list, bool_cols: list, cat_cols: list, scaler: StandardScaler, vocabs: Dict[str, Dict[str, int]], scale_numeric: bool):
        d = clean_categorical(clean_numeric(clean_bool(df, bool_cols), num_cols), cat_cols)
        X_num = d[num_cols].to_numpy(dtype=np.float32) if len(num_cols) else np.zeros((len(d), 0), dtype=np.float32)
        X_bool = d[bool_cols].astype(int).to_numpy(dtype=np.float32) if len(bool_cols) else np.zeros((len(d), 0), dtype=np.float32)
        if scale_numeric and X_num.shape[1]:
            X_num = scaler.transform(X_num).astype(np.float32)
        self.x_num = torch.from_numpy(np.concatenate([X_num, X_bool], axis=1).astype(np.float32))
        self.x_cat = []
        for c in cat_cols:
            vocab = vocabs[c]
            arr = d[c].astype(str).fillna("").map(lambda v: vocab.get(v, 0)).astype(np.int64).to_numpy()
            self.x_cat.append(torch.from_numpy(arr))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.float32).reshape(-1, 1))

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, i: int):
        cats = [t[i] for t in self.x_cat]
        return self.x_num[i], cats, self.y[i]


class EmbMLP(nn.Module):
    def __init__(self, num_in: int, cat_card_dims: List[Tuple[int, int]]):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(card, dim) for card, dim in cat_card_dims])
        emb_out = int(sum(dim for _, dim in cat_card_dims))
        h1 = 256
        h2 = 128
        h3 = 64
        self.fc1 = nn.Linear(num_in + emb_out, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, 1)

    def forward(self, x_num: torch.Tensor, x_cat: List[torch.Tensor]) -> torch.Tensor:
        embs = []
        for emb, xc in zip(self.embs, x_cat):
            embs.append(emb(xc))
        x = torch.cat([x_num] + embs, dim=1) if embs else x_num
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)


def emb_dim(card: int) -> int:
    return int(min(64, max(4, round((card + 1) ** 0.25 * 8))))


def nn_fit_save(df_train: pd.DataFrame, y_train: np.ndarray, meta: dict, out_dir: str) -> None:
    num_cols = meta["numeric"]
    bool_cols = meta["bool"]
    cat_cols = meta["categorical"]

    d = clean_categorical(clean_numeric(clean_bool(df_train, bool_cols), num_cols), cat_cols)

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_num = d[num_cols].to_numpy(dtype=np.float32) if len(num_cols) else np.zeros((len(d), 0), dtype=np.float32)
    if X_num.shape[1]:
        scaler.fit(X_num)

    vocabs = {c: build_vocab(d[c]) for c in cat_cols}
    cat_card_dims = []
    for c in cat_cols:
        card = int(max(vocabs[c].values(), default=0) + 1)
        cat_card_dims.append((card, emb_dim(card)))

    num_in = int((len(num_cols)) + (len(bool_cols)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = NNDataset(d, y_train, num_cols, bool_cols, cat_cols, scaler, vocabs, scale_numeric=True)
    dl = DataLoader(ds, batch_size=1024, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))

    model = EmbMLP(num_in=num_in, cat_card_dims=cat_card_dims).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    model.train()
    epochs = 6
    for _ in range(epochs):
        for xb_num, xb_cat, yb in dl:
            xb_num = xb_num.to(device)
            xb_cat = [t.to(device) for t in xb_cat]
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb_num, xb_cat)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

    ensure_dir(out_dir)
    joblib.dump(scaler, f"{out_dir}/nn_scaler.joblib")
    with open(f"{out_dir}/nn_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "expects_scaled_numeric": True,
                "numeric": num_cols,
                "bool": bool_cols,
                "categorical": cat_cols,
                "vocabs": vocabs,
                "cat_card_dims": cat_card_dims
            },
            f,
            ensure_ascii=False,
            indent=2
        )
    torch.save(model.state_dict(), f"{out_dir}/nn_state.pt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--channels", required=False, default=None)
    ap.add_argument("--schema", required=False, default="reports/schema.json")
    ap.add_argument("--target", required=False, default="hits")
    ap.add_argument("--models_dir", required=False, default="models")
    ap.add_argument("--test_size", required=False, type=float, default=0.2)
    args = ap.parse_args()

    ensure_dir(args.models_dir)
    logger = setup_logging(f"{args.models_dir}/train.log")

    if not os.path.exists(args.schema):
        raise ValueError(f"schema not found: {args.schema}. Run 01_data_analysis.py first.")

    with open(args.schema, "r", encoding="utf-8") as f:
        schema = json.load(f)

    df = prepare_dataframe(args.videos, args.channels)
    df = add_derived_features(df)

    if args.target not in df.columns:
        raise ValueError(f"target '{args.target}' not found")

    df = df.drop_duplicates(subset=["id"]) if "id" in df.columns else df

    y = pd.to_numeric(df[args.target], errors="coerce").fillna(0.0).astype(float).values

    time_col = "publication_ts" if "publication_ts" in df.columns else ("created_ts" if "created_ts" in df.columns else None)
    if time_col is None:
        df["_dummy_time"] = np.arange(len(df))
        time_col = "_dummy_time"

    train_idx, test_idx, split_info = time_holdout_split(df, time_col, args.test_size)
    with open(f"{args.models_dir}/split.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    df_holdout = df.iloc[test_idx].copy()
    df_holdout.to_parquet(f"{args.models_dir}/holdout.parquet", index=False)

    logger.info(f"split={split_info} holdout_rows={len(df_holdout)}")

    fs = schema["feature_sets"]
    for key in ["linear", "catboost", "nn"]:
        meta = fs[key]
        feats = meta["numeric"] + meta["bool"] + meta["categorical"]
        for c in feats:
            if c not in df.columns:
                df[c] = np.nan

    df_train = df.iloc[train_idx].copy()
    y_train = y[train_idx]

    linear_fit_save(df_train, y_train, fs["linear"], args.models_dir)
    logger.info("saved_linear")

    catboost_fit_save(df_train, y_train, fs["catboost"], args.models_dir)
    logger.info("saved_catboost")

    nn_fit_save(df_train, y_train, fs["nn"], args.models_dir)
    logger.info("saved_nn")

    with open(f"{args.models_dir}/schema_used.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    logger.info(f"done models_dir={args.models_dir}")


if __name__ == "__main__":
    main()
