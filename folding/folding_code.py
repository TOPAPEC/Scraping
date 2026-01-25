import argparse
import json
import time
import base64
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from catboost import CatBoostRegressor, Pool

def b64fig():
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    s = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return s

def load_ndjson(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def build_dataset(path_vid, path_ch):
    df_v = pd.json_normalize(load_ndjson(path_vid))
    df_c = pd.json_normalize(load_ndjson(path_ch))

    df_v["created_ts"] = pd.to_datetime(df_v.get("created_ts"), errors="coerce")
    df_v["hour"] = df_v["created_ts"].dt.hour.fillna(-1).astype(int)
    df_v["dow"] = df_v["created_ts"].dt.dayofweek.fillna(-1).astype(int)
    df_v["month"] = df_v["created_ts"].dt.month.fillna(-1).astype(int)
    df_v["title_len"] = df_v.get("title", "").fillna("").astype(str).str.len()
    df_v["desc_len"] = df_v.get("description", "").fillna("").astype(str).str.len()
    df_v["log_duration"] = np.log1p(pd.to_numeric(df_v.get("duration", 0), errors="coerce").fillna(0))
    df_v["log_desc_len"] = np.log1p(pd.to_numeric(df_v["desc_len"], errors="coerce").fillna(0))
    df_v["log_hits"] = np.log1p(pd.to_numeric(df_v.get("hits"), errors="coerce").astype(float))

    df_c = df_c.rename(columns={c: f"ch_{c}" for c in df_c.columns if c != "channel_id"})
    df_c["ch_subscribers"] = pd.to_numeric(df_c.get("ch_subscribers", 0), errors="coerce").fillna(0)
    df_c["ch_title_len"] = df_c.get("ch_title", "").fillna("").astype(str).str.len()
    df_c["ch_desc_len"] = df_c.get("ch_description", "").fillna("").astype(str).str.len()
    df_c["log_ch_subscribers"] = np.log1p(df_c["ch_subscribers"])
    ch_meta_cols = [c for c in df_c.columns if str(c).startswith("ch_meta.")]
    df_c["ch_meta_count"] = df_c[ch_meta_cols].notna().sum(axis=1) if ch_meta_cols else 0

    df = df_v.merge(df_c, left_on="author.id", right_on="channel_id", how="left")

    exclude_cols = [
        "id","track_id","author.id","channel_id","category.id","action_reason.id",
        "title","description","feed_name",
        "author.name","author.avatar_url","author.site_url",
        "category.name","category.category_url","pg_rating.logo","action_reason.name",
        "video_url","thumbnail_url","picture_url","preview_url","embed_url","feed_url","html",
        "ch_title","ch_description","ch_avatar_url","ch_url",
        "common_subscription_product_codes","ch_jsonld","ch_meta",
        "created_ts","last_update_ts","publication_ts","future_publication",
        "stream_type","product_id",
        "duration","desc_len","ch_subscribers"
    ]
    exclude_cols += [c for c in df.columns if str(c).startswith("ch_meta.")]
    X_cols = [c for c in df.columns if c not in set(exclude_cols + ["hits", "log_hits"])]

    df_model = df[X_cols + ["log_hits"]].dropna(subset=["log_hits"]).copy()
    X = df_model[X_cols].copy()
    y = df_model["log_hits"].to_numpy(dtype=float)

    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
        else:
            X[c] = X[c].fillna("Unknown").astype(str)

    cat_features = [i for i, c in enumerate(X.columns) if not pd.api.types.is_numeric_dtype(X[c])]
    groups = df.loc[df_model.index, "author.id"].fillna("Unknown").astype(str).to_numpy()
    ts = pd.to_datetime(df.loc[df_model.index, "created_ts"], errors="coerce")
    return X, y, cat_features, groups, ts

def r2_np(y, p):
    y = np.asarray(y).reshape(-1)
    p = np.asarray(p).reshape(-1)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 0.0 if ss_tot == 0 else float(1.0 - ss_res / ss_tot)

def rmse_np(y, p):
    y = np.asarray(y).reshape(-1)
    p = np.asarray(p).reshape(-1)
    return float(np.sqrt(np.mean((y - p) ** 2)))

def mae_np(y, p):
    y = np.asarray(y).reshape(-1)
    p = np.asarray(p).reshape(-1)
    return float(np.mean(np.abs(y - p)))

def _rng(seed):
    return np.random.default_rng(seed)

def holdout_random(n, test_size=0.2, seed=42):
    idx = np.arange(n)
    _rng(seed).shuffle(idx)
    n_test = int(np.floor(n * test_size))
    n_test = max(1, min(n - 1, n_test))
    return idx[n_test:], idx[:n_test]

def holdout_time(ts, test_size=0.2):
    order = np.argsort(pd.to_datetime(ts, errors="coerce").fillna(pd.Timestamp.min).to_numpy(dtype="datetime64[ns]"))
    n = len(order)
    n_test = int(np.floor(n * test_size))
    n_test = max(1, min(n - 1, n_test))
    return order[:-n_test].astype(int), order[-n_test:].astype(int)

class KFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=42):
        self.k = int(n_splits)
        self.shuffle = bool(shuffle)
        self.seed = int(random_state)

    def split(self, X, y=None, groups=None):
        n = len(X)
        idx = np.arange(n)
        if self.shuffle:
            _rng(self.seed).shuffle(idx)
        fs = np.full(self.k, n // self.k, dtype=int)
        fs[: n % self.k] += 1
        st = np.cumsum(np.r_[0, fs[:-1]])
        for f, s in zip(fs, st):
            v = idx[s:s + f]
            t = np.concatenate([idx[:s], idx[s + f:]])
            yield t, v

class StratifiedKFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=42, bins=10):
        self.k = int(n_splits)
        self.shuffle = bool(shuffle)
        self.seed = int(random_state)
        self.bins = int(bins)

    def make_bins(self, y):
        y = np.asarray(y).reshape(-1)
        edges = np.unique(np.quantile(y, np.linspace(0, 1, self.bins + 1)))
        return np.zeros_like(y, dtype=int) if len(edges) < 3 else np.digitize(y, edges[1:-1], right=False).astype(int)

    def split(self, X, y=None, groups=None):
        y = np.asarray(y).reshape(-1)
        labels = self.make_bins(y)
        r = _rng(self.seed)
        uniq, inv = np.unique(labels, return_inverse=True)
        folds = [[] for _ in range(self.k)]
        for c in range(len(uniq)):
            idx = np.where(inv == c)[0]
            if self.shuffle:
                r.shuffle(idx)
            for j, ix in enumerate(idx):
                folds[j % self.k].append(int(ix))
        n = len(y)
        for j in range(self.k):
            v = np.array(folds[j], dtype=int)
            v.sort()
            m = np.ones(n, dtype=bool)
            m[v] = False
            t = np.where(m)[0]
            yield t, v

class LeaveOneOut200:
    def __init__(self, m=200, seed=42):
        self.m = int(m)
        self.seed = int(seed)

    def split(self, X, y=None, groups=None):
        n = len(X)
        m = min(self.m, n)
        idx = _rng(self.seed).choice(n, size=m, replace=False)
        for i in idx:
            v = np.array([int(i)], dtype=int)
            t = np.concatenate([np.arange(0, int(i), dtype=int), np.arange(int(i) + 1, n, dtype=int)])
            yield t, v

class GroupKFold:
    def __init__(self, n_splits=5):
        self.k = int(n_splits)

    def split(self, X, y=None, groups=None):
        g = np.asarray(groups).astype(str).reshape(-1)
        uniq, inv = np.unique(g, return_inverse=True)
        sizes = np.bincount(inv).astype(int)
        order = np.argsort(-sizes)
        fold_g = [[] for _ in range(self.k)]
        fold_sz = np.zeros(self.k, dtype=int)
        for gi in order:
            fj = int(np.argmin(fold_sz))
            fold_g[fj].append(int(gi))
            fold_sz[fj] += sizes[gi]
        n = len(g)
        for j in range(self.k):
            vg = set(fold_g[j])
            v = np.where(np.isin(inv, list(vg)))[0].astype(int)
            m = np.ones(n, dtype=bool)
            m[v] = False
            t = np.where(m)[0].astype(int)
            yield t, v

class TimeSeriesSplit:
    def __init__(self, n_splits=5, test_size=None, gap=0, max_train_size=None):
        self.k = int(n_splits)
        self.test_size = None if test_size is None else int(test_size)
        self.gap = int(gap)
        self.max_train_size = None if max_train_size is None else int(max_train_size)

    def split(self, X, y=None, groups=None):
        n = len(X)
        ts = self.test_size if self.test_size is not None else n // (self.k + 1)
        for k in range(self.k):
            train_end = n - (self.k - k) * ts - self.gap
            val_start = train_end + self.gap
            val_end = val_start + ts
            t0 = 0 if self.max_train_size is None else max(0, train_end - self.max_train_size)
            t = np.arange(t0, train_end, dtype=int)
            v = np.arange(val_start, val_end, dtype=int)
            yield t, v

def fit_predict_cb(Xtr, ytr, Xva, yva, cat_features, params):
    trp = Pool(Xtr, ytr, cat_features=cat_features)
    vap = Pool(Xva, yva, cat_features=cat_features)
    m = CatBoostRegressor(**params)
    t0 = time.time()
    m.fit(trp, eval_set=vap, verbose=False)
    dt = time.time() - t0
    pred = m.predict(vap)
    return pred, dt

def compute_bin_edges(y, bins=12):
    y = np.asarray(y).reshape(-1)
    edges = np.unique(np.quantile(y, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        mn, mx = float(np.min(y)), float(np.max(y))
        edges = np.linspace(mn, mx, bins + 1)
    return edges

def bin_ids(y, edges):
    y = np.asarray(y).reshape(-1)
    return np.digitize(y, edges[1:-1], right=False).astype(int)

def fold_bin_props(y, folds, edges):
    b = bin_ids(y, edges)
    B = int(np.max(b) + 1)
    P_val = []
    P_tr = []
    for tr, va in folds:
        bv = b[va]
        bt = b[tr]
        pv = np.bincount(bv, minlength=B).astype(float)
        pt = np.bincount(bt, minlength=B).astype(float)
        pv = pv / max(1.0, pv.sum())
        pt = pt / max(1.0, pt.sum())
        P_val.append(pv)
        P_tr.append(pt)
    return np.vstack(P_tr), np.vstack(P_val)

def plot_split_map(n, folds, order=None, max_n=2500, title=""):
    n_show = min(int(max_n), int(n))
    ordv = np.arange(n, dtype=int) if order is None else np.asarray(order).astype(int)
    inv = np.empty(n, dtype=int)
    inv[ordv] = np.arange(n, dtype=int)
    M = np.zeros((len(folds), n_show), dtype=float)
    for i, (_, va) in enumerate(folds):
        pos = inv[np.asarray(va, dtype=int)]
        pos = pos[pos < n_show]
        M[i, pos] = 1.0
    plt.figure(figsize=(12, 0.35 * max(6, len(folds))))
    plt.imshow(M, aspect="auto", interpolation="nearest")
    plt.title(title)
    plt.xlabel("индекс объекта (в выбранном порядке)")
    plt.ylabel("фолд")
    plt.colorbar(fraction=0.02, pad=0.02)
    return b64fig()

def plot_bin_heatmaps(P_tr, P_val, title=""):
    plt.figure(figsize=(12, 6))
    plt.imshow(P_val, aspect="auto", interpolation="nearest")
    plt.title(title + " — доли бинов таргета в VAL по фолдам")
    plt.xlabel("бин таргета (log_hits)")
    plt.ylabel("фолд")
    plt.colorbar(fraction=0.02, pad=0.02)
    img1 = b64fig()

    plt.figure(figsize=(12, 6))
    plt.imshow(P_tr, aspect="auto", interpolation="nearest")
    plt.title(title + " — доли бинов таргета в TRAIN по фолдам")
    plt.xlabel("бин таргета (log_hits)")
    plt.ylabel("фолд")
    plt.colorbar(fraction=0.02, pad=0.02)
    img2 = b64fig()

    return img1, img2

def plot_group_bars(groups, folds, title=""):
    g = np.asarray(groups).astype(str)
    vals = []
    for _, va in folds:
        vals.append((len(va), len(np.unique(g[va]))))
    vals = np.asarray(vals, dtype=float)
    x = np.arange(len(folds))
    plt.figure(figsize=(12, 5))
    plt.bar(x - 0.2, vals[:, 0], width=0.4)
    plt.bar(x + 0.2, vals[:, 1], width=0.4)
    plt.title(title + " — размер VAL и число уникальных групп в VAL")
    plt.xlabel("фолд")
    plt.ylabel("количество")
    plt.xticks(x, [str(i) for i in x])
    plt.legend(["|VAL|", "unique groups in VAL"])
    return b64fig()

def plot_time_windows(n, folds, title=""):
    plt.figure(figsize=(12, 0.5 * max(6, len(folds))))
    for i, (tr, va) in enumerate(folds):
        tr0, tr1 = int(np.min(tr)), int(np.max(tr)) + 1
        va0, va1 = int(np.min(va)), int(np.max(va)) + 1
        plt.plot([tr0, tr1], [i, i], linewidth=6)
        plt.plot([va0, va1], [i, i], linewidth=6)
    plt.xlim(0, n)
    plt.yticks(np.arange(len(folds)), [str(i) for i in range(len(folds))])
    plt.title(title + " — окна TRAIN/VAL (по времени, индекс после сортировки)")
    plt.xlabel("временной индекс")
    plt.ylabel("фолд")
    plt.legend(["TRAIN", "VAL"], loc="lower right")
    return b64fig()

def plot_loo_error_scatter(y, folds, preds, title=""):
    idx = []
    err = []
    for (tr, va), p in zip(folds, preds):
        i = int(va[0])
        idx.append(i)
        err.append(float(abs(y[i] - p[0])))
    idx = np.asarray(idx)
    err = np.asarray(err)
    o = np.argsort(idx)
    plt.figure(figsize=(12, 4))
    plt.plot(idx[o], err[o], marker="o", linestyle="-")
    plt.title(title + " — |ошибка| на точках LOO (log-шкала)")
    plt.xlabel("индекс объекта")
    plt.ylabel("|y - ŷ|")
    return b64fig()

def plot_summary_boxplots(rows, metric_key, title):
    names = [r["strategy"] for r in rows]
    data = [r["per_fold"][metric_key] for r in rows]
    plt.figure(figsize=(12, 6))
    plt.boxplot(data, labels=names, showfliers=False)
    plt.title(title)
    plt.ylabel(metric_key)
    plt.xticks(rotation=20, ha="right")
    return b64fig()

def plot_runtime_bars(rows, title):
    names = [r["strategy"] for r in rows]
    times = [r["total_fit_time"] for r in rows]
    plt.figure(figsize=(12, 5))
    plt.bar(names, times)
    plt.title(title)
    plt.ylabel("секунды (суммарное время fit по фолдам)")
    plt.xticks(rotation=20, ha="right")
    return b64fig()

def run_strategy(name, splitter, X, y, cat_features, params, groups=None, order=None, progress_total=None):
    X0 = X
    y0 = y
    g0 = groups
    if order is not None:
        X0 = X.iloc[order].reset_index(drop=True)
        y0 = y[order]
        g0 = g0[order] if g0 is not None else None

    folds = list(splitter.split(X0, y=y0, groups=g0))
    per_r2, per_rmse, per_mae, per_fit = [], [], [], []
    preds_store = []
    it = folds
    for tr, va in tqdm(it, total=progress_total, desc=name, leave=False):
        pred, dt = fit_predict_cb(X0.iloc[tr], y0[tr], X0.iloc[va], y0[va], cat_features, params)
        per_fit.append(dt)
        per_r2.append(r2_np(y0[va], pred))
        per_rmse.append(rmse_np(y0[va], pred))
        per_mae.append(mae_np(y0[va], pred))
        preds_store.append(pred)

    per_fold = {
        "R2": np.asarray(per_r2, dtype=float),
        "RMSE": np.asarray(per_rmse, dtype=float),
        "MAE": np.asarray(per_mae, dtype=float),
        "fit_time": np.asarray(per_fit, dtype=float)
    }

    return {
        "strategy": name,
        "folds": folds,
        "order": None if order is None else np.asarray(order).astype(int),
        "X_used_n": len(X0),
        "y_used": y0,
        "groups_used": g0,
        "per_fold": per_fold,
        "preds": preds_store,
        "total_fit_time": float(np.sum(per_fold["fit_time"])),
        "rmse_mean": float(np.mean(per_fold["RMSE"])),
        "rmse_std": float(np.std(per_fold["RMSE"])),
        "r2_mean": float(np.mean(per_fold["R2"])),
        "r2_std": float(np.std(per_fold["R2"])),
        "mae_mean": float(np.mean(per_fold["MAE"])),
        "mae_std": float(np.std(per_fold["MAE"]))
    }

def make_html_report(out_path, meta, blocks, summary_table_html):
    style = """
    <style>
      body{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#222;background:#fafafa}
      .card{background:#fff;border:1px solid #e6e6e6;border-radius:10px;padding:18px;margin:16px 0;box-shadow:0 2px 10px rgba(0,0,0,0.04)}
      h1,h2,h3{margin:0 0 10px 0}
      .muted{color:#666;font-size:13px}
      img{max-width:1200px;width:100%;border:1px solid #eee;border-radius:10px}
      code{background:#f4f4f4;padding:2px 6px;border-radius:6px}
      table{border-collapse:collapse;width:100%;font-size:13px}
      th,td{border:1px solid #ddd;padding:8px;text-align:left}
      th{background:#f0f0f0}
      .grid{display:grid;grid-template-columns:1fr;gap:12px}
    </style>
    """
    html = f"""
    <!doctype html>
    <html lang="ru">
    <head>
      <meta charset="utf-8">
      <title>Отчёт: стратегии кросс-валидации и визуализация разбиений</title>
      {style}
    </head>
    <body>
      <h1>Стратегии кросс-валидации: реализация без sklearn + визуализация</h1>
      <div class="muted">Датасет: Rutube (videos.ndjson + channels.ndjson). Модель: CatBoostRegressor (GPU, iterations=100).</div>

      <div class="card">
        <h2>Ключевые обозначения и формулы</h2>
        <div class="grid">
          <div>
            Пусть <code>I = {{0,1,...,n-1}}</code> — индексы объектов. Для каждого фолда <code>j</code>:
            <ul>
              <li><code>V_j</code> — множество валидации (val)</li>
              <li><code>T_j = I \\ V_j</code> — множество обучения (train)</li>
            </ul>
            Метрики по фолдам: <code>RMSE</code>, <code>MAE</code>, <code>R²</code>. Также считаем время обучения на фолде.
          </div>
          <div>
            <b>K-Fold:</b> <code>I = ⊔_{{j=1..K}} V_j</code>, <code>|V_j|≈n/K</code>.<br>
            <b>Stratified (по бинам таргета):</b> строим бин <code>b_i</code> для каждого объекта и распределяем индексы по фолдам так, чтобы доли <code>P(b)</code> в каждом <code>V_j</code> были близки к глобальным.<br>
            <b>LOO:</b> <code>V_i={{i}}</code>, <code>T_i=I\\{{i}}</code>. Здесь используем sampled-LOO: проверяем ровно 200 индексов (val), train при этом полный (все остальные).<br>
            <b>GroupKFold:</b> есть группы <code>g_i</code>, и запрещаем пересечение групп между train и val: <code>{{g_i: i∈V_j}} ∩ {{g_i: i∈T_j}} = ∅</code>.<br>
            <b>TimeSeriesSplit:</b> сортируем по времени и делаем train как префикс, val как следующий блок (без “заглядывания в будущее”).
          </div>
        </div>
      </div>

      <div class="card">
        <h2>Сводная таблица результатов</h2>
        {summary_table_html}
      </div>

      {''.join(blocks)}

    </body>
    </html>
    """
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

def df_to_html_table(df):
    return df.to_html(index=False, escape=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path_vid", type=str, default=r"scraped_data\v1_vids_and_channels_100k\videos.ndjson")
    ap.add_argument("--path_ch", type=str, default=r"scraped_data\v1_vids_and_channels_100k\channels.ndjson")
    ap.add_argument("--out_html", type=str, default="cv_report.html")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--strat_bins", type=int, default=12)
    ap.add_argument("--holdout", type=str, default="random", choices=["random", "time"])
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--time_test_size", type=int, default=None)
    ap.add_argument("--time_gap", type=int, default=0)
    ap.add_argument("--time_max_train_size", type=int, default=None)
    ap.add_argument("--loo_m", type=int, default=200)
    args = ap.parse_args()

    X, y, cat_features, groups, ts = build_dataset(args.path_vid, args.path_ch)

    if args.holdout == "time":
        tr_idx, te_idx = holdout_time(ts, test_size=args.test_size)
    else:
        tr_idx, te_idx = holdout_random(len(X), test_size=args.test_size, seed=args.seed)

    Xtr = X.iloc[tr_idx].reset_index(drop=True)
    ytr = y[tr_idx]
    gtr = groups[tr_idx]
    tstr = pd.to_datetime(ts.iloc[tr_idx].reset_index(drop=True), errors="coerce")
    order_time = np.argsort(tstr.fillna(pd.Timestamp.min).to_numpy(dtype="datetime64[ns]")).astype(int)

    params = dict(
        iterations=100,
        learning_rate=0.1,
        depth=8,
        loss_function="RMSE",
        random_seed=int(args.seed),
        task_type="GPU"
    )

    strat = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed, bins=args.strat_bins)
    kfold = KFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    loo = LeaveOneOut200(m=int(args.loo_m), seed=args.seed)
    gkf = GroupKFold(n_splits=args.k)
    tss = TimeSeriesSplit(n_splits=args.k, test_size=args.time_test_size, gap=args.time_gap, max_train_size=args.time_max_train_size)

    strategies = [
        ("KFold", kfold, dict(groups=None, order=None, progress_total=args.k)),
        (f"StratifiedKFold(bins={args.strat_bins})", strat, dict(groups=None, order=None, progress_total=args.k)),
        (f"LOO(sampled m={args.loo_m})", loo, dict(groups=None, order=None, progress_total=min(args.loo_m, len(Xtr)))),
        ("GroupKFold(author.id)", gkf, dict(groups=gtr, order=None, progress_total=args.k)),
        ("TimeSeriesSplit(created_ts)", tss, dict(groups=None, order=order_time, progress_total=args.k)),
    ]

    results = []
    for name, splitter, kw in tqdm(strategies, desc="All strategies"):
        results.append(run_strategy(name, splitter, Xtr, ytr, cat_features, params, **kw))

    rows = []
    for r in results:
        rows.append({
            "strategy": r["strategy"],
            "folds": len(r["folds"]),
            "RMSE_mean": r["rmse_mean"],
            "RMSE_std": r["rmse_std"],
            "R2_mean": r["r2_mean"],
            "R2_std": r["r2_std"],
            "MAE_mean": r["mae_mean"],
            "MAE_std": r["mae_std"],
            "total_fit_time_s": r["total_fit_time"]
        })
    df_sum = pd.DataFrame(rows)
    summary_table_html = df_to_html_table(df_sum)

    edges = compute_bin_edges(ytr, bins=12)

    blocks = []

    img_box_rmse = plot_summary_boxplots(results, "RMSE", "Сравнение стратегий: распределение RMSE по фолдам")
    img_box_r2 = plot_summary_boxplots(results, "R2", "Сравнение стратегий: распределение R² по фолдам")
    img_time = plot_runtime_bars(results, "Сравнение стратегий: суммарное время обучения по фолдам")

    blocks.append(f"""
    <div class="card">
      <h2>Итоговые сравнения</h2>
      <p>Ниже — распределения метрик по фолдам (важно для понимания стабильности оценки) и суммарное время обучения (важно для учёта стоимости стратегии).</p>
      <h3>RMSE по фолдам</h3>
      <img src="data:image/png;base64,{img_box_rmse}">
      <h3>R² по фолдам</h3>
      <img src="data:image/png;base64,{img_box_r2}">
      <h3>Время обучения</h3>
      <img src="data:image/png;base64,{img_time}">
      <p class="muted">Интерпретация: стратегии с “честными” ограничениями (GroupKFold/TimeSeriesSplit) часто дают более пессимистичную, но реалистичную оценку. LOO (даже sampled) дорогой, потому что это много отдельных обучений.</p>
    </div>
    """)

    for r in results:
        name = r["strategy"]
        folds = r["folds"]
        y_used = r["y_used"]
        g_used = r["groups_used"]
        order = r["order"]
        n_used = r["X_used_n"]

        if name.startswith("LOO"):
            toy_n = 20
            toy_folds = []
            for i in range(toy_n):
                v = np.array([i], dtype=int)
                t = np.concatenate([np.arange(0, i, dtype=int), np.arange(i + 1, toy_n, dtype=int)])
                toy_folds.append((t, v))
            img_split = plot_split_map(toy_n, toy_folds, order=None, max_n=toy_n, title=name + " — схема разбиения (toy n=20)")
            img_err = plot_loo_error_scatter(y_used, folds, r["preds"], title=name)
            P_tr, P_val = fold_bin_props(y_used, folds, edges)
            img_val, img_tr = plot_bin_heatmaps(P_tr, P_val, title=name)
            blocks.append(f"""
            <div class="card">
              <h2>{name}</h2>
              <p><b>Идея:</b> для выбранного индекса <code>i</code> берём <code>V={{i}}</code>, <code>T=I\\{{i}}</code>. В этом отчёте LOO делается приближённо: проверяем ровно <code>{min(args.loo_m, len(Xtr))}</code> объектов, но train каждый раз полный (все остальные).</p>
              <h3>Как выглядит разбиение (на маленьком примере)</h3>
              <img src="data:image/png;base64,{img_split}">
              <h3>Распределение ошибок по проверенным точкам</h3>
              <img src="data:image/png;base64,{img_err}">
              <h3>Сдвиг распределения таргета по бинам (VAL)</h3>
              <img src="data:image/png;base64,{img_val}">
              <h3>Сдвиг распределения таргета по бинам (TRAIN)</h3>
              <img src="data:image/png;base64,{img_tr}">
              <p class="muted">Почему медленно: это {min(args.loo_m, len(Xtr))} отдельных обучений CatBoost на почти полном train. Ускорение достигается только уменьшением числа проверок (m).</p>
            </div>
            """)
            continue

        if name.startswith("TimeSeriesSplit"):
            img_split = plot_split_map(n_used, folds, order=None, max_n=2500, title=name + " — split map (после сортировки по времени)")
            img_windows = plot_time_windows(n_used, folds, title=name)
            P_tr, P_val = fold_bin_props(y_used, folds, edges)
            img_val, img_tr = plot_bin_heatmaps(P_tr, P_val, title=name)
            blocks.append(f"""
            <div class="card">
              <h2>{name}</h2>
              <p><b>Идея:</b> сортируем по времени и строим фолды так, чтобы train всегда был “в прошлом”, а val — “в будущем”. Формально на фолде <code>j</code>: <code>T = [0..t_j)</code>, <code>V = [t_j+gap .. t_j+gap+test_size)</code>.</p>
              <h3>Split map (val отмечен единицами)</h3>
              <img src="data:image/png;base64,{img_split}">
              <h3>Окна train/val по временной оси</h3>
              <img src="data:image/png;base64,{img_windows}">
              <h3>Доли бинов таргета в VAL</h3>
              <img src="data:image/png;base64,{img_val}">
              <h3>Доли бинов таргета в TRAIN</h3>
              <img src="data:image/png;base64,{img_tr}">
              <p class="muted">Эта стратегия особенно важна при наличии трендов/дрейфа: случайные разбиения могут “подсмотреть будущее” и переоценить качество.</p>
            </div>
            """)
            continue

        if name.startswith("GroupKFold"):
            img_split = plot_split_map(n_used, folds, order=None, max_n=2500, title=name + " — split map (индексный порядок)")
            img_groups = plot_group_bars(g_used, folds, title=name)
            P_tr, P_val = fold_bin_props(y_used, folds, edges)
            img_val, img_tr = plot_bin_heatmaps(P_tr, P_val, title=name)
            blocks.append(f"""
            <div class="card">
              <h2>{name}</h2>
              <p><b>Идея:</b> запрещаем утечку по группам. Если есть риск, что объекты одной группы (например, один автор/канал) похожи, то нельзя, чтобы они оказались и в train, и в val.</p>
              <h3>Split map</h3>
              <img src="data:image/png;base64,{img_split}">
              <h3>Сколько объектов и уникальных групп попадает в VAL на каждом фолде</h3>
              <img src="data:image/png;base64,{img_groups}">
              <h3>Доли бинов таргета в VAL</h3>
              <img src="data:image/png;base64,{img_val}">
              <h3>Доли бинов таргета в TRAIN</h3>
              <img src="data:image/png;base64,{img_tr}">
              <p class="muted">Оценка часто становится более “строгой”, но она честнее, если в данных есть кластеризация по группам.</p>
            </div>
            """)
            continue

        if name.startswith("StratifiedKFold"):
            labels = strat.make_bins(y_used)
            vis_order = np.argsort(labels).astype(int)
            img_split = plot_split_map(n_used, folds, order=vis_order, max_n=2500, title=name + " — split map (порядок: сортировка по бину таргета)")
            P_tr, P_val = fold_bin_props(y_used, folds, edges)
            img_val, img_tr = plot_bin_heatmaps(P_tr, P_val, title=name)
            blocks.append(f"""
            <div class="card">
              <h2>{name}</h2>
              <p><b>Идея:</b> сначала дискретизируем непрерывный таргет на бины <code>b_i</code>, затем раскладываем индексы по фолдам так, чтобы доли бинов в каждом <code>V_j</code> были близки к глобальным.</p>
              <h3>Split map (видно, что val “равномерно” покрывает бины)</h3>
              <img src="data:image/png;base64,{img_split}">
              <h3>Доли бинов таргета в VAL</h3>
              <img src="data:image/png;base64,{img_val}">
              <h3>Доли бинов таргета в TRAIN</h3>
              <img src="data:image/png;base64,{img_tr}">
              <p class="muted">Это полезно при сильной асимметрии таргета: обычный KFold может случайно “перекосить” val по хвостам распределения.</p>
            </div>
            """)
            continue

        if name == "KFold":
            img_split = plot_split_map(n_used, folds, order=None, max_n=2500, title=name + " — split map")
            P_tr, P_val = fold_bin_props(y_used, folds, edges)
            img_val, img_tr = plot_bin_heatmaps(P_tr, P_val, title=name)
            blocks.append(f"""
            <div class="card">
              <h2>{name}</h2>
              <p><b>Идея:</b> разбиваем индексы <code>I</code> на <code>K</code> примерно равных непересекающихся частей <code>V_j</code>. На фолде <code>j</code> учим на <code>T_j = I \\ V_j</code> и проверяем на <code>V_j</code>.</p>
              <h3>Split map</h3>
              <img src="data:image/png;base64,{img_split}">
              <h3>Доли бинов таргета в VAL</h3>
              <img src="data:image/png;base64,{img_val}">
              <h3>Доли бинов таргета в TRAIN</h3>
              <img src="data:image/png;base64,{img_tr}">
              <p class="muted">Это базовая стратегия: быстрая и обычно стабильная, но может быть нечестной при групповой структуре или временном дрейфе.</p>
            </div>
            """)

    meta = {
        "n_total": len(X),
        "n_train": len(Xtr),
        "n_test": len(te_idx),
        "features": int(X.shape[1]),
        "cat_features": int(len(cat_features)),
        "params": params
    }

    make_html_report(args.out_html, meta, blocks, summary_table_html)
    print(f"Saved: {args.out_html}")

if __name__ == "__main__":
    main()
