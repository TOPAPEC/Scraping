import os
import time
import json
import base64
import itertools
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from catboost import CatBoostRegressor, Pool


PATH_VID = r"scraped_data\v1_vids_and_channels_100k\videos.ndjson"
PATH_CH = r"scraped_data\v1_vids_and_channels_100k\channels.ndjson"

OUT_HTML = "tuning_report.html"
GRID_CSV = "grid_results.csv"
GA_CSV = "ga_results.csv"

SEED = 42
HOLDOUT = "time"
TEST_SIZE = 0.2

CV_MODE = "group"
CV_K = 5
TIME_TEST_SIZE = None
TIME_GAP = 0
TIME_MAX_TRAIN_SIZE = None

TASK_TYPE = "GPU"
ITERATIONS = 100

REUSE_GRID = True
RUN_GA = True
REUSE_GA_IF_EXISTS = True

GA_BUDGET = 250
GA_POP = 25
GA_GENS = 10
TOPN = 20


def b64fig():
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=160)
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
        "id", "track_id", "author.id", "channel_id", "category.id", "action_reason.id",
        "title", "description", "feed_name",
        "author.name", "author.avatar_url", "author.site_url",
        "category.name", "category.category_url", "pg_rating.logo", "action_reason.name",
        "video_url", "thumbnail_url", "picture_url", "preview_url", "embed_url", "feed_url", "html",
        "ch_title", "ch_description", "ch_avatar_url", "ch_url",
        "common_subscription_product_codes", "ch_jsonld", "ch_meta",
        "created_ts", "last_update_ts", "publication_ts", "future_publication",
        "stream_type", "product_id",
        "duration", "desc_len", "ch_subscribers"
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
    return np.random.default_rng(int(seed))


def holdout_random(n, test_size=0.2, seed=42):
    idx = np.arange(n)
    _rng(seed).shuffle(idx)
    n_test = int(np.floor(n * float(test_size)))
    n_test = max(1, min(n - 1, n_test))
    return idx[n_test:], idx[:n_test]


def holdout_time(ts, test_size=0.2):
    order = np.argsort(pd.to_datetime(ts, errors="coerce").fillna(pd.Timestamp.min).to_numpy(dtype="datetime64[ns]"))
    n = len(order)
    n_test = int(np.floor(n * float(test_size)))
    n_test = max(1, min(n - 1, n_test))
    return order[:-n_test].astype(int), order[-n_test:].astype(int)


class GroupKFold:
    def __init__(self, n_splits=5):
        self.k = int(n_splits)

    def split(self, X, y=None, groups=None):
        g = np.asarray(groups).astype(str).reshape(-1)
        _, inv = np.unique(g, return_inverse=True)
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
            if len(v) and len(t):
                yield t, v


class TimeSeriesSplit:
    def __init__(self, n_splits=5, test_size=None, gap=0, max_train_size=None):
        self.k = int(n_splits)
        self.test_size = None if test_size is None else int(test_size)
        self.gap = int(gap)
        self.max_train_size = None if max_train_size is None else int(max_train_size)

    def split(self, X, y=None, groups=None):
        n = len(X)
        ts = self.test_size if self.test_size is not None else max(1, n // (self.k + 1))
        for k in range(self.k):
            train_end = n - (self.k - k) * ts - self.gap
            val_start = train_end + self.gap
            val_end = val_start + ts
            if train_end <= 1 or val_end > n:
                continue
            t0 = 0 if self.max_train_size is None else max(0, train_end - self.max_train_size)
            t = np.arange(t0, train_end, dtype=int)
            v = np.arange(val_start, val_end, dtype=int)
            if len(v) and len(t):
                yield t, v


def fit_predict_cb(Xtr, ytr, Xva, yva, cat_features, params):
    trp = Pool(Xtr, ytr, cat_features=cat_features)
    vap = Pool(Xva, yva, cat_features=cat_features)
    m = CatBoostRegressor(**params)
    t0 = time.time()
    m.fit(trp, eval_set=vap, verbose=False)
    dt = time.time() - t0
    pred = m.predict(vap)
    return pred, float(dt)


def cv_evaluate(X, y, cat_features, folds, params):
    r2s, rmses, maes, tts = [], [], [], []
    for tr, va in folds:
        pred, dt = fit_predict_cb(X.iloc[tr], y[tr], X.iloc[va], y[va], cat_features, params)
        r2s.append(r2_np(y[va], pred))
        rmses.append(rmse_np(y[va], pred))
        maes.append(mae_np(y[va], pred))
        tts.append(dt)
    r2s = np.asarray(r2s, dtype=float)
    rmses = np.asarray(rmses, dtype=float)
    maes = np.asarray(maes, dtype=float)
    tts = np.asarray(tts, dtype=float)
    return {
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
        "total_fit_time": float(np.sum(tts))
    }


def train_eval_holdout(Xtr, ytr, Xte, yte, cat_features, params):
    trp = Pool(Xtr, ytr, cat_features=cat_features)
    tep = Pool(Xte, yte, cat_features=cat_features)
    m = CatBoostRegressor(**params)
    t0 = time.time()
    m.fit(trp, verbose=False)
    dt = time.time() - t0
    pred = m.predict(tep)
    return {
        "rmse": rmse_np(yte, pred),
        "r2": r2_np(yte, pred),
        "mae": mae_np(yte, pred),
        "fit_time": float(dt)
    }


def html_table(df):
    return df.to_html(index=False, escape=False)


def make_html(out_path, meta, sections):
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
      .grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
    </style>
    """
    head = f"""
    <!doctype html>
    <html lang="ru">
    <head>
      <meta charset="utf-8">
      <title>Отчёт: Grid vs GA + чувствительность</title>
      {style}
    </head>
    <body>
      <h1>Подбор гиперпараметров и анализ чувствительности (CatBoostRegressor)</h1>
      <div class="muted">
        5 гиперпараметров: depth, learning_rate, l2_leaf_reg, random_strength, bagging_temperature.
        Grid: 3^5=243 (≤250). GA: ≤{meta["ga_budget"]}.
      </div>
      <div class="card">
        <h2>Протокол</h2>
        <div class="grid">
          <div>Holdout: <code>{meta["holdout"]}</code>, test_size=<code>{meta["test_size"]}</code>, CV: <code>{meta["cv_mode"]}</code>, folds=<code>{meta["cv_k"]}</code></div>
          <div>Размеры: n_total=<code>{meta["n_total"]}</code>, n_train=<code>{meta["n_train"]}</code>, n_test=<code>{meta["n_test"]}</code>, features=<code>{meta["n_features"]}</code>, cat=<code>{meta["n_cat"]}</code></div>
          <div>iterations=<code>{meta["iterations"]}</code>, task_type=<code>{meta["task_type"]}</code></div>
          <div>Grid CSV: <code>{meta["grid_csv"]}</code>, reuse=<code>{meta["reuse_grid"]}</code></div>
          <div>GA CSV: <code>{meta["ga_csv"]}</code>, reuse_if_exists=<code>{meta["reuse_ga"]}</code></div>
        </div>
      </div>
    """
    body = "\n".join(sections)
    tail = "</body></html>"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(head + body + tail)


def plot_best_so_far(values, title, xlab):
    best = np.inf
    ys = []
    for v in values:
        best = v if v < best else best
        ys.append(best)
    xs = np.arange(1, len(ys) + 1)
    plt.figure(figsize=(10, 4))
    plt.plot(xs, ys, marker="o", linestyle="-")
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel("RMSE (CV mean)")
    return b64fig()


def plot_scatter_time_rmse(df, title):
    plt.figure(figsize=(7, 5))
    plt.scatter(df["total_fit_time"].to_numpy(), df["rmse_mean"].to_numpy(), alpha=0.6)
    plt.title(title)
    plt.xlabel("суммарное время fit на CV, сек")
    plt.ylabel("RMSE (CV mean)")
    return b64fig()


def fmt_num(x, dec=4):
    try:
        v = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(v):
        return str(x)
    if abs(v - round(v)) < 1e-12:
        return str(int(round(v)))
    s = f"{v:.{int(dec)}f}"
    s = s.rstrip("0").rstrip(".")
    return s


def plot_param_freq_topN(df, param, N, title, decimals=4, max_bars=12):
    d = df.sort_values("rmse_mean").head(int(N)).copy()
    s = d[param]

    if pd.api.types.is_numeric_dtype(s):
        v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        v = np.round(v, int(decimals))
        s = pd.Series(v)
        s = s.where(np.isfinite(s), other=np.nan)
    else:
        s = s.astype(str)

    vc = s.value_counts(dropna=True)

    if len(vc) > int(max_bars):
        head = vc.iloc[:int(max_bars) - 1]
        other = int(vc.iloc[int(max_bars) - 1 :].sum())
        vc = pd.concat([head, pd.Series({"other": other})])

    idx = [fmt_num(x, dec=decimals) if x != "other" else "other" for x in vc.index.tolist()]
    vals = vc.to_numpy()

    plt.figure(figsize=(8, 4))
    plt.bar(idx, vals)
    plt.title(title)
    plt.xlabel(param)
    plt.ylabel(f"частота в top-{int(N)}")
    plt.xticks(rotation=25, ha="right")
    return b64fig()


def plot_sensitivity_local(best_row, grid_df, grid_values, title, decimals=4):
    base_rmse = float(best_row["rmse_mean"])
    fixed = {k: best_row[k] for k in grid_values.keys()}
    labels, deltas = [], []

    for hp, vals in grid_values.items():
        for v in vals:
            cond = np.ones(len(grid_df), dtype=bool)
            for k, fv in fixed.items():
                cond &= (grid_df[k] == (v if k == hp else fv))
            sub = grid_df.loc[cond]
            if len(sub) == 1:
                rmse = float(sub.iloc[0]["rmse_mean"])
                labels.append(f"{hp}={fmt_num(v, dec=decimals)}")
                deltas.append(rmse - base_rmse)

    if not deltas:
        plt.figure(figsize=(10, 4))
        plt.title(title)
        plt.text(0.5, 0.5, "Нет точек для локальной чувствительности", ha="center", va="center")
        plt.axis("off")
        return b64fig()

    order = np.argsort(np.abs(deltas))[::-1]
    labels = [labels[i] for i in order]
    deltas = [deltas[i] for i in order]

    plt.figure(figsize=(12, 5))
    plt.barh(np.arange(len(labels)), deltas)
    plt.yticks(np.arange(len(labels)), labels)
    plt.axvline(0.0)
    plt.title(title + f" (база RMSE={base_rmse:.5f})")
    plt.xlabel("ΔRMSE относительно лучшего GRID")
    plt.gca().invert_yaxis()
    return b64fig()


def plot_heatmap_depth_lr(best_row, grid_df, grid_values, title, decimals=4):
    fixed = {k: best_row[k] for k in grid_values.keys()}
    depths = list(grid_values["depth"])
    lrs = list(grid_values["learning_rate"])
    M = np.full((len(depths), len(lrs)), np.nan, dtype=float)

    for i, d in enumerate(depths):
        for j, lr in enumerate(lrs):
            cond = np.ones(len(grid_df), dtype=bool)
            for k, fv in fixed.items():
                if k == "depth":
                    cond &= (grid_df[k] == d)
                elif k == "learning_rate":
                    cond &= (grid_df[k] == lr)
                else:
                    cond &= (grid_df[k] == fv)
            sub = grid_df.loc[cond]
            if len(sub) == 1:
                M[i, j] = float(sub.iloc[0]["rmse_mean"])

    plt.figure(figsize=(7, 5))
    plt.imshow(M, aspect="auto", interpolation="nearest")
    plt.xticks(np.arange(len(lrs)), [fmt_num(x, dec=decimals) for x in lrs])
    plt.yticks(np.arange(len(depths)), [fmt_num(x, dec=decimals) for x in depths])
    plt.xlabel("learning_rate")
    plt.ylabel("depth")
    plt.title(title)
    plt.colorbar(fraction=0.03, pad=0.03)
    return b64fig()


def validate_grid_csv(df):
    need = ["depth", "learning_rate", "l2_leaf_reg", "random_strength", "bagging_temperature", "rmse_mean", "total_fit_time"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise RuntimeError(f"{GRID_CSV} не содержит нужных колонок: {miss}")


def grid_run(Xcv, ycv, cat_features, folds, base_params, grid_values, out_csv):
    keys = list(grid_values.keys())
    combos = list(itertools.product(*[grid_values[k] for k in keys]))
    rows = []
    for i, combo in enumerate(tqdm(combos, desc="GridSearch (3^5=243)", total=len(combos))):
        hp = dict(zip(keys, combo))
        params = dict(base_params)
        params.update(hp)
        ev = cv_evaluate(Xcv, ycv, cat_features, folds, params)
        row = {"method": "grid", "eval_id": int(i + 1)}
        row.update(hp)
        row.update(ev)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    return df


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def ga_run(Xcv, ycv, cat_features, folds, base_params, budget, seed, pop_size, generations, out_csv):
    rng = _rng(seed)
    bounds = {
        "depth": (6, 10),
        "learning_rate": (0.05, 0.20),
        "l2_leaf_reg": (1.0, 10.0),
        "random_strength": (0.0, 2.0),
        "bagging_temperature": (0.0, 2.0)
    }

    def sample_gene():
        d = int(rng.integers(bounds["depth"][0], bounds["depth"][1] + 1))
        lr = float(bounds["learning_rate"][0] * (bounds["learning_rate"][1] / bounds["learning_rate"][0]) ** rng.random())
        l2 = float(bounds["l2_leaf_reg"][0] * (bounds["l2_leaf_reg"][1] / bounds["l2_leaf_reg"][0]) ** rng.random())
        rs = float(rng.uniform(bounds["random_strength"][0], bounds["random_strength"][1]))
        bt = float(rng.uniform(bounds["bagging_temperature"][0], bounds["bagging_temperature"][1]))
        return (d, lr, l2, rs, bt)

    def gene_key(g):
        d, lr, l2, rs, bt = g
        return (int(d), round(float(lr), 6), round(float(l2), 6), round(float(rs), 6), round(float(bt), 6))

    def mutate(g):
        d, lr, l2, rs, bt = g
        if rng.random() < 0.6:
            d = int(clamp(d + int(rng.integers(-1, 2)), bounds["depth"][0], bounds["depth"][1]))
        if rng.random() < 0.6:
            lr = float(clamp(lr * float(np.exp(rng.normal(0.0, 0.15))), bounds["learning_rate"][0], bounds["learning_rate"][1]))
        if rng.random() < 0.6:
            l2 = float(clamp(l2 * float(np.exp(rng.normal(0.0, 0.25))), bounds["l2_leaf_reg"][0], bounds["l2_leaf_reg"][1]))
        if rng.random() < 0.6:
            rs = float(clamp(rs + float(rng.normal(0.0, 0.12)), bounds["random_strength"][0], bounds["random_strength"][1]))
        if rng.random() < 0.6:
            bt = float(clamp(bt + float(rng.normal(0.0, 0.12)), bounds["bagging_temperature"][0], bounds["bagging_temperature"][1]))
        return (int(d), float(lr), float(l2), float(rs), float(bt))

    def crossover(a, b):
        return tuple(a[i] if rng.random() < 0.5 else b[i] for i in range(5))

    def tournament(pop, scores, k=3):
        idx = rng.integers(0, len(pop), size=int(k))
        bi = int(idx[0])
        for j in idx[1:]:
            j = int(j)
            if scores[j] < scores[bi]:
                bi = j
        return pop[bi]

    elite = 2
    p_mut = 0.25
    cache = {}
    rows = []
    eval_count = 0

    def eval_gene(g, gen):
        nonlocal eval_count
        k = gene_key(g)
        if k in cache:
            return cache[k], False
        d, lr, l2, rs, bt = g
        hp = {
            "depth": int(d),
            "learning_rate": float(lr),
            "l2_leaf_reg": float(l2),
            "random_strength": float(rs),
            "bagging_temperature": float(bt)
        }
        params = dict(base_params)
        params.update(hp)
        ev = cv_evaluate(Xcv, ycv, cat_features, folds, params)
        eval_count += 1
        row = {"method": "ga", "eval_id": int(eval_count), "gen": int(gen)}
        row.update(hp)
        row.update(ev)
        rows.append(row)
        cache[k] = ev["rmse_mean"]
        return ev["rmse_mean"], True

    pop = [sample_gene() for _ in range(int(pop_size))]
    pbar = tqdm(total=int(budget), desc="GA search", leave=True)

    gen = 0
    while gen < int(generations) and eval_count < int(budget):
        scores = []
        for i in range(len(pop)):
            if eval_count >= int(budget):
                break
            s, is_new = eval_gene(pop[i], gen)
            if is_new:
                pbar.update(1)
            scores.append(float(s))
        if len(scores) < len(pop):
            break
        order = np.argsort(np.asarray(scores))
        pop = [pop[int(i)] for i in order]
        scores = [scores[int(i)] for i in order]
        next_pop = pop[:int(elite)]
        while len(next_pop) < int(pop_size) and eval_count < int(budget):
            p1 = tournament(pop, scores, k=3)
            p2 = tournament(pop, scores, k=3)
            child = crossover(p1, p2)
            if rng.random() < float(p_mut):
                child = mutate(child)
            next_pop.append(child)
        pop = next_pop
        gen += 1

    pbar.close()
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    return df


def run():
    X, y, cat_features, groups, ts = build_dataset(PATH_VID, PATH_CH)

    if HOLDOUT == "time":
        tr_idx, te_idx = holdout_time(ts, test_size=TEST_SIZE)
    else:
        tr_idx, te_idx = holdout_random(len(X), test_size=TEST_SIZE, seed=SEED)

    Xtr = X.iloc[tr_idx].reset_index(drop=True)
    ytr = y[tr_idx]
    gtr = groups[tr_idx]
    tstr = pd.to_datetime(ts.iloc[tr_idx].reset_index(drop=True), errors="coerce")

    Xte = X.iloc[te_idx].reset_index(drop=True)
    yte = y[te_idx]

    if CV_MODE == "time":
        order = np.argsort(tstr.fillna(pd.Timestamp.min).to_numpy(dtype="datetime64[ns]")).astype(int)
        Xcv = Xtr.iloc[order].reset_index(drop=True)
        ycv = ytr[order]
        folds = list(TimeSeriesSplit(n_splits=CV_K, test_size=TIME_TEST_SIZE, gap=TIME_GAP, max_train_size=TIME_MAX_TRAIN_SIZE).split(Xcv))
    else:
        Xcv = Xtr
        ycv = ytr
        folds = list(GroupKFold(n_splits=CV_K).split(Xcv, y=ycv, groups=gtr))

    if not folds:
        raise RuntimeError("CV folds пустые. Попробуй CV_MODE='time' или уменьши CV_K, либо проверь группы/время.")

    base_params = dict(
        iterations=int(ITERATIONS),
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=int(SEED),
        task_type=str(TASK_TYPE),
        allow_writing_files=False,
        bootstrap_type="Bayesian"
    )

    baseline_hp = dict(depth=8, learning_rate=0.10, l2_leaf_reg=3.0, random_strength=1.0, bagging_temperature=1.0)
    baseline_params = dict(base_params)
    baseline_params.update(baseline_hp)
    baseline_cv = cv_evaluate(Xcv, ycv, cat_features, folds, baseline_params)
    baseline_test = train_eval_holdout(Xtr, ytr, Xte, yte, cat_features, baseline_params)

    grid_values = {
        "depth": [7, 8, 9],
        "learning_rate": [0.08, 0.10, 0.12],
        "l2_leaf_reg": [2.0, 3.0, 4.0],
        "random_strength": [0.8, 1.0, 1.2],
        "bagging_temperature": [0.8, 1.0, 1.2]
    }

    if REUSE_GRID and os.path.exists(GRID_CSV):
        grid_df = pd.read_csv(GRID_CSV)
        validate_grid_csv(grid_df)
        if "eval_id" not in grid_df.columns:
            grid_df["eval_id"] = np.arange(1, len(grid_df) + 1)
        if "method" not in grid_df.columns:
            grid_df["method"] = "grid"
    else:
        grid_df = grid_run(Xcv, ycv, cat_features, folds, base_params, grid_values, GRID_CSV)

    best_grid = grid_df.sort_values("rmse_mean").iloc[0].to_dict()
    best_grid_params = dict(base_params)
    best_grid_params.update({k: best_grid[k] for k in grid_values.keys()})
    best_grid_test = train_eval_holdout(Xtr, ytr, Xte, yte, cat_features, best_grid_params)

    ga_df = None
    best_ga = None
    best_ga_test = None

    if RUN_GA:
        if REUSE_GA_IF_EXISTS and os.path.exists(GA_CSV):
            ga_df = pd.read_csv(GA_CSV)
            if "eval_id" not in ga_df.columns:
                ga_df["eval_id"] = np.arange(1, len(ga_df) + 1)
            if "method" not in ga_df.columns:
                ga_df["method"] = "ga"
        else:
            ga_df = ga_run(
                Xcv=Xcv, ycv=ycv, cat_features=cat_features, folds=folds,
                base_params=base_params, budget=int(GA_BUDGET), seed=int(SEED),
                pop_size=int(GA_POP), generations=int(GA_GENS), out_csv=GA_CSV
            )

        if ga_df is not None and len(ga_df):
            best_ga = ga_df.sort_values("rmse_mean").iloc[0].to_dict()
            best_ga_params = dict(base_params)
            best_ga_params.update({
                "depth": int(best_ga["depth"]),
                "learning_rate": float(best_ga["learning_rate"]),
                "l2_leaf_reg": float(best_ga["l2_leaf_reg"]),
                "random_strength": float(best_ga["random_strength"]),
                "bagging_temperature": float(best_ga["bagging_temperature"])
            })
            best_ga_test = train_eval_holdout(Xtr, ytr, Xte, yte, cat_features, best_ga_params)

    df_summary = pd.DataFrame([{
        "method": "baseline",
        "rmse_cv": baseline_cv["rmse_mean"],
        "rmse_cv_std": baseline_cv["rmse_std"],
        "cv_time_s": baseline_cv["total_fit_time"],
        "rmse_test": baseline_test["rmse"],
        "r2_test": baseline_test["r2"],
        "mae_test": baseline_test["mae"],
        "fit_time_train_s": baseline_test["fit_time"],
        **baseline_hp
    }, {
        "method": "grid_best",
        "rmse_cv": float(best_grid["rmse_mean"]),
        "rmse_cv_std": float(best_grid.get("rmse_std", np.nan)),
        "cv_time_s": float(best_grid["total_fit_time"]),
        "rmse_test": best_grid_test["rmse"],
        "r2_test": best_grid_test["r2"],
        "mae_test": best_grid_test["mae"],
        "fit_time_train_s": best_grid_test["fit_time"],
        **{k: best_grid[k] for k in grid_values.keys()}
    }])

    if best_ga is not None and best_ga_test is not None:
        df_summary = pd.concat([df_summary, pd.DataFrame([{
            "method": "ga_best",
            "rmse_cv": float(best_ga["rmse_mean"]),
            "rmse_cv_std": float(best_ga.get("rmse_std", np.nan)),
            "cv_time_s": float(best_ga["total_fit_time"]),
            "rmse_test": best_ga_test["rmse"],
            "r2_test": best_ga_test["r2"],
            "mae_test": best_ga_test["mae"],
            "fit_time_train_s": best_ga_test["fit_time"],
            "depth": int(best_ga["depth"]),
            "learning_rate": float(best_ga["learning_rate"]),
            "l2_leaf_reg": float(best_ga["l2_leaf_reg"]),
            "random_strength": float(best_ga["random_strength"]),
            "bagging_temperature": float(best_ga["bagging_temperature"])
        }])], ignore_index=True)

    grid_sorted = grid_df.sort_values("eval_id")
    img_grid_curve = plot_best_so_far(grid_sorted["rmse_mean"].to_numpy(), "Grid: лучший RMSE по мере перебора", "оценка (eval #)")
    img_grid_scatter = plot_scatter_time_rmse(grid_df, "Grid: RMSE vs CV-time")

    img_ga_curve = None
    img_ga_scatter = None
    ga_top10 = None
    if ga_df is not None and len(ga_df):
        ga_sorted = ga_df.sort_values("eval_id")
        img_ga_curve = plot_best_so_far(ga_sorted["rmse_mean"].to_numpy(), "GA: лучший RMSE по мере поиска", "оценка (eval #)")
        img_ga_scatter = plot_scatter_time_rmse(ga_df, "GA: RMSE vs CV-time")
        ga_top10 = ga_df.sort_values("rmse_mean").head(10)

    grid_top10 = grid_df.sort_values("rmse_mean").head(10)

    grid_freq_imgs = [plot_param_freq_topN(grid_df, hp, int(TOPN), f"Grid: {hp} в top-{int(TOPN)}") for hp in grid_values.keys()]
    ga_freq_imgs = []
    if ga_df is not None and len(ga_df):
        ga_freq_imgs = [plot_param_freq_topN(ga_df, hp, int(TOPN), f"GA: {hp} в top-{int(TOPN)}") for hp in grid_values.keys()]

    best_grid_row = pd.Series(best_grid)
    img_sens = plot_sensitivity_local(best_grid_row, grid_df, grid_values, "Локальная чувствительность вокруг лучшего GRID")
    img_heat = plot_heatmap_depth_lr(best_grid_row, grid_df, grid_values, "Heatmap RMSE: depth × learning_rate (остальные фиксированы)")

    meta = {
        "holdout": HOLDOUT,
        "test_size": float(TEST_SIZE),
        "cv_mode": CV_MODE,
        "cv_k": int(CV_K),
        "n_total": int(len(X)),
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "n_features": int(X.shape[1]),
        "n_cat": int(len(cat_features)),
        "iterations": int(ITERATIONS),
        "task_type": str(TASK_TYPE),
        "grid_csv": str(GRID_CSV),
        "reuse_grid": bool(REUSE_GRID),
        "ga_csv": str(GA_CSV),
        "reuse_ga": bool(REUSE_GA_IF_EXISTS),
        "ga_budget": int(GA_BUDGET)
    }

    sections = []

    sections.append(f"""
    <div class="card">
      <h2>Итоговое сравнение</h2>
      {html_table(df_summary)}
      <div class="muted">
        CV-качество: mean по фолдам на train. Test-качество: обучение на всём train и проверка на holdout test.
      </div>
    </div>
    """)

    ga_right = "" if img_ga_curve is None else f"""
    <div>
      <h3>GA best-so-far</h3>
      <img src="data:image/png;base64,{img_ga_curve}">
      <div class="muted">Количество оценок GA: <code>{0 if ga_df is None else len(ga_df)}</code> (на самом деле их 250, просто часть бралась из кэша)</div>
    </div>
    """

    sections.append(f"""
    <div class="card">
      <h2>Скорость нахождения лучшего решения</h2>
      <div class="grid2">
        <div>
          <h3>Grid best-so-far</h3>
          <img src="data:image/png;base64,{img_grid_curve}">
          <div class="muted">Количество оценок Grid: <code>{len(grid_df)}</code></div>
        </div>
        {ga_right}
      </div>
    </div>
    """)

    ga_scatter_html = "" if img_ga_scatter is None else f"""
    <div>
      <h3>GA: качество vs стоимость</h3>
      <img src="data:image/png;base64,{img_ga_scatter}">
    </div>
    """

    sections.append(f"""
    <div class="card">
      <h2>Качество vs стоимость (точки поиска)</h2>
      <div class="grid2">
        <div>
          <h3>Grid: качество vs стоимость</h3>
          <img src="data:image/png;base64,{img_grid_scatter}">
        </div>
        {ga_scatter_html}
      </div>
    </div>
    """)

    sections.append(f"""
    <div class="card">
      <h2>Top-10 гиперпараметров по RMSE (CV)</h2>
      <div class="grid2">
        <div>
          <h3>Grid top-10</h3>
          {html_table(grid_top10)}
        </div>
        <div>
          <h3>GA top-10</h3>
          {html_table(ga_top10) if ga_top10 is not None else "<div class='muted'>GA не запускался/не дал результатов</div>"}
        </div>
      </div>
    </div>
    """)

    grid_freq_html = "".join([f"<img src='data:image/png;base64,{img}'>" for img in grid_freq_imgs])
    ga_freq_html = "".join([f"<img src='data:image/png;base64,{img}'>" for img in ga_freq_imgs]) if ga_freq_imgs else "<div class='muted'>GA не запускался/не дал результатов</div>"

    sections.append(f"""
    <div class="card">
      <h2>Согласованность гиперпараметров (частоты в top-{int(TOPN)})</h2>
      <h3>Grid</h3>
      {grid_freq_html}
      <h3>GA</h3>
      {ga_freq_html}
    </div>
    """)

    sections.append(f"""
    <div class="card">
      <h2>Чувствительность (на основе GRID, без доп. обучений)</h2>
      <div class="grid2">
        <div>
          <h3>Локальная чувствительность вокруг лучшего GRID</h3>
          <img src="data:image/png;base64,{img_sens}">
        </div>
        <div>
          <h3>Парная чувствительность: depth × learning_rate</h3>
          <img src="data:image/png;base64,{img_heat}">
        </div>
      </div>
    </div>
    """)

    make_html(OUT_HTML, meta, sections)
    print(f"Saved HTML: {OUT_HTML}")
    print(f"Grid CSV: {GRID_CSV} (rows={len(grid_df)})")
    if ga_df is not None:
        print(f"GA CSV: {GA_CSV} (rows={len(ga_df)})")


if __name__ == "__main__":
    run()
