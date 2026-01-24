import json
import os
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("ML_API_URL", "http://localhost:8000")
MODELS_DIR = os.getenv("MODELS_DIR", "models")


def api_get(path: str):
    r = requests.get(f"{API_URL}{path}", timeout=60)
    if r.status_code != 200:
        raise RuntimeError(r.text)
    return r.json()


def api_post(path: str, payload: dict):
    r = requests.post(f"{API_URL}{path}", json=payload, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(r.text)
    return r.json()


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


def main():
    st.set_page_config(page_title="ML Models Playground", layout="wide")
    st.title("Интерактивное ML-приложение: выбор модели, проверка данных, метрики, важности")

    schema = load_schema()
    holdout = load_holdout()
    if schema is None or holdout is None:
        st.error("Нет models/schema_used.json или models/holdout.parquet. Сначала запусти 01_data_analysis.py и 02_train_models.py.")
        return

    target = schema["target"]
    models = ["linear", "catboost", "nn"]

    cols = st.columns([2, 2, 2, 3])
    with cols[0]:
        model_name = st.selectbox("Модель", models, index=1)
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
            if not custom_on_holdout:
                res = api_get("/holdout_metrics")
                st.json(res.get(model_name, res))
            else:
                dfh = holdout.sample(n=int(holdout_n), random_state=42) if int(holdout_n) < len(holdout) else holdout
                y_true = pd.to_numeric(dfh.get(target, 0.0), errors="coerce").fillna(0.0).astype(float).tolist()
                rec = dfh.reindex(columns=list(set(dfh.columns) | set(features))).copy()
                records = rec[features].to_dict(orient="records")
                payload = {"records": records, "scale_numeric": bool(scale_numeric), "y_true": y_true, "custom_metric": custom_expr_holdout}
                out = api_post(f"/predict/{model_name}", payload)
                st.write("Base metrics")
                st.json(out.get("metrics"))
                st.write("Custom metric")
                st.json(out.get("custom_metric"))
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
        y_true = pd.to_numeric(df_new[target], errors="coerce").fillna(0.0).astype(float).tolist() if has_y else None

        rec = df_new.reindex(columns=list(set(df_new.columns) | set(features))).copy()
        records = rec[features].to_dict(orient="records")

        custom_expr = st.text_input("Кастомная метрика (numpy-выражение). Доступно: np, y_true, y_pred", value="np.mean(np.abs(y_true - y_pred))")
        if st.button("Predict + метрики на загруженных данных"):
            try:
                payload = {"records": records, "scale_numeric": bool(scale_numeric), "y_true": y_true, "custom_metric": (custom_expr if has_y else None)}
                out = api_post(f"/predict/{model_name}", payload)
                st.write("Предсказания (первые 50)")
                st.dataframe(pd.DataFrame({"y_pred": out["y_pred"]}).head(50), use_container_width=True)
                if out.get("metrics") is not None:
                    st.write("Базовые метрики")
                    st.json(out["metrics"])
                if out.get("custom_metric") is not None:
                    st.write("Кастомная метрика")
                    st.json(out["custom_metric"])
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
            yt = None
            if y_true_input is not None and y_true_input.strip():
                parts = [p.strip() for p in y_true_input.replace("\n", ",").split(",") if p.strip()]
                yt = [float(x) for x in parts]
            payload = {"records": records, "scale_numeric": bool(scale_numeric), "y_true": yt, "custom_metric": (custom_expr2 if yt is not None else None)}
            out = api_post(f"/predict/{model_name}", payload)
            st.dataframe(pd.DataFrame({"y_pred": out["y_pred"]}), use_container_width=True)
            if out.get("metrics") is not None:
                st.json(out["metrics"])
            if out.get("custom_metric") is not None:
                st.json(out["custom_metric"])
        except Exception as e:
            st.error(str(e))

    st.subheader("Feature importance")
    if st.button("Показать важности (встроенная + permutation где доступно)"):
        try:
            imp = api_get(f"/importance/{model_name}?metric={metric_for_importance}&n_rows={int(n_rows_pi)}&seed=42")
            st.json(imp)
        except Exception as e:
            st.error(str(e))


if __name__ == "__main__":
    main()
