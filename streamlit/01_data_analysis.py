import argparse
import json
import numpy as np
import pandas as pd

from ml_core import (
    setup_logging, prepare_dataframe, add_derived_features,
    infer_feature_types, pick_default_feature_sets, compute_nonneg_numeric,
    most_frequent_values, ensure_dir
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--channels", required=False, default=None)
    ap.add_argument("--target", required=False, default="hits")
    ap.add_argument("--out_dir", required=False, default="reports")
    ap.add_argument("--max_cat_values", required=False, type=int, default=5000)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    logger = setup_logging(f"{args.out_dir}/analysis.log")

    df = prepare_dataframe(args.videos, args.channels)
    df = add_derived_features(df)

    logger.info(f"rows={len(df)} cols={len(df.columns)}")
    logger.info(f"columns_sample={df.columns[:30].tolist()}")

    if args.target not in df.columns:
        raise ValueError(f"target '{args.target}' not found in data")

    df = df.drop_duplicates(subset=["id"]) if "id" in df.columns else df
    logger.info(f"rows_after_dedup={len(df)}")

    miss = df.isna().mean().sort_values(ascending=False)
    top_miss = miss.head(30)
    logger.info("top_missing_share=" + json.dumps(top_miss.to_dict(), ensure_ascii=False))

    types = infer_feature_types(df, args.target)
    logger.info(f"bool_cols_count={len(types['bool'])} numeric_cols_count={len(types['numeric'])} categorical_cols_count={len(types['categorical'])}")

    fs = pick_default_feature_sets(df, args.target)
    for m in ["linear", "catboost", "nn"]:
        logger.info(f"{m}_features_bool={fs[m]['bool']}")
        logger.info(f"{m}_features_numeric={fs[m]['numeric']}")
        logger.info(f"{m}_features_categorical={fs[m]['categorical']}")

    all_numeric = sorted(list(set(fs["linear"]["numeric"] + fs["catboost"]["numeric"] + fs["nn"]["numeric"])))
    nonneg = compute_nonneg_numeric(df, all_numeric)
    logger.info(f"nonneg_numeric={nonneg}")

    cat_cols_all = sorted(list(set(fs["linear"]["categorical"] + fs["catboost"]["categorical"] + fs["nn"]["categorical"])))
    cat_options = {}
    for c in cat_cols_all:
        if c not in df.columns:
            continue
        cat_options[c] = most_frequent_values(df, c, args.max_cat_values)

    y = pd.to_numeric(df[args.target], errors="coerce").fillna(0.0).astype(float).values
    logger.info(f"target={args.target} y_min={float(np.min(y))} y_p50={float(np.median(y))} y_p95={float(np.quantile(y, 0.95))} y_mean={float(np.mean(y))}")

    schema = {
        "target": args.target,
        "feature_sets": fs,
        "feature_types_inferred": types,
        "nonneg_numeric": nonneg,
        "cat_options": cat_options
    }
    with open(f"{args.out_dir}/schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    logger.info(f"saved_schema={args.out_dir}/schema.json")


if __name__ == "__main__":
    main()
