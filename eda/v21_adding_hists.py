import argparse
import json
import base64
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FEATURES_4X4 = [
    "log_ch_subscribers", "log_duration", "ch_meta_count", "origin_type",
    "is_official", "is_reborn_channel", "is_licensed", "month",
    "ch_desc_len", "pg_rating.age", "ch_title_len", "is_livestream",
    "log_desc_len", "is_adult", "author.is_allowed_offline", "is_on_air"
]

def load_ndjson(path, limit=0):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def fig_to_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode("utf-8")

def coerce_numeric(s):
    x = pd.to_numeric(s, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    return x

def build_features(videos_ndjson, channels_ndjson):
    df_v = pd.json_normalize(videos_ndjson)
    df_c = pd.json_normalize(channels_ndjson)

    if "created_ts" in df_v.columns:
        dt = pd.to_datetime(df_v["created_ts"], errors="coerce", utc=False)
        df_v["month"] = dt.dt.month.fillna(-1).astype(int)
    else:
        df_v["month"] = -1

    if "description" in df_v.columns:
        desc = df_v["description"].fillna("").astype(str)
        df_v["desc_len"] = desc.str.len()
    else:
        df_v["desc_len"] = 0

    if "duration" in df_v.columns:
        dur = coerce_numeric(df_v["duration"]).fillna(0.0)
        df_v["log_duration"] = np.log1p(dur)
    else:
        df_v["log_duration"] = 0.0

    df_v["log_desc_len"] = np.log1p(coerce_numeric(df_v.get("desc_len", 0)).fillna(0.0))

    df_c = df_c.rename(columns={c: f"ch_{c}" for c in df_c.columns if c != "channel_id"})

    ch_sub = coerce_numeric(df_c.get("ch_subscribers", 0)).fillna(0.0)
    df_c["log_ch_subscribers"] = np.log1p(ch_sub)

    df_c["ch_title_len"] = df_c.get("ch_title", "").fillna("").astype(str).str.len()
    df_c["ch_desc_len"] = df_c.get("ch_description", "").fillna("").astype(str).str.len()

    ch_meta_cols = [c for c in df_c.columns if c.startswith("ch_meta.")]
    df_c["ch_meta_count"] = df_c[ch_meta_cols].notna().sum(axis=1) if ch_meta_cols else 0

    left_key = "author.id"
    right_key = "channel_id"
    if left_key not in df_v.columns or right_key not in df_c.columns:
        df = df_v.copy()
        for c in ["log_ch_subscribers", "ch_meta_count", "ch_desc_len", "ch_title_len"]:
            if c not in df.columns:
                df[c] = np.nan
        return df

    df = df_v.merge(df_c, left_on=left_key, right_on=right_key, how="left")
    return df

def plot_4x4_frequency_grid(df, features=FEATURES_4X4, out_png="eda_feature_hists_4x4.png", title="Feature frequency histograms (4x4)", bins=50):
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.ravel()

    for i, feat in enumerate(features):
        ax = axes[i]
        if feat not in df.columns:
            ax.set_axis_off()
            continue

        s = df[feat]

        if s.dtype == bool:
            vc = s.fillna(False).astype(int).value_counts().sort_index()
            ax.bar(vc.index.astype(str), vc.values)
            ax.set_xticklabels([str(x) for x in vc.index], rotation=0)
        else:
            x = coerce_numeric(s)
            nonna_ratio = float(x.notna().mean()) if len(x) else 0.0
            if nonna_ratio >= 0.8:
                x = x.fillna(0.0)
                nun = int(x.nunique(dropna=True))
                if nun <= 20:
                    vc = x.astype(int).value_counts().sort_index()
                    ax.bar(vc.index.astype(str), vc.values)
                    ax.set_xticklabels([str(v) for v in vc.index], rotation=0)
                else:
                    vals = x.values
                    ax.hist(vals, bins=bins)
            else:
                vc = s.astype(str).fillna("Unknown").value_counts().head(30)
                ax.barh(vc.index[::-1], vc.values[::-1])

        ax.set_title(feat)
        ax.grid(False)

    for j in range(len(features), 16):
        axes[j].set_axis_off()

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    b64 = fig_to_b64(fig)
    return b64

def build_div_html(img_b64, header="Feature Frequency Histograms (4x4)"):
    return f"""
<div class="stats-box">
  <h3>{header}</h3>
  <img src="data:image/png;base64,{img_b64}" alt="Feature frequency histograms 4x4" style="max-width:100%;border:1px solid #ddd;border-radius:4px;margin-top:10px;">
</div>
""".strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=r"scraped_data\v1_vids_and_channels_100k\videos.ndjson")
    ap.add_argument("--channels", default=r"scraped_data\v1_vids_and_channels_100k\channels.ndjson")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out_png", default="eda_feature_hists_4x4.png")
    ap.add_argument("--out_html", default="eda_feature_hists_div.html")
    ap.add_argument("--bins", type=int, default=50)
    args = ap.parse_args()

    v = load_ndjson(args.videos, limit=args.limit)
    c = load_ndjson(args.channels, limit=args.limit)
    df = build_features(v, c)

    for feat in FEATURES_4X4:
        if feat not in df.columns:
            df[feat] = np.nan

    img_b64 = plot_4x4_frequency_grid(df, out_png=args.out_png, bins=args.bins)
    div = build_div_html(img_b64)

    with open(args.out_html, "w", encoding="utf-8") as f:
        f.write(div)

    print(div)

if __name__ == "__main__":
    main()
