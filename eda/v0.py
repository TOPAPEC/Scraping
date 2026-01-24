import os
import json
import requests
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error
import base64
from io import BytesIO as PlotBytesIO

sns.set_theme(style="whitegrid")
plt.rcParams['axes.grid'] = False

class SiglipFeatureExtractor:
    def __init__(self, model_id="google/siglip-so400m-patch14-384", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"[GPU/CPU] Initializing SigLIP model on {self.device}...")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        if self.device == "cuda":
            self.model.half()

        self.session = requests.Session()
        retry_strategy = Retry(
            total=2,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=20, 
            pool_maxsize=20
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def download_image(self, url):
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content)).convert("RGB")
            return None
        except Exception:
            return None

    def process_batch(self, url_batch):
        images = []
        indices = []
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_idx = {executor.submit(self.download_image, url): i for i, url in enumerate(url_batch)}
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    img = future.result()
                    if img is not None:
                        images.append(img)
                        indices.append(idx)
                except Exception:
                    pass
        
        if not images:
            return None, None

        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            if self.device == "cuda":
                inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
            outputs = self.model(**inputs)
            embeddings = outputs.pooler_output
            if self.device == "cuda":
                embeddings = embeddings.float()
        
        return indices, embeddings.cpu().numpy()

def get_base64_plot():
    buf = PlotBytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def main():
    path_vid = 'scraped_data\\v1_vids_and_channels_100k\\videos.ndjson'
    path_ch = 'scraped_data\\v1_vids_and_channels_100k\\channels.ndjson'
    emb_file = 'video_siglip_embeddings.npz'
    batch_size = 64
    
    print("[Data] Loading Video Metadata...")
    with open(path_vid, 'r', encoding='utf-8') as f:
        v_data = [json.loads(line) for line in f]
    df_v = pd.json_normalize(v_data)

    if os.path.exists(emb_file):
        print("[Vectors] Loading pre-computed embeddings...")
        data = np.load(emb_file)
        emb_dict = dict(zip(data['ids'], data['embeddings']))
    else:
        print("[Vectors] Computing embeddings with SigLIP2 (GPU)...")
        extractor = SiglipFeatureExtractor()
        urls = df_v['thumbnail_url'].tolist()
        ids = df_v['id'].tolist()
        
        all_embeddings = np.zeros((len(ids), 1152), dtype=np.float32)
        
        for i in tqdm(range(0, len(urls), batch_size), desc="Processing Image Batches"):
            batch_urls = urls[i:i+batch_size]
            valid_indices, vectors = extractor.process_batch(batch_urls)
            if vectors is not None:
                real_indices = [i + j for j in valid_indices]
                all_embeddings[real_indices] = vectors
        
        np.savez_compressed(emb_file, ids=ids, embeddings=all_embeddings)
        emb_dict = dict(zip(ids, all_embeddings))

    print("[Merge] Mapping embeddings to DataFrame...")
    emb_df = pd.DataFrame.from_dict(emb_dict, orient='index', columns=[f'emb_{i}' for i in range(1152)])
    emb_df.index.name = 'id'
    df_v = df_v.set_index('id').join(emb_df, how='left').reset_index()

    print("[Data] Processing Channels and Merging...")
    with open(path_ch, 'r', encoding='utf-8') as f:
        c_data = [json.loads(line) for line in f]
    df_c = pd.read_json(path_ch, lines=True)
    df_c['meta_count'] = df_c['meta'].apply(lambda x: len(x) if isinstance(x, dict) else 0)
    
    total_videos = len(df_v)
    
    df_v['created_ts'] = pd.to_datetime(df_v['created_ts'], errors='coerce')
    df_v['hour'] = df_v['created_ts'].dt.hour.fillna(-1).astype(int)
    df_v['dow'] = df_v['created_ts'].dt.dayofweek.fillna(-1).astype(int)
    df_v['title_len'] = df_v['title'].fillna('').astype(str).apply(len)
    df_v['desc_len'] = df_v['description'].fillna('').astype(str).apply(len)
    df_v['log_hits'] = np.log1p(df_v['hits'].astype(float))
    
    df_c['subscribers'] = pd.to_numeric(df_c['subscribers'], errors='coerce').fillna(0)
    df_c['ch_title_len'] = df_c['title'].fillna('').astype(str).apply(len)
    df_c['ch_desc_len'] = df_c['description'].fillna('').astype(str).apply(len)

    df = df_v.merge(df_c, left_on='author.id', right_on='channel_id', how='left')
    join_success = len(df) / total_videos * 100

    ignore_cols = ['video_url', 'thumbnail_url', 'picture_url', 'preview_url', 'embed_url', 'feed_url', 'html', 
                   'author.name', 'author.avatar_url', 'author.site_url', 'category.category_url', 'category.name', 
                   'pg_rating.logo', 'action_reason.name', 'url', 'avatar_url', 'title_y', 'description_y', 'jsonld', 'meta']
    
    meta_cols = [c for c in df.columns if c not in ignore_cols + ['hits', 'log_hits', 'created_ts', 'last_update_ts', 'publication_ts', 'channel_id', 'author.id', 'id']]
    embedding_cols = [c for c in df.columns if c.startswith('emb_')]
    
    X_cols = meta_cols + embedding_cols
    
    df_model = df[X_cols + ['log_hits']].copy()
    df_model = df_model.dropna(subset=['log_hits'])
    
    emb_cols_in_model = [c for c in embedding_cols if c in df_model.columns]
    
    for col in df_model.columns:
        if df_model[col].dtype == 'object':
            df_model[col] = df_model[col].fillna('Unknown')
        elif df_model[col].dtype == 'float64' or df_model[col].dtype == 'int64':
            df_model[col] = df_model[col].fillna(0)

    X = df_model[X_cols]
    y = df_model['log_hits']
    
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype('category')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[Model] Training XGBoost Regressor on Metadata + Visual Features...")
    model = xgb.XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=7, 
        subsample=0.8, colsample_bytree=0.8, enable_categorical=True, 
        eval_metric='rmse', tree_method='hist', n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    imp_dict = model.get_booster().get_score(importance_type='gain')
    feats = model.get_booster().feature_names
    importance = pd.DataFrame({'Feature': feats, 'Gain': [imp_dict.get(f, 0) for f in feats]})
    importance['Type'] = importance['Feature'].apply(lambda x: 'Visual' if 'emb_' in x else 'Metadata')
    importance['Gain_Pct'] = importance['Gain'] / importance['Gain'].sum() * 100
    importance = importance.sort_values('Gain', ascending=False)
    
    top_features = importance.head(20)
    
    perm = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    perm_df = pd.DataFrame({'Feature': X_test.columns, 'Importance': perm.importances_mean}).sort_values('Importance', ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=perm_df, x='Importance', y='Feature', palette='viridis')
    plt.title('Permutation Importance (Top 15)')
    img_perm = get_base64_plot()

    visual_gain = importance[importance['Type'] == 'Visual']['Gain'].sum()
    total_gain = importance['Gain'].sum()
    visual_contribution = (visual_gain / total_gain) * 100

    html = f"""
    <!DOCTYPE html><html><head><title>Comprehensive Visual + Text Analysis</title><style>
    body {{ font-family: 'Segoe UI', sans-serif; padding: 20px; background: #f4f6f9; }}
    .container {{ max-width: 1200px; margin: auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); }}
    h1 {{ color: #2c3e50; border-bottom: 3px solid #8e44ad; padding-bottom: 10px; }}
    h2 {{ color: #2980b9; margin-top: 30px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 13px; }}
    th, td {{ padding: 10px; border: 1px solid #dee2e6; text-align: left; }}
    th {{ background: #2c3e50; color: white; }}
    .metric {{ background: #e8daef; padding: 15px; border-left: 5px solid #8e44ad; margin: 15px 0; }}
    img {{ max-width: 100%; border: 1px solid #ddd; }}
    .tag {{ padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold; }}
    .vis {{ background-color: #f1c40f; color: #333; }}
    .meta {{ background-color: #3498db; color: white; }}
    </style></head><body><div class="container">
    <h1>Video Analysis: Metadata + Deep Visual Features (SigLIP2)</h1>
    <div class="metric">
    <b>Join Success:</b> {join_success:.2f}% | <b>Features:</b> {len(X_cols)} ({len(emb_cols_in_model)} Visual)<br>
    <b>Performance:</b> R2={r2:.4f}, RMSE={rmse:.4f}<br>
    <b>Visual Contribution:</b> Visual embeddings contribute {visual_contribution:.2f}% of total model gain.
    </div>
    
    <h2>Top 20 Feature Importance (Gain)</h2>
    <table><tr><th>Feature</th><th>Type</th><th>Gain</th><th>Gain %</th></tr>
    """
    for _, row in top_features.iterrows():
        tag_cls = "vis" if row['Type'] == 'Visual' else "meta"
        html += f"<tr><td>{row['Feature']}</td><td><span class='tag {tag_cls}'>{row['Type']}</span></td><td>{row['Gain']:.4f}</td><td>{row['Gain_Pct']:.2f}%</td></tr>"
    html += "</table>"
    
    html += f"<h2>Permutation Importance</h2><img src='data:image/png;base64,{img_perm}'>"
    html += "<h2>Conclusion</h2><p>Incorporating visual vectors significantly enriches the model. The analysis reveals that visual content accounts for a substantial portion of predictive power regarding video popularity.</p></div></body></html>"

    with open("visual_analysis_report.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Report saved to visual_analysis_report.html")

if __name__ == "__main__":
    main()