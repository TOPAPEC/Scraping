import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import json
import base64
from io import BytesIO
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import shapiro, mannwhitneyu
import warnings

warnings.filterwarnings('ignore')
sns.set(style="whitegrid")
plt.rcParams['axes.grid'] = False

def get_base64_plot():
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.read()).decode('utf-8')

def validate_and_fix_dtypes_hash(X, verbose=True):
    print("\n=== VALIDATING AND ENCODING DATA (HASHING STRATEGY) ===")
    for col in X.columns:
        is_numeric = pd.api.types.is_numeric_dtype(X[col].dtype)
        
        if is_numeric:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            if verbose:
                print(f"Column: {col:<30} | Numeric -> Float")
        else:
            codes, uniques = pd.factorize(X[col].astype(str))
            X[col] = codes
            if verbose:
                print(f"Column: {col:<30} | String -> Integer Codes (Unique: {len(uniques)})")
                
    print("=== VALIDATION COMPLETE ===\n")
    return X

def perform_statistical_analysis(df, target_col):
    print("\n[Step 4a] Running Full Statistical Analysis (EDA)...")
    
    results = {}
    
    sample_target = df[target_col].sample(n=min(5000, len(df)), random_state=42)
    stat, p = shapiro(sample_target)
    is_normal = p > 0.05
    results['normality'] = {
        'statistic': stat, 'p_value': p, 'is_normal': is_normal,
        'interpretation': 'Data looks Gaussian (fail to reject H0)' if is_normal else 'Data does not look Gaussian (reject H0)'
    }
    print(f"Normality Test (Shapiro-Wilk): p={p:.4f} -> {results['normality']['interpretation']}")

    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr(method='spearman')
    
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False, 
                xticklabels=False, yticklabels=False, cbar=True)
    plt.title(f'Spearman Correlation Matrix (All {len(numeric_df.columns)} Features)')
    plt.tight_layout()
    img_corr = get_base64_plot()
    
    median_target = df[target_col].median()
    df['target_class'] = (df[target_col] > median_target).astype(int)
    
    mwu_results = []
    features_to_test = [c for c in df.columns if c not in [target_col, 'target_class']]
    
    for col in features_to_test:
        temp_col = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        group_high = temp_col[df['target_class'] == 1]
        group_low = temp_col[df['target_class'] == 0]
        
        if len(group_high) == 0 or len(group_low) == 0:
            continue
            
        try:
            stat, p_val = mannwhitneyu(group_high, group_low, alternative='two-sided')
            
            n1, n2 = len(group_high), len(group_low)
            rank_biserial = 1 - (2 * stat) / (n1 * n2)
            
            mwu_results.append({
                'Feature': col, 
                'Statistic': stat, 
                'P_Value': p_val,
                'Effect_Size': rank_biserial
            })
        except:
            pass

    mwu_df = pd.DataFrame(mwu_results)
    mwu_df['Abs_Effect_Size'] = mwu_df['Effect_Size'].abs()
    mwu_df = mwu_df.sort_values('Abs_Effect_Size', ascending=False)
    
    return results, img_corr, mwu_df

def main():
    path_vid = 'scraped_data\\v1_vids_and_channels_100k\\videos.ndjson'
    path_ch = 'scraped_data\\v1_vids_and_channels_100k\\channels.ndjson'

    print("[Step 1/5] Loading and normalizing metadata...")
    with open(path_vid, 'r', encoding='utf-8') as f:
        v_data = [json.loads(line) for line in f]
    df_v = pd.json_normalize(v_data)

    with open(path_ch, 'r', encoding='utf-8') as f:
        c_data = [json.loads(line) for line in f]
    df_c = pd.json_normalize(c_data)

    print("[Step 2/5] Renaming channel columns and engineering features...")
    total_videos = len(df_v)
    
    df_v['created_ts'] = pd.to_datetime(df_v['created_ts'], errors='coerce')
    df_v['hour'] = df_v['created_ts'].dt.hour.fillna(-1).astype(int)
    df_v['dow'] = df_v['created_ts'].dt.dayofweek.fillna(-1).astype(int)
    df_v['month'] = df_v['created_ts'].dt.month.fillna(-1).astype(int)
    df_v['title_len'] = df_v['title'].fillna('').astype(str).apply(len)
    df_v['desc_len'] = df_v['description'].fillna('').astype(str).apply(len)
    df_v['log_hits'] = np.log1p(df_v['hits'].astype(float))
    
    df_c = df_c.rename(columns={c: f"ch_{c}" for c in df_c.columns if c != 'channel_id'})
    df_c['ch_subscribers'] = pd.to_numeric(df_c['ch_subscribers'], errors='coerce').fillna(0)
    df_c['ch_title_len'] = df_c['ch_title'].fillna('').astype(str).apply(len)
    df_c['ch_desc_len'] = df_c['ch_description'].fillna('').astype(str).apply(len)
    
    ch_meta_cols = [c for c in df_c.columns if c.startswith('ch_meta.')]
    if ch_meta_cols:
        df_c['ch_meta_count'] = df_c[ch_meta_cols].notna().sum(axis=1)
    else:
        df_c['ch_meta_count'] = 0

    print("[Step 3/5] Merging datasets...")
    join_key_video = 'author.id'
    join_key_channel = 'channel_id'
    df = df_v.merge(df_c, left_on=join_key_video, right_on=join_key_channel, how='left')
    join_success_rate = len(df) / total_videos * 100

    exclude_cols = [
        'id', 'track_id', 'author.id', 'channel_id', 'category.id', 'action_reason.id',
        'title', 'description', 'feed_name', 
        'author.name', 'author.avatar_url', 'author.site_url',
        'category.name', 'category.category_url', 'pg_rating.logo', 'action_reason.name',
        'video_url', 'thumbnail_url', 'picture_url', 'preview_url', 'embed_url', 'feed_url', 'html', 
        'ch_title', 'ch_description', 'ch_avatar_url', 'ch_url',
        'common_subscription_product_codes', 'ch_jsonld', 'ch_meta',
        'created_ts', 'last_update_ts', 'publication_ts', 'future_publication', 
        'stream_type', 'product_id'
    ]
    exclude_cols.extend([c for c in df.columns if c.startswith('ch_meta.')])

    X_cols = [c for c in df.columns if c not in exclude_cols + ['hits', 'log_hits']]
    df_model = df[X_cols + ['log_hits']].copy()
    df_model = df_model.dropna(subset=['log_hits'])
    
    print("Cleaning data...")
    for col in df_model.columns:
        if col in ['ch_subscribers']:
            df_model[col] = pd.to_numeric(df_model[col], errors='coerce').fillna(0)
        elif pd.api.types.is_numeric_dtype(df_model[col]):
            df_model[col] = df_model[col].fillna(0)
        else:
            df_model[col] = df_model[col].fillna('Unknown')

    stats_results, img_corr, mwu_df = perform_statistical_analysis(df_model, 'log_hits')

    X = df_model[X_cols]
    y = df_model['log_hits']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data Split: Train={len(X_train)}, Test={len(X_test)}")

    print("[Step 4b/5] Validating and Training XGBoost...")
    
    X_train = validate_and_fix_dtypes_hash(X_train.copy(), verbose=True)
    X_test = validate_and_fix_dtypes_hash(X_test.copy(), verbose=False)
    
    model = xgb.XGBRegressor(
        n_estimators=400, 
        learning_rate=0.05, 
        max_depth=7, 
        subsample=0.8, 
        colsample_bytree=0.8, 
        eval_metric='rmse', 
        tree_method='hist', 
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)

    perm = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    perm_df = pd.DataFrame({'Feature': X_test.columns, 'Importance': perm.importances_mean, 'Std': perm.importances_std})
    perm_df = perm_df.sort_values('Importance', ascending=False).head(20)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=perm_df, x='Importance', y='Feature', palette='viridis')
    plt.title('Permutation Importance (Global)')
    img_perm = get_base64_plot()
    
    imp_dict = model.get_booster().get_score(importance_type='gain')
    if imp_dict is None:
        imp_dict = {}
        
    feats = model.get_booster().feature_names
    if feats is None:
        feats = X_test.columns.tolist()
        
    importance = pd.DataFrame({'Feature': feats, 'Gain': [imp_dict.get(f, 0) for f in feats]})
    importance = importance.sort_values('Gain', ascending=False).head(20)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance, x='Gain', y='Feature', palette='magma')
    plt.title('XGBoost Feature Importance (Gain)')
    img_xgb_gain = get_base64_plot()
    
    print("Calculating SHAP values for local interpretation...")
    explainer = shap.TreeExplainer(model)
    sample_idx = np.random.choice(len(X_test), size=min(200, len(X_test)), replace=False)
    X_test_sample = X_test.iloc[sample_idx]
    
    shap_values = explainer.shap_values(X_test_sample)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_sample, show=False)
    plt.title("SHAP Summary (Feature Impact on Model Output)")
    img_shap_summary = get_base64_plot()
    
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                         base_values=explainer.expected_value, 
                                         data=X_test_sample.iloc[0], 
                                         feature_names=X_test.columns.tolist()),
                                        show=False)
    plt.title("SHAP Waterfall Plot (Local Explanation for 1 Sample)")
    img_shap_local = get_base64_plot()

    print("Generating report...")
    
    mwu_html = ""
    for _, row in mwu_df.head(50).iterrows():
        color = "green" if abs(row['Effect_Size']) > 0.1 else "orange"
        mwu_html += f"<tr style='color:{color}'><td>{row['Feature']}</td><td>{row['P_Value']:.2e}</td><td>{row['Effect_Size']:.4f}</td></tr>"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Advanced Statistical Analysis & Model Interpretation</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; padding: 20px; background-color: #f4f7f6; color: #333; }}
            .container {{ max-width: 1200px; margin: auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #2980b9; margin-top: 30px; }}
            .metric {{ background: #eaf2f8; padding: 15px; border-left: 5px solid #3498db; margin: 15px 0; border-radius: 4px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 12px; }}
            th, td {{ padding: 10px; border: 1px solid #ddd; text-align: left; }}
            th {{ background-color: #34495e; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; margin-top: 10px; }}
            .stats-box {{ background: #fff8e1; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .metrics-table td {{ font-weight: bold; }}
            .highlight {{ font-weight: bold; color: #d35400; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Rutube Analysis: Statistical Tests & Model Interpretation</h1>
            
            <div class="metric">
                <b>Join Key Used:</b> Video['{join_key_video}'] == Channel['{join_key_channel}']<br>
                <b>Dataset Join Success:</b> {join_success_rate:.2f}%<br>
                <b>Target Normality (Shapiro-Wilk):</b> p-value = {stats_results['normality']['p_value']:.4f} ({stats_results['normality']['interpretation']})
            </div>

            <h2>Model Metrics (Train vs Test)</h2>
            <table class="metrics-table">
                <tr><th>Dataset</th><th>R2 Score</th><th>RMSE</th><th>MAE</th></tr>
                <tr><td>Train</td><td>{train_r2:.4f}</td><td>{train_rmse:.4f}</td><td>{train_mae:.4f}</td></tr>
                <tr><td>Test</td><td>{test_r2:.4f}</td><td>{test_rmse:.4f}</td><td>{test_mae:.4f}</td></tr>
            </table>

            <h2>1. Statistical Analysis (EDA)</h2>
            <div class="stats-box">
                <h3>Mann-Whitney U Test (Significance & Effect Size)</h3>
                <p>Tests if feature distributions differ between "High Hits" and "Low Hits" videos. 
                Due to large sample size, p-values are effectively 0. We use <b>Rank-Biserial Correlation (Effect Size)</b> to judge practical importance. 
                |Effect| > 0.1 is considered meaningful.</p>
                <table>
                    <tr><th>Feature</th><th>P-Value</th><th>Effect Size (Rank-Biserial)</th></tr>
                    {mwu_html}
                </table>
            </div>
            <h3>Spearman Correlation Matrix (All Features)</h3>
            <img src='data:image/png;base64,{img_corr}' alt='Correlation Matrix'>

            <h2>2. Global Interpretation</h2>
            
            <h3>XGBoost Feature Importance (Gain)</h3>
            <p>Internal metric: Average gain of splits using the feature.</p>
            <img src='data:image/png;base64,{img_xgb_gain}' alt='XGBoost Gain Importance'>
            
            <h3>Permutation Importance (Perturbation)</h3>
            <p>Drop in model accuracy when feature values are randomly shuffled.</p>
            <img src='data:image/png;base64,{img_perm}' alt='Permutation Importance'>

            <h2>3. Local Interpretation (SHAP)</h2>
            <p>SHAP values show how each feature contributes to pushing the prediction away from the average.</p>
            
            <h3>SHAP Summary (Global View of Local Effects)</h3>
            <img src='data:image/png;base64,{img_shap_summary}' alt='SHAP Summary'>
            
            <h3>SHAP Waterfall (Single Instance Explanation)</h3>
            <img src='data:image/png;base64,{img_shap_local}' alt='SHAP Waterfall'>
            
        </div>
    </body>
    </html>
    """

    with open("metadata_analysis_statistical.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Report saved to metadata_analysis_statistical.html")

if __name__ == "__main__":
    main()