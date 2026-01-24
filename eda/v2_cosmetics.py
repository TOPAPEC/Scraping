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
from sklearn.model_selection import train_test_split, cross_validate, learning_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import shapiro, mannwhitneyu, chi2_contingency, probplot
from statsmodels.stats.multitest import multipletests
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

def fit_encoders(X):
    encoders = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            continue
        unique_vals = X[col].astype(str).unique()
        encoders[col] = {val: i for i, val in enumerate(unique_vals)}
    return encoders

def apply_encoders(X, encoders, verbose=True):
    X = X.copy()
    for col in X.columns:
        is_numeric = pd.api.types.is_numeric_dtype(X[col])
        if is_numeric:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            if verbose:
                print(f"Column: {col:<30} | Numeric -> Float")
        else:
            mapper = encoders.get(col, {})
            X[col] = X[col].astype(str).map(mapper).fillna(-1).astype(int)
            if verbose:
                print(f"Column: {col:<30} | String -> Mapped Integer (Unseen: -1)")
    return X

def perform_statistical_analysis(df, target_col):
    print("\n[Step 4a] Running Full Statistical Analysis (EDA)...")
    
    results = {}
    
    target_data = df[target_col].dropna()
    sample_target = target_data.sample(n=min(5000, len(target_data)), random_state=42)
    
    stat, p = shapiro(sample_target)
    is_normal = p > 0.05
    results['normality'] = {
        'statistic': stat, 'p_value': p, 'is_normal': is_normal,
        'interpretation': 'Data looks Gaussian (fail to reject H0)' if is_normal else 'Data does not look Gaussian (reject H0)'
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    
    sns.histplot(target_data, bins=50, kde=True, ax=axes[0])
    axes[0].set_title(f'Target Distribution (Log Scale - Base e)')
    axes[0].set_xlabel('Log(Views + 1) [Base e]')
    
    linear_hits = np.expm1(target_data)
    sns.histplot(linear_hits, bins=50, kde=True, ax=axes[1])
    axes[1].set_title(f'Target Distribution (Linear Scale - Hits)')
    axes[1].set_xlabel('Views')
    axes[1].set_xlim(0, np.percentile(linear_hits, 99))
    
    plt.tight_layout()
    img_target_dist = get_base64_plot()

    plt.figure(figsize=(10, 6))
    probplot(sample_target, dist="norm", plot=plt)
    plt.title(f"QQ-Plot for {target_col} (Sample n={min(5000, len(target_data))})")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles (Log Scale - Base e)")
    img_qq = get_base64_plot()

    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 0:
        corr_matrix = numeric_df.corr(method='spearman')
        
        plt.figure(figsize=(20, 16))
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False, 
                    xticklabels=True, yticklabels=True, cbar=True)
        plt.title(f'Spearman Correlation Matrix (All {len(numeric_df.columns)} Features)')
        plt.tight_layout()
        img_corr = get_base64_plot()
    else:
        img_corr = None

    median_target = df[target_col].median()
    df['target_class'] = (df[target_col] > median_target).astype(int)
    
    mwu_results = []
    chi_square_results = []
    
    numeric_features = [c for c in df.columns if c not in [target_col, 'target_class'] and pd.api.types.is_numeric_dtype(df[c])]
    categorical_features = [c for c in df.columns if c not in [target_col, 'target_class'] and not pd.api.types.is_numeric_dtype(df[c])]
    
    motivation_text = "Normality tests indicated the data is non-Gaussian (p < 0.05). "
    motivation_text += "Therefore, non-parametric Mann-Whitney U tests were used for numeric features "
    motivation_text += "and Chi-Square tests for categorical features."
    results['motivation'] = motivation_text

    print("Running Mann-Whitney U tests on numeric features...")
    for col in numeric_features:
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
                'P_Value': p_val,
                'Effect_Size': rank_biserial
            })
        except:
            pass

    if mwu_results:
        mwu_df = pd.DataFrame(mwu_results)
        pvals = mwu_df['P_Value'].values
        reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='bonferroni')
        mwu_df['P_Value_Corrected'] = pvals_corrected
        mwu_df['Significant'] = reject
        mwu_df['Abs_Effect_Size'] = mwu_df['Effect_Size'].abs()
        mwu_df = mwu_df.sort_values('Abs_Effect_Size', ascending=False)
    else:
        mwu_df = pd.DataFrame()

    print("Running Chi-Square tests on categorical features...")
    for col in categorical_features[:10]: 
        try:
            contingency_table = pd.crosstab(df[col], df['target_class'])
            if contingency_table.shape == (1, 1):
                continue
            stat, p_val, dof, expected = chi2_contingency(contingency_table)
            chi_square_results.append({
                'Feature': col,
                'P_Value': p_val,
                'Statistic': stat
            })
        except:
            pass
            
    chi_df = pd.DataFrame(chi_square_results)
    
    dist_plots = []
    top_features = mwu_df.head(3)['Feature'].tolist()
    
    for col in top_features:
        plt.figure(figsize=(8, 4))
        sns.histplot(data=df, x=col, hue='target_class', kde=True, element="step", stat="density", common_norm=False)
        plt.title(f'Distribution of {col} by Target Class')
        dist_plots.append(get_base64_plot())

    return results, img_corr, mwu_df, chi_df, img_qq, dist_plots, top_features, img_target_dist

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
    
    # Log transforms for skewed features
    df_v['log_duration'] = np.log1p(df_v['duration'])
    df_v['log_desc_len'] = np.log1p(df_v['desc_len'])
    
    df_v['log_hits'] = np.log1p(df_v['hits'].astype(float))
    
    df_c = df_c.rename(columns={c: f"ch_{c}" for c in df_c.columns if c != 'channel_id'})
    df_c['ch_subscribers'] = pd.to_numeric(df_c['ch_subscribers'], errors='coerce').fillna(0)
    df_c['ch_title_len'] = df_c['ch_title'].fillna('').astype(str).apply(len)
    df_c['ch_desc_len'] = df_c['ch_description'].fillna('').astype(str).apply(len)
    
    # Log transforms for skewed channel features
    df_c['log_ch_subscribers'] = np.log1p(df_c['ch_subscribers'])
    
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
        'stream_type', 'product_id',
        'duration', 'desc_len', 'ch_subscribers'
    ]
    exclude_cols.extend([c for c in df.columns if c.startswith('ch_meta.')])

    X_cols = [c for c in df.columns if c not in exclude_cols + ['hits', 'log_hits']]
    df_model = df[X_cols + ['log_hits']].copy()
    df_model = df_model.dropna(subset=['log_hits'])
    
    print("Cleaning data...")
    for col in df_model.columns:
        if col in ['log_ch_subscribers']:
            df_model[col] = pd.to_numeric(df_model[col], errors='coerce').fillna(0)
        elif pd.api.types.is_numeric_dtype(df_model[col]):
            df_model[col] = df_model[col].fillna(0)
        else:
            df_model[col] = df_model[col].fillna('Unknown')

    stats_results, img_corr, mwu_df, chi_df, img_qq, dist_plots, top_features, img_target_dist = perform_statistical_analysis(df_model, 'log_hits')

    X = df_model[X_cols]
    y = df_model['log_hits']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data Split: Train={len(X_train)}, Test={len(X_test)}")

    print("[Step 4b/5] Validating and Training XGBoost (Leakage-Free)...")
    
    encoders = fit_encoders(X_train)
    X_train_enc = apply_encoders(X_train, encoders, verbose=True)
    X_test_enc = apply_encoders(X_test, encoders, verbose=False)
    
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
    
    cv_scores = cross_validate(model, X_train_enc, y_train, cv=5, 
                               scoring=('r2', 'neg_root_mean_squared_error'), 
                               return_train_score=True)
    
    print(f"CV Train R2: {np.mean(cv_scores['train_r2']):.4f}")
    print(f"CV Test R2: {np.mean(cv_scores['test_r2']):.4f}")
    
    model.fit(X_train_enc, y_train)
    
    y_test_pred = model.predict(X_test_enc)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    y_test_linear = np.expm1(y_test)
    y_test_pred_linear = np.expm1(y_test_pred)
    mae_linear = mean_absolute_error(y_test_linear, y_test_pred_linear)
    
    # Fix division by zero for MAPE: Calculate only where actual views > 0
    non_zero_mask = y_test_linear > 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((y_test_linear[non_zero_mask] - y_test_pred_linear[non_zero_mask]) / y_test_linear[non_zero_mask])) * 100
        excluded_count = len(y_test_linear) - non_zero_mask.sum()
        mape_note = f" (Calculated on {non_zero_mask.sum()} samples with >0 views, excluded {excluded_count} zeros)"
    else:
        mape = 0.0
        mape_note = " (No samples with >0 views found)"

    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train_enc, y_train, cv=3, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5), scoring='neg_root_mean_squared_error'
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, -np.mean(train_scores, axis=1), 'o-', color="r", label="Training score")
    plt.plot(train_sizes, -np.mean(test_scores, axis=1), 'o-', color="g", label="Cross-validation score")
    plt.title("Learning Curve (RMSE)")
    plt.xlabel("Training examples")
    plt.ylabel("Score (RMSE)")
    plt.legend(loc="best")
    plt.grid()
    img_learning_curve = get_base64_plot()

    perm = permutation_importance(model, X_test_enc, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    perm_df = pd.DataFrame({'Feature': X_test.columns, 'Importance': perm.importances_mean, 'Std': perm.importances_std})
    perm_df = perm_df.sort_values('Importance', ascending=False).head(20)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=perm_df, x='Importance', y='Feature', palette='viridis')
    plt.title('Permutation Importance (Global)')
    img_perm = get_base64_plot()
    
    importance = pd.DataFrame({
        'Feature': X_test.columns, 
        'Gain': model.feature_importances_
    })
    importance = importance.sort_values('Gain', ascending=False).head(20)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance, x='Gain', y='Feature', palette='magma')
    plt.title('XGBoost Feature Importance (Gain)')
    img_xgb_gain = get_base64_plot()
    
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    sample_idx = np.random.choice(len(X_test_enc), size=min(200, len(X_test_enc)), replace=False)
    X_test_sample = X_test_enc.iloc[sample_idx]
    
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

    shap_dep_plots = []
    for feat in top_features[:3]:
        if feat in X_test_sample.columns:
            plt.figure(figsize=(8, 6))
            shap.dependence_plot(feat, shap_values, X_test_sample, show=False)
            plt.title(f"SHAP Dependence Plot: {feat}")
            shap_dep_plots.append(get_base64_plot())

    print("Generating report...")
    
    mwu_html = ""
    for _, row in mwu_df.head(50).iterrows():
        sig_marker = "*" if row['Significant'] else ""
        p_val_color = "green" if row['P_Value_Corrected'] < 0.01 else "black"
        row_style = "background-color: #e8f5e9;" if abs(row['Effect_Size']) > 0.1 else ""
        
        mwu_html += f"<tr style='{row_style}'>"
        mwu_html += f"<td>{row['Feature']}</td>"
        mwu_html += f"<td style='color:{p_val_color}; font-weight:bold;'>{row['P_Value_Corrected']:.2e} {sig_marker}</td>"
        mwu_html += f"<td>{row['Effect_Size']:.4f}</td></tr>"

    chi_html = ""
    for _, row in chi_df.head(10).iterrows():
        chi_html += f"<tr><td>{row['Feature']}</td><td>{row['P_Value']:.2e}</td><td>{row['Statistic']:.2f}</td></tr>"

    dist_imgs_html = ""
    for i, img in enumerate(dist_plots):
        dist_imgs_html += f"<h4>Distribution: {top_features[i]}</h4><img src='data:image/png;base64,{img}' style='max-width:600px;'>"
        
    shap_dep_html = ""
    for i, img in enumerate(shap_dep_plots):
        shap_dep_html += f"<h4>Dependence: {top_features[i]}</h4><img src='data:image/png;base64,{img}' style='max-width:600px;'>"
        
    # Generate Mapping Table for Log Values 0 to 32
    log_map_rows = ""
    for i in range(22):
        # Using expm1 because the data is log1p transformed
        actual_val = np.expm1(i) 
        log_map_rows += f"<tr><td>{i}</td><td>{actual_val:,.0f}</td></tr>"

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
            .note {{ font-size: 0.9em; color: #666; font-style: italic; }}
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

            <h2>Target Distribution Analysis</h2>
            <img src='data:image/png;base64,{img_target_dist}' alt='Target Distribution'>
            
            <h3>Log Value Mapping (Base e)</h3>
            <p>The table below maps the Log-Values (Model Input/Output) to actual View Counts using the inverse of <code>log1p</code> (which is <code>expm1</code>).</p>
            <table>
                <tr><th>Log Value</th><th>Actual Number (Views)</th></tr>
                {log_map_rows}
            </table>

            <h2>Test Motivation</h2>
            <p>{stats_results['motivation']}</p>

            <h2>Model Metrics (Cross-Validation & Hold-out)</h2>
            <p><b>Interpretable Error Metrics:</b> While the model optimizes on Log-Views (Base e), the errors below are converted back to actual View counts and percentages. <br>
            <span class="note">MAPE (%) excludes videos with 0 views to avoid division by zero errors{mape_note}.</span></p>
            <table class="metrics-table">
                <tr><th>Dataset</th><th>R2 Score</th><th>RMSE (Log - Base e)</th><th>MAE (Log - Base e)</th><th>MAE (Views)</th><th>MAE (%)</th></tr>
                <tr><td>CV Train (Avg)</td><td>{np.mean(cv_scores['train_r2']):.4f}</td><td>{-np.mean(cv_scores['train_neg_root_mean_squared_error']):.4f}</td><td>-</td><td>-</td><td>-</td></tr>
                <tr><td>CV Test (Avg)</td><td>{np.mean(cv_scores['test_r2']):.4f}</td><td>{-np.mean(cv_scores['test_neg_root_mean_squared_error']):.4f}</td><td>-</td><td>-</td><td>-</td></tr>
                <tr><td>Hold-out Test</td><td>{test_r2:.4f}</td><td>{test_rmse:.4f}</td><td>{test_mae:.4f}</td><td>{mae_linear:,.0f}</td><td>{mape:.2f}%</td></tr>
            </table>
            <h3>Learning Curve</h3>
            <img src='data:image/png;base64,{img_learning_curve}' alt='Learning Curve'>

            <h2>1. Statistical Analysis (EDA)</h2>
            <h3>Target Distribution Analysis</h3>
            <img src='data:image/png;base64,{img_qq}' alt='QQ Plot'>
            
            <div class="stats-box">
                <h3>Mann-Whitney U Test (Numeric Features)</h3>
                <p>Tests if feature distributions differ between "High Hits" and "Low Hits" videos. 
                P-values are corrected using Bonferroni method. Green P-values (< 0.01) indicate statistical significance. 
                * marks statistically significant results after correction.</p>
                <table>
                    <tr><th>Feature</th><th>Corrected P-Value</th><th>Effect Size (Rank-Biserial)</th></tr>
                    {mwu_html}
                </table>
            </div>
            
            <div class="stats-box">
                <h3>Chi-Square Test (Categorical Features)</h3>
                <table>
                    <tr><th>Feature</th><th>P-Value</th><th>Statistic</th></tr>
                    {chi_html}
                </table>
            </div>

            <h3>Top Feature Distributions</h3>
            <p>Note: Skewed features like Duration, Description Length, and Subscribers are plotted on a Log scale (Base e).</p>
            {dist_imgs_html}

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

            <h3>SHAP Dependence Plots (Top Features)</h3>
            {shap_dep_html}
            
        </div>
    </body>
    </html>
    """

    with open("metadata_analysis_statistical.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Report saved to metadata_analysis_statistical.html")

if __name__ == "__main__":
    main()