"""
leaching_model.py
=================
Trains and evaluates Gradient Boosting Regressor (GBR) models for predicting
leaching recovery (%) of Li, Co, Mn, and Ni from battery black mass.

Four experiment configurations are run automatically:
  1. Original dataset  — with categorical features (leaching / reducing agent)
  2. Original dataset  — numeric features only
  3. Combined dataset  — with categorical features
  4. Combined dataset  — numeric features only

For each experiment the script prints:
  • Table 2  : 80/20 train-test split  (R², RMSE, MAE + Δ columns)
  • Table A  : 5-fold CV training-fold summary  (mean ± std)
  • Table B  : 5-fold CV validation-fold summary (mean ± std)

And (optionally) saves three PNG figures:
  - parity plot  (actual vs predicted)
  - partial dependence plots (PDP)
  - SHAP feature importance bar charts

Usage:
  Run in Google Colab after mounting Drive (cell at bottom mounts automatically).
  Update FILE_PATH_ORIGINAL / FILE_PATH_COMBINED if your paths differ.
"""

import logging
import warnings
warnings.filterwarnings('ignore')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing   import OneHotEncoder
from sklearn.impute           import SimpleImputer
from sklearn.pipeline         import Pipeline
from sklearn.compose          import ColumnTransformer
from sklearn.ensemble         import GradientBoostingRegressor
from sklearn.inspection       import partial_dependence
from sklearn.metrics          import r2_score, mean_absolute_error, mean_squared_error


# ── File paths ────────────────────────────────────────────────────────────────

FILE_PATH_ORIGINAL = '/content/drive/MyDrive/final/original.xlsx'
FILE_PATH_COMBINED = '/content/drive/MyDrive/final/Final_combined.xlsx'


# ── Feature definitions ───────────────────────────────────────────────────────

TARGET_COLUMNS = ['Li', 'Co', 'Mn', 'Ni']

NUMERIC_FEATURES = [
    'Li in feed  %', 'Co in feed %', 'Mn in feed  %', 'Ni in feed %',
    'Concentration, M', 'Concentration %', 'Time,min  ', 'Temperature, C',
]
CATEGORICAL_FEATURES = ['Leaching agent ', 'Type of reducing agent ']

# Interaction and polynomial terms added during feature engineering.
# Included in model training but excluded from SHAP / PDP displays.
ENGINEERED_FEATURES = [
    'Time_x_Temp', 'Leach_Conc_x_Time',
    'Temp_Squared', 'Time_Squared', 'Acid_to_Reducer',
]
ENGINEERED_FEATURES_SET = set(ENGINEERED_FEATURES)


# ── Aesthetics ────────────────────────────────────────────────────────────────

METAL_COLORS = {'Li': '#4C72B0', 'Co': '#DD8452', 'Mn': '#55A868', 'Ni': '#C44E52'}
METAL_LABELS = {'Li': 'Li Recovery', 'Co': 'Co Recovery',
                'Mn': 'Mn Recovery', 'Ni': 'Ni Recovery'}

# Human-readable axis labels for plots
RENAME_FEAT = {
    'Li in feed  %':    'Li Feed (%)',
    'Co in feed %':     'Co Feed (%)',
    'Mn in feed  %':    'Mn Feed (%)',
    'Ni in feed %':     'Ni Feed (%)',
    'Concentration, M': 'Leaching Agent Conc. (M)',
    'Concentration %':  'Reducing Agent Conc. (%)',
    'Time,min  ':       'Leaching Time (min)',
    'Temperature, C':   'Temperature (°C)',
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_and_clean(file_path: str) -> pd.DataFrame:
    """Load Excel file, coerce target columns to numeric, drop out-of-range rows."""
    print(f"  Loading '{file_path}'...")
    df = pd.read_excel(file_path)
    print(f"    Loaded {len(df)} rows, {df.shape[1]} columns.")
    for metal in TARGET_COLUMNS:
        df[metal] = pd.to_numeric(df[metal], errors='coerce')
        df = df[(df[metal] <= 100) | (df[metal].isna())]
    for col in CATEGORICAL_FEATURES:
        df[col] = (df[col].astype(str).str.strip().str.title()
                   .replace(['None', 'None_Used', 'Nan', 'nan'], 'Unknown'))
    print(f"    After cleaning: {len(df)} rows remain.")
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append interaction and polynomial features to capture non-linear effects:
      - Time × Temperature  (synergistic leaching effect)
      - Leaching concentration × Time
      - Temperature², Time²  (quadratic curvature)
      - Acid-to-reducer ratio  (stoichiometric proxy)
    """
    df  = df.copy()
    eps = 1e-6  # prevent division by zero for Acid_to_Reducer
    df['Time_x_Temp']       = df['Time,min  ']       * df['Temperature, C']
    df['Leach_Conc_x_Time'] = df['Concentration, M'] * df['Time,min  ']
    df['Temp_Squared']       = df['Temperature, C']  ** 2
    df['Time_Squared']       = df['Time,min  ']      ** 2
    df['Acid_to_Reducer']    = df['Concentration, M'] / (df['Concentration %'] + eps)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ML core
# ══════════════════════════════════════════════════════════════════════════════

def build_preprocessor(numeric_features: list, categorical_features: list):
    """
    ColumnTransformer with:
      - median imputation + passthrough for numeric features
      - OHE (ignore unknown) for categorical features (omit if list is empty)
    """
    transformers = [
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]),
         numeric_features),
    ]
    if categorical_features:
        transformers.append((
            'cat',
            Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore',
                                               sparse_output=False))]),
            categorical_features,
        ))
    return ColumnTransformer(transformers)


def train_global_model(X_train, y_train, preprocessor) -> dict:
    """
    Fit one GBR per target metal on the training split.
    Metals with fewer than 10 valid rows are skipped (stored as None).
    Returns {metal: fitted GBR or None}.
    """
    X_proc = preprocessor.fit_transform(X_train)
    gbm_params = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                      subsample=0.8, min_samples_leaf=5, random_state=42)
    models = {}
    for metal in y_train.columns:
        mask = y_train[metal].notna().values
        if mask.sum() < 10:
            models[metal] = None
            continue
        gbm = GradientBoostingRegressor(**gbm_params)
        gbm.fit(X_proc[mask], y_train[metal].values[mask])
        models[metal] = gbm
        print(f"    {metal}: trained on {mask.sum():,} rows.")
    return models


def compute_metrics(true_vals, pred_vals) -> dict:
    """Return R², RMSE, MAE, and sample size for a set of predictions."""
    return {
        'R2'  : r2_score(true_vals, pred_vals),
        'RMSE': np.sqrt(mean_squared_error(true_vals, pred_vals)),
        'MAE' : mean_absolute_error(true_vals, pred_vals),
        'n'   : len(true_vals),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Metrics tables
# ══════════════════════════════════════════════════════════════════════════════

def _sep_and_fmt(col_labels):
    """
    Build a separator line and a row-formatter function for ASCII tables.
    col_labels: list of (column_name, width) tuples.
    """
    sep = '+' + '+'.join('-' * (w + 2) for _, w in col_labels) + '+'

    def fmt_cell(key, val, width):
        if isinstance(val, float):
            s = f'{val:+.4f}' if key.startswith('Δ') else (f'{val:.4f}' if 'R²' in key or 'R2' in key else f'{val:.2f}')
        else:
            s = str(val)
        return ' ' + s.ljust(width) + ' '

    def fmt_row(d):
        return '|' + '|'.join(fmt_cell(k, d.get(k, ''), w) for k, w in col_labels) + '|'

    return sep, fmt_row


def print_train_test_table(models, preprocessor,
                            X_train, y_train, X_test, y_test,
                            target_columns, dataset_label: str = ''):
    """
    Compute and print Table 2: per-metal and pooled metrics for the 80/20 split.
    Δ columns show the difference between test and train (negative = generalisation gap).
    """
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)

    rows = []
    all_tr_true, all_tr_pred = [], []
    all_te_true, all_te_pred = [], []

    for metal in target_columns:
        gbm = models.get(metal)
        if gbm is None:
            continue
        tr_mask = y_train[metal].notna().values
        te_mask = y_test[metal].notna().values
        tr_m = compute_metrics(y_train[metal].values[tr_mask],
                               gbm.predict(X_train_proc[tr_mask]))
        te_m = compute_metrics(y_test[metal].values[te_mask],
                               gbm.predict(X_test_proc[te_mask]))
        rows.append({
            'Target'    : f'{metal} extracted (%)', 'Model': 'GBR',
            'Train R²'  : tr_m['R2'],   'Train RMSE': tr_m['RMSE'], 'Train MAE': tr_m['MAE'],
            'Test R²'   : te_m['R2'],   'Test RMSE' : te_m['RMSE'], 'Test MAE' : te_m['MAE'],
            'ΔR²'       : te_m['R2']   - tr_m['R2'],
            'ΔRMSE'     : te_m['RMSE'] - tr_m['RMSE'],
            'ΔMAE'      : te_m['MAE']  - tr_m['MAE'],
        })
        all_tr_true.extend(y_train[metal].values[tr_mask])
        all_tr_pred.extend(gbm.predict(X_train_proc[tr_mask]))
        all_te_true.extend(y_test[metal].values[te_mask])
        all_te_pred.extend(gbm.predict(X_test_proc[te_mask]))

    # Pooled row across all metals
    tr_all = compute_metrics(np.array(all_tr_true), np.array(all_tr_pred))
    te_all = compute_metrics(np.array(all_te_true), np.array(all_te_pred))
    pooled = {
        'Target': 'ALL (pooled)', 'Model': 'GBR',
        'Train R²': tr_all['R2'],   'Train RMSE': tr_all['RMSE'], 'Train MAE': tr_all['MAE'],
        'Test R²' : te_all['R2'],   'Test RMSE' : te_all['RMSE'], 'Test MAE' : te_all['MAE'],
        'ΔR²'  : te_all['R2']   - tr_all['R2'],
        'ΔRMSE': te_all['RMSE'] - tr_all['RMSE'],
        'ΔMAE' : te_all['MAE']  - tr_all['MAE'],
    }

    col_labels = [
        ('Target', 22), ('Model', 6),
        ('Train R²', 9), ('Train RMSE', 11), ('Train MAE', 10),
        ('Test R²',  8), ('Test RMSE',  10), ('Test MAE',   9),
        ('ΔR²', 8), ('ΔRMSE', 8), ('ΔMAE', 8),
    ]
    sep, fmt_row = _sep_and_fmt(col_labels)
    title = f'Table 2. R², RMSE, MAE — 80/20 train/test split'
    if dataset_label:
        title += f'  [{dataset_label}]'
    print(f'\n  {title}')
    print(sep)
    print(fmt_row({k: k for k, _ in col_labels}))
    print(sep)
    for r in rows:
        print(fmt_row(r))
    print(sep)
    print(fmt_row(pooled))
    print(sep)

    return pd.DataFrame(rows)


def run_cross_validation(df, numeric_features, categorical_features,
                          target_columns, n_splits=5, dataset_label: str = ''):
    """
    Run stratified k-fold CV and print:
      Table A — per-fold training metrics (mean ± std)
      Table B — per-fold validation metrics (mean ± std)
    """
    print(f"\n  Running {n_splits}-fold CV  [{dataset_label}]...")

    all_feature_cols = numeric_features + categorical_features
    X = df[all_feature_cols]
    y = df[target_columns]

    kf      = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_tr = {m: {'R2': [], 'RMSE': [], 'MAE': []} for m in target_columns}
    fold_va = {m: {'R2': [], 'RMSE': [], 'MAE': []} for m in target_columns}

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
        X_tr, X_va = X.iloc[tr_idx],  X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx],  y.iloc[va_idx]

        prep    = build_preprocessor(numeric_features, categorical_features)
        fmodels = train_global_model(X_tr, y_tr, prep)
        X_tr_proc = prep.transform(X_tr)
        X_va_proc = prep.transform(X_va)

        for metal in target_columns:
            gbm = fmodels.get(metal)
            if gbm is None:
                continue
            for split_mask, split_X, split_y, store in [
                (y_tr[metal].notna().values, X_tr_proc, y_tr[metal], fold_tr[metal]),
                (y_va[metal].notna().values, X_va_proc, y_va[metal], fold_va[metal]),
            ]:
                if split_mask.sum() >= 2:
                    m = compute_metrics(split_y.values[split_mask],
                                        gbm.predict(split_X[split_mask]))
                    for k in ('R2', 'RMSE', 'MAE'):
                        store[k].append(m[k])
        print(f"    Fold {fold} done.")

    def _build_rows_pooled(fd):
        rows = []
        all_r2, all_rmse, all_mae = [], [], []
        for metal in target_columns:
            d = fd[metal]
            if not d['R2']:
                continue
            rows.append({
                'Target'   : f'{metal} extracted (%)', 'Model': 'GBR',
                'R² mean'  : np.mean(d['R2']),   'R² std'  : np.std(d['R2']),
                'RMSE mean': np.mean(d['RMSE']),  'RMSE std': np.std(d['RMSE']),
                'MAE mean' : np.mean(d['MAE']),   'MAE std' : np.std(d['MAE']),
            })
            all_r2.extend(d['R2']); all_rmse.extend(d['RMSE']); all_mae.extend(d['MAE'])
        pooled = {
            'Target'   : 'ALL (pooled)', 'Model': 'GBR',
            'R² mean'  : np.mean(all_r2),   'R² std'  : np.std(all_r2),
            'RMSE mean': np.mean(all_rmse),  'RMSE std': np.std(all_rmse),
            'MAE mean' : np.mean(all_mae),   'MAE std' : np.std(all_mae),
        }
        return rows, pooled

    col_labels = [
        ('Target', 22), ('Model', 6),
        ('R² mean', 9), ('R² std', 8),
        ('RMSE mean', 10), ('RMSE std', 9),
        ('MAE mean', 9), ('MAE std', 8),
    ]
    sep, fmt_row = _sep_and_fmt(col_labels)
    for table_name, fd in [
        (f'Table A. {n_splits}-fold CV — Training folds   [{dataset_label}]',  fold_tr),
        (f'Table B. {n_splits}-fold CV — Validation folds  [{dataset_label}]', fold_va),
    ]:
        rows, pooled = _build_rows_pooled(fd)
        print(f'\n  {table_name}')
        print(sep)
        print(fmt_row({k: k for k, _ in col_labels}))
        print(sep)
        for r in rows:
            print(fmt_row(r))
        print(sep)
        print(fmt_row(pooled))
        print(sep)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_parity(models, preprocessor, X_test, y_test,
                target_columns, save_path='parity_plots.png'):
    """
    Scatter plot of actual vs predicted recovery (%) for each metal,
    plus a combined panel. Perfect prediction = red dashed diagonal.
    """
    X_proc = preprocessor.transform(X_test)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    all_true, all_pred = [], []

    for i, metal in enumerate(target_columns):
        gbm = models.get(metal)
        if gbm is None:
            axes[i].set_visible(False)
            continue
        mask  = y_test[metal].notna().values
        true  = y_test[metal].values[mask]
        pred  = np.clip(gbm.predict(X_proc[mask]), 0, 100)
        m     = compute_metrics(true, pred)
        ax    = axes[i]
        ax.scatter(true, pred, alpha=0.45, s=18, color=METAL_COLORS[metal], edgecolors='none')
        ax.plot([0, 100], [0, 100], 'r--', lw=1.2, label='Perfect')
        ax.set_xlabel('Actual (%)', fontsize=10)
        ax.set_ylabel('Predicted (%)', fontsize=10)
        ax.set_title(METAL_LABELS[metal], fontsize=12, fontweight='bold')
        ax.set_xlim(0, 105); ax.set_ylim(0, 105)
        stats = f"R²={m['R2']:.3f}\nMAE={m['MAE']:.1f}%\nRMSE={m['RMSE']:.1f}%"
        ax.text(0.04, 0.96, stats, transform=ax.transAxes, va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax.text(0.96, 0.04, f'n={mask.sum():,}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8, color='grey')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        all_true.extend(true); all_pred.extend(pred)

    # Combined all-metals panel
    ax = axes[4]
    for metal in target_columns:
        gbm = models.get(metal)
        if gbm is None:
            continue
        mask = y_test[metal].notna().values
        true = y_test[metal].values[mask]
        pred = np.clip(gbm.predict(X_proc[mask]), 0, 100)
        ax.scatter(true, pred, alpha=0.3, s=12, color=METAL_COLORS[metal],
                   edgecolors='none', label=metal)
    ax.plot([0, 100], [0, 100], 'r--', lw=1.2)
    all_m = compute_metrics(np.array(all_true), np.array(all_pred))
    stats = f"R²={all_m['R2']:.3f}\nMAE={all_m['MAE']:.1f}%\nRMSE={all_m['RMSE']:.1f}%"
    ax.text(0.04, 0.96, stats, transform=ax.transAxes, va='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.set_xlabel('Actual (%)', fontsize=10); ax.set_ylabel('Predicted (%)', fontsize=10)
    ax.set_title('All Metals Combined', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 105); ax.set_ylim(0, 105)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    axes[5].set_visible(False)
    fig.suptitle('Actual vs Predicted Leaching Efficiency',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Parity plots saved → '{save_path}'")
    plt.show()


def plot_partial_dependence(models, preprocessor, X_train,
                             all_numeric_features, categorical_features,
                             target_columns, pdp_features=None,
                             save_path='pdp_plots.png'):
    """
    Partial dependence plots (PDPs) for key process variables.
    One row per target metal; columns correspond to the selected features.
    Engineered features are excluded — only original numeric columns are shown.
    """
    if pdp_features is None:
        pdp_features = ['Temperature, C', 'Time,min  ',
                        'Concentration, M', 'Concentration %']
    pdp_features = [f for f in pdp_features if f in all_numeric_features]
    if not pdp_features:
        print("  No valid PDP features found — skipping PDP plot.")
        return

    X_proc       = preprocessor.transform(X_train)
    feat_idx_map = {name: i for i, name in enumerate(all_numeric_features)}
    n_rows, n_cols = len(target_columns), len(pdp_features)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4 * n_cols, 3.5 * n_rows), sharey='row')
    if n_rows == 1: axes = axes[np.newaxis, :]
    if n_cols == 1: axes = axes[:, np.newaxis]

    for ri, metal in enumerate(target_columns):
        gbm   = models.get(metal)
        color = METAL_COLORS[metal]
        for ci, feat in enumerate(pdp_features):
            ax      = axes[ri][ci]
            feat_ix = feat_idx_map.get(feat)
            if gbm is None or feat_ix is None:
                ax.set_visible(False)
                continue
            res       = partial_dependence(gbm, X_proc, features=[feat_ix],
                                           kind='average', grid_resolution=50)
            grid_vals = res['grid_values'][0]
            avg_pred  = res['average'][0]
            ax.plot(grid_vals, avg_pred, color=color, lw=2)
            ax.fill_between(grid_vals, avg_pred, avg_pred, alpha=0.15, color=color)
            ax.grid(True, alpha=0.3); ax.tick_params(labelsize=8)
            if ri == 0:
                ax.set_title(feat.strip().replace('  ', ' '), fontsize=9, fontweight='bold')
            if ci == 0:
                ax.set_ylabel(f'{metal} – PDP', fontsize=9)

    fig.suptitle('Partial Dependence Plots', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  PDP plots saved → '{save_path}'")
    plt.show()


def plot_shap_importance(models, preprocessor, X_train,
                          all_numeric_features, categorical_features,
                          target_columns, top_n=8,
                          save_path='shap_importance.png'):
    """
    SHAP-based feature importance using TreeExplainer.
    Engineered features are stripped; only original inputs are shown.
    One panel per metal + one global-average panel.
    """
    import shap
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 11})

    # Build full feature name list in the order the preprocessor outputs them
    ohe_names = []
    if categorical_features:
        ohe       = preprocessor.named_transformers_['cat']['onehot']
        ohe_names = list(ohe.get_feature_names_out(categorical_features))
    raw_names = np.array(all_numeric_features + ohe_names)

    # Keep only non-engineered features for display
    keep_mask = np.array([f not in ENGINEERED_FEATURES_SET for f in raw_names])
    raw_kept  = raw_names[keep_mask]

    def pretty(name):
        if name in RENAME_FEAT:
            return RENAME_FEAT[name]
        for cat in categorical_features:
            base = cat.strip()
            if name.startswith(base):
                suffix = name[len(base):].lstrip('_ ').strip()
                return f"{base.title()} — {suffix}" if suffix else base.title()
        return name

    feat_labels    = np.array([pretty(n) for n in raw_kept])
    X_proc         = preprocessor.transform(X_train)
    trained_metals = [m for m in target_columns if models.get(m) is not None]

    n_panels    = len(trained_metals) + 1          # +1 for global average
    n_cols_plot = min(3, n_panels)
    n_rows_plot = (n_panels + n_cols_plot - 1) // n_cols_plot
    fig, axes   = plt.subplots(n_rows_plot, n_cols_plot,
                                figsize=(6.5 * n_cols_plot, 5 * n_rows_plot))
    axes = np.array(axes).flatten()

    LIGHT = '#f7f7f7'; GRID = '#e8e8e8'
    all_mean_shap = []

    for i, metal in enumerate(trained_metals):
        gbm       = models[metal]
        explainer = shap.TreeExplainer(gbm)
        shap_vals = explainer.shap_values(X_proc)     # shape: (n_samples, n_all_features)
        sv_kept   = shap_vals[:, keep_mask]
        mean_shap = np.abs(sv_kept).mean(axis=0)
        all_mean_shap.append(mean_shap)

        idx   = np.argsort(mean_shap)[-top_n:]
        color = METAL_COLORS[metal]
        ax    = axes[i]
        bars  = ax.barh(feat_labels[idx], mean_shap[idx],
                        color=color, alpha=0.85, edgecolor='white', linewidth=0.4)
        ax.set_xlabel('Mean |SHAP value|', fontsize=9)
        ax.set_title(f'{metal} Extraction (%)', fontsize=11, fontweight='bold')
        ax.set_facecolor(LIGHT)
        ax.grid(axis='x', color=GRID, lw=0.6)
        ax.spines[['top', 'right']].set_visible(False)
        for bar, val in zip(bars, mean_shap[idx]):
            ax.text(val + mean_shap.max() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=7.5, color='#333')

    # Global average panel
    avg_shap = np.mean(all_mean_shap, axis=0)
    idx_avg  = np.argsort(avg_shap)[-top_n:]
    ax_avg   = axes[len(trained_metals)]
    bars = ax_avg.barh(feat_labels[idx_avg], avg_shap[idx_avg],
                       color='#4a4a6a', alpha=0.85, edgecolor='white', linewidth=0.4)
    ax_avg.set_xlabel('Mean |SHAP value|', fontsize=9)
    ax_avg.set_title('Global Average (All Metals)', fontsize=11, fontweight='bold')
    ax_avg.set_facecolor(LIGHT)
    ax_avg.grid(axis='x', color=GRID, lw=0.6)
    ax_avg.spines[['top', 'right']].set_visible(False)
    for bar, val in zip(bars, avg_shap[idx_avg]):
        ax_avg.text(val + avg_shap.max() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=7.5, color='#333')

    for j in range(len(trained_metals) + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Figure 4.  SHAP-based Feature Importance — '
                 f'GBR Models (Top {top_n} Original Features)',
                 fontsize=13, fontweight='bold', y=1.01, color='#1a1a2e')
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    print(f"  SHAP importance saved → '{save_path}'")
    plt.show()

    # Print numerical summary
    summary = pd.DataFrame(
        {metal: pd.Series(all_mean_shap[i], index=feat_labels)
         for i, metal in enumerate(trained_metals)}
    )
    summary['Average'] = summary.mean(axis=1)
    print("\n  Mean |SHAP| per original feature:")
    print(summary.sort_values('Average', ascending=False).round(4).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Experiment runner
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(df, use_categoricals: bool, experiment_label: str,
                   save_prefix: str, generate_plots: bool = True, n_cv_splits: int = 5):
    """
    Full pipeline for one configuration:
      - use_categoricals=True  → include leaching / reducing agent columns
      - use_categoricals=False → numeric-only model

    Steps:
      1. Select features, split 80/20 train/test.
      2. Fit preprocessor + train GBRs.
      3. Print Table 2 (train-test split metrics).
      4. Print Tables A & B (k-fold CV metrics).
      5. Optionally generate and save parity, PDP, and SHAP plots.

    Returns: (models, preprocessor, X_train, X_test, y_train, y_test)
    """
    banner = f'  EXPERIMENT: {experiment_label}'
    print('\n' + '═' * len(banner))
    print(banner)
    print('═' * len(banner))

    all_numeric  = NUMERIC_FEATURES + ENGINEERED_FEATURES
    cat_features = CATEGORICAL_FEATURES if use_categoricals else []
    feature_cols = all_numeric + cat_features

    X = df[feature_cols]
    y = df[TARGET_COLUMNS]
    print(f"\n  Dataset: {len(X):,} samples  |  Features: {len(all_numeric)} numeric"
          + (f" + {len(cat_features)} categorical" if cat_features else " (no categoricals)"))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42)
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    preprocessor = build_preprocessor(all_numeric, cat_features)
    print(f"\n  Training models...")
    models = train_global_model(X_train, y_train, preprocessor)

    print_train_test_table(models, preprocessor,
                            X_train, y_train, X_test, y_test,
                            TARGET_COLUMNS, dataset_label=experiment_label)

    run_cross_validation(df, all_numeric, cat_features,
                          TARGET_COLUMNS, n_splits=n_cv_splits,
                          dataset_label=experiment_label)

    if generate_plots:
        print(f"\n  Generating plots for [{experiment_label}]...")
        plot_parity(models, preprocessor, X_test, y_test, TARGET_COLUMNS,
                    save_path=f'{save_prefix}_parity.png')
        plot_partial_dependence(
            models, preprocessor, X_train,
            all_numeric, cat_features, TARGET_COLUMNS,
            pdp_features=['Temperature, C', 'Time,min  ',
                          'Concentration, M', 'Concentration %'],
            save_path=f'{save_prefix}_pdp.png')
        plot_shap_importance(models, preprocessor, X_train,
                              all_numeric, cat_features, TARGET_COLUMNS,
                              top_n=8, save_path=f'{save_prefix}_shap.png')

    return models, preprocessor, X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Main: mount Drive, load data, run all four experiments
# ══════════════════════════════════════════════════════════════════════════════

from google.colab import drive
drive.mount('/content/drive')

print('Loading datasets...')
df_original = add_engineered_features(load_and_clean(FILE_PATH_ORIGINAL))
df_combined = add_engineered_features(load_and_clean(FILE_PATH_COMBINED))

# Each call is fully independent — results do not depend on execution order.
run_experiment(df_original, use_categoricals=True,
               experiment_label='Original dataset — with categoricals',
               save_prefix='orig_cat')

run_experiment(df_original, use_categoricals=False,
               experiment_label='Original dataset — without categoricals',
               save_prefix='orig_nocat')

run_experiment(df_combined, use_categoricals=True,
               experiment_label='Combined dataset — with categoricals',
               save_prefix='comb_cat')

run_experiment(df_combined, use_categoricals=False,
               experiment_label='Combined dataset — without categoricals',
               save_prefix='comb_nocat')

print('\n' + '═' * 60)
print('  ALL EXPERIMENTS COMPLETE')
print('═' * 60)