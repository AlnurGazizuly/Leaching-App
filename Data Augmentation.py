"""
synthesize_data.py
==================
Generates synthetic leaching-recovery data using Gradient Boosting Models (GBM).

Workflow:
  1. Load and clean the original experimental dataset.
  2. Train one GBM per target metal (Li, Co, Mn, Ni) to learn feed-condition → recovery mappings.
  3. Estimate realistic noise levels from per-group residuals.
  4. Synthesize new rows by perturbing process conditions within observed ranges.
  5. Predict recovery targets with the GBMs, then add calibrated noise.
  6. Save:
       - synthetic-only dataset  → OUTPUT_FILE
       - original + augmented    → COMBINED_OUTPUT

Outputs are validated against the original distribution before saving.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# ── Configuration ─────────────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

INPUT_FILE      = '/content/drive/MyDrive/final/original.xlsx'        # Your input file name. Check github for the initial dataset
OUTPUT_FILE     = '/content/drive/MyDrive/final/final(synthesized)ss.xlsx' # Synthesized file name
COMBINED_OUTPUT = '/content/drive/MyDrive/final/Final_combinedss.xlsx' # Combined file output. Change the names accordingly

TARGET_TOTAL_ROWS  = 5000   # approximate row count for synthetic-only output
AUGMENT_MULTIPLIER = 5      # each group is augmented this many times for the combined file
NOISE_SCALE        = 0.04   # fallback noise fraction of std when residuals are unavailable
RANDOM_SEED        = 42

# Column groups — keep in sync with the Excel headers
FEED_COLS    = ['Li in feed  %', 'Co in feed %', 'Mn in feed  %', 'Ni in feed %']
COND_COLS    = ['Concentration, M', 'Concentration %', 'Time,min  ', 'Temperature, C']
CAT_COLS     = ['Leaching agent ', 'Type of reducing agent ']
TARGET_COLS  = ['Li', 'Co', 'Mn', 'Ni']
ALL_COLS     = FEED_COLS + CAT_COLS + COND_COLS + TARGET_COLS


# ── Data helpers ──────────────────────────────────────────────────────────────

def standardize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise string categories: strip whitespace, title-case, unify nulls → 'Unknown'."""
    df = df.copy()
    for col in CAT_COLS:
        df[col] = (df[col].astype(str).str.strip().str.title()
                   .replace(['None', 'None_Used', 'Nan', 'nan', 'Unknown'], 'Unknown'))
    return df


# ── Model training ────────────────────────────────────────────────────────────

def build_target_model(df: pd.DataFrame):
    """
    Train one GBM per target metal on the full original dataset.
    Returns the fitted preprocessor and a {metal: gbm} dict.
    Models with fewer than 5 valid rows are skipped (stored as None).
    """
    numeric_features = FEED_COLS + COND_COLS

    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), numeric_features),
        ('cat', Pipeline([('ohe',     OneHotEncoder(handle_unknown='ignore',
                                                    sparse_output=False))]),   CAT_COLS),
    ])

    X = df[numeric_features + CAT_COLS]
    preprocessor.fit(X)

    models = {}
    for metal in TARGET_COLS:
        mask = df[metal].notna()
        if mask.sum() < 5:
            models[metal] = None
            continue
        gbm = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=3, random_state=RANDOM_SEED
        )
        gbm.fit(preprocessor.transform(X[mask]), df[metal][mask].values)
        models[metal] = gbm

    return preprocessor, models


# ── Noise estimation ──────────────────────────────────────────────────────────

def measure_noise_per_group(df: pd.DataFrame) -> dict:
    """
    Estimate realistic noise for each (metal, leaching_agent, reducing_agent) group.
    Uses the std of within-condition residuals; falls back to NOISE_SCALE * global std.
    Minimum noise floor: 0.5 percentage points.
    Returns {(metal, leaching, reducing): noise_std}.
    """
    noise = {}
    for name, grp in df.groupby(CAT_COLS):
        for metal in TARGET_COLS:
            vals = grp[metal].dropna()
            if len(vals) < 3:
                noise[(metal,) + name] = 2.0
                continue

            # Residuals from within-condition means
            residuals = []
            for _, cg in grp.groupby(COND_COLS, dropna=False)[metal]:
                cg = cg.dropna()
                if len(cg) > 1:
                    residuals.extend((cg - cg.mean()).values)

            noise_val = np.std(residuals) if len(residuals) >= 3 else vals.std() * NOISE_SCALE
            noise[(metal,) + name] = max(noise_val, 0.5)

    return noise


# ── Condition perturbation ────────────────────────────────────────────────────

def perturb_conditions(group_data: pd.DataFrame,
                        n_samples: int,
                        rng: np.random.Generator) -> pd.DataFrame:
    """
    Resample process conditions from the group with small Gaussian perturbations
    (5% of each feature's observed range). Values are clipped to the observed range.
    Groups with no reducing agent keep Concentration % = 0 unchanged.
    Time and Temperature are rounded to integers.
    """
    resampled = group_data[COND_COLS].sample(
        n=n_samples, replace=True, random_state=int(rng.integers(0, 1e6))
    ).reset_index(drop=True)

    group_has_no_reducer = (group_data[CAT_COLS[1]].iloc[0] == 'Unknown')

    for col in COND_COLS:
        col_range = group_data[col].max() - group_data[col].min()

        if col == 'Concentration %' and group_has_no_reducer:
            resampled[col] = 0.0
            continue
        if col_range < 1e-9:
            continue

        noise = rng.normal(0, col_range * 0.05, size=n_samples)
        resampled[col] = (resampled[col] + noise).clip(
            group_data[col].min(), group_data[col].max()
        )
        if col in ['Time,min  ', 'Temperature, C']:
            resampled[col] = resampled[col].round().astype(int)

    return resampled


# ── Target generation ─────────────────────────────────────────────────────────

def generate_targets(synth_conditions: pd.DataFrame,
                     feed_row: pd.Series,
                     group_name: tuple,
                     preprocessor,
                     models: dict,
                     noise_lookup: dict,
                     orig_group: pd.DataFrame,
                     rng: np.random.Generator) -> pd.DataFrame:
    """
    Predict recovery targets for synthetic rows using trained GBMs,
    then add calibrated Gaussian noise.

    Rules:
    - Metals never reported for a group remain NaN.
    - Predictions are clipped to [0, 100] and to [orig_min − 5, orig_max + 5].
    """
    n = len(synth_conditions)

    # Build input DataFrame with fixed feed composition and categorical context
    X_synth = synth_conditions.copy()
    for col in FEED_COLS:
        X_synth[col] = feed_row[col]
    for col in CAT_COLS:
        X_synth[col] = group_name[CAT_COLS.index(col)]

    X_proc = preprocessor.transform(X_synth[FEED_COLS + COND_COLS + CAT_COLS])

    target_df = pd.DataFrame(index=range(n))
    for metal in TARGET_COLS:
        metal_vals = orig_group[metal].dropna()
        if len(metal_vals) == 0 or models.get(metal) is None:
            target_df[metal] = np.nan
            continue

        pred  = models[metal].predict(X_proc)
        noise = rng.normal(0, noise_lookup.get((metal,) + group_name, 2.0), size=n)
        noisy = np.clip(pred + noise, 0.0, 100.0)
        noisy = np.clip(noisy,
                        max(0.0,   metal_vals.min() - 5.0),
                        min(100.0, metal_vals.max() + 5.0))
        target_df[metal] = np.round(noisy, 2)

    return target_df


# ── Group-level synthesis ──────────────────────────────────────────────────────

def synthesize_group(group_name: tuple,
                     orig_group: pd.DataFrame,
                     n_to_generate: int,
                     preprocessor,
                     models: dict,
                     noise_lookup: dict,
                     rng: np.random.Generator) -> pd.DataFrame:
    """
    Synthesize n_to_generate rows for one (leaching_agent, reducing_agent) group.
    Rows are split evenly across unique feed compositions found in the group.
    Groups with fewer than 2 original rows are handled by simple replication.
    """
    if len(orig_group) < 2:
        return orig_group.sample(n=n_to_generate, replace=True,
                                 random_state=int(rng.integers(0, 1e6))
                                 ).reset_index(drop=True)

    feed_combos = orig_group[FEED_COLS].drop_duplicates().reset_index(drop=True)
    n_combos    = len(feed_combos)
    base_n      = n_to_generate // n_combos
    remainder   = n_to_generate  % n_combos

    rows = []
    for i, (_, feed_row) in enumerate(feed_combos.iterrows()):
        n_this = base_n + (1 if i < remainder else 0)
        if n_this == 0:
            continue

        # Restrict to matching feed composition; fall back to full group if empty
        mask = np.ones(len(orig_group), dtype=bool)
        for col in FEED_COLS:
            mask &= (orig_group[col].values == feed_row[col])
        sub_group = orig_group[mask] if mask.any() else orig_group

        synth_conds   = perturb_conditions(sub_group, n_this, rng)
        synth_targets = generate_targets(synth_conds, feed_row, group_name,
                                          preprocessor, models, noise_lookup,
                                          sub_group, rng)

        row_df = synth_conds.copy()
        for col in FEED_COLS:
            row_df[col] = feed_row[col]
        row_df[CAT_COLS[0]] = group_name[0]
        row_df[CAT_COLS[1]] = group_name[1]
        for col in TARGET_COLS:
            row_df[col] = synth_targets[col].values

        rows.append(row_df)

    return pd.concat(rows, ignore_index=True)[ALL_COLS] if rows else pd.DataFrame(columns=ALL_COLS)


# ── Validation ────────────────────────────────────────────────────────────────

def validate_synthetic(orig: pd.DataFrame, synth: pd.DataFrame):
    """
    Print a comparison of key distribution statistics between
    original and synthetic datasets (mean, std, IQR, out-of-bounds %, per-group drift).
    """
    print("\n" + "=" * 65)
    print("VALIDATION: ORIGINAL vs SYNTHETIC STATISTICS")
    print("=" * 65)

    for metal in TARGET_COLS:
        o_vals = orig[metal].dropna()
        s_vals = synth[metal].dropna()
        if len(s_vals) == 0:
            continue
        print(f"\n  {metal}:")
        print(f"    Mean  : orig={o_vals.mean():.2f}  synth={s_vals.mean():.2f}  "
              f"(Δ={abs(s_vals.mean()-o_vals.mean()):.2f})")
        print(f"    Std   : orig={o_vals.std():.2f}   synth={s_vals.std():.2f}  "
              f"(Δ={abs(s_vals.std()-o_vals.std()):.2f})")
        print(f"    Q25   : orig={o_vals.quantile(0.25):.2f}   synth={s_vals.quantile(0.25):.2f}")
        print(f"    Q75   : orig={o_vals.quantile(0.75):.2f}   synth={s_vals.quantile(0.75):.2f}")
        pct_oob = ((s_vals < 0) | (s_vals > 100)).mean() * 100
        print(f"    Out-of-bounds [0,100]: {pct_oob:.2f}%  ← must be 0.00%")

    print("\n  Condition ranges (synthetic must not exceed original):")
    for col in COND_COLS:
        o_min, o_max = orig[col].min(), orig[col].max()
        s_min, s_max = synth[col].min(), synth[col].max()
        ok = "✓" if s_min >= o_min - 0.01 and s_max <= o_max + 0.01 else "✗ VIOLATION"
        print(f"    {col:25s}: orig=[{o_min:.2f},{o_max:.2f}]  "
              f"synth=[{s_min:.2f},{s_max:.2f}]  {ok}")

    print("\n  Per-group mean drift (top 5 worst, should be < 5%):")
    for metal in ['Li', 'Co']:
        drift = (synth.groupby(CAT_COLS)[metal].mean()
                 - orig.groupby(CAT_COLS)[metal].mean()).abs().dropna().sort_values(ascending=False)
        print(f"    {metal} drift (top 5):")
        for idx, val in drift.head(5).items():
            print(f"      {idx[0]} + {idx[1]}: {val:.2f}%")

    print("=" * 65)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(RANDOM_SEED)

    # 1. Load and clean original data
    print(f"Loading '{INPUT_FILE}'...")
    orig = pd.read_excel(INPUT_FILE)
    orig = standardize_categoricals(orig)
    for metal in TARGET_COLS:
        orig[metal] = pd.to_numeric(orig[metal], errors='coerce')
        orig = orig[(orig[metal] <= 100) | (orig[metal].isna())]
    print(f"  {len(orig)} rows loaded after cleaning.")

    # 2. Train GBMs and estimate per-group noise
    print("\nTraining GBM prediction models on original data...")
    preprocessor, models = build_target_model(orig)
    print(f"  Models trained for: {[m for m in TARGET_COLS if models.get(m)]}")

    print("Measuring per-group noise levels...")
    noise_lookup = measure_noise_per_group(orig)

    # 3. Determine how many synthetic rows to generate per group (proportional)
    groups       = orig.groupby(CAT_COLS)
    group_names  = list(groups.groups.keys())
    total_orig   = len(orig)
    n_per_group  = {
        name: max(1, int(round(len(groups.get_group(name)) / total_orig * TARGET_TOTAL_ROWS)))
        for name in group_names
    }

    # 4. Synthesize rows for each group
    print(f"\nGenerating ~{TARGET_TOTAL_ROWS} synthetic rows across {len(group_names)} groups...")
    all_synthetic = []
    for name in group_names:
        orig_group  = groups.get_group(name).reset_index(drop=True)
        synth_group = synthesize_group(name, orig_group, n_per_group[name],
                                        preprocessor, models, noise_lookup, rng)
        all_synthetic.append(synth_group)
        print(f"  ✓ {name[0]} + {name[1]}: {n_per_group[name]} synthetic rows")

    synthetic_df = pd.concat(all_synthetic, ignore_index=True)[ALL_COLS]

    # 5. Validate and save synthetic-only file
    validate_synthetic(orig, synthetic_df)
    synthetic_df.to_excel(OUTPUT_FILE, index=False)
    print(f"\nSynthetic-only dataset saved → '{OUTPUT_FILE}'  ({len(synthetic_df)} rows)")

    # 6. Build combined file: original + AUGMENT_MULTIPLIER × augmented per group
    print(f"\nBuilding combined dataset (original + {AUGMENT_MULTIPLIER}× augmented per group)...")
    augmented_parts = []
    for name in group_names:
        orig_group  = groups.get_group(name).reset_index(drop=True)
        synth_group = synthesize_group(name, orig_group, len(orig_group) * AUGMENT_MULTIPLIER,
                                        preprocessor, models, noise_lookup, rng)
        augmented_parts.append(synth_group)

    augmented_df = pd.concat(augmented_parts, ignore_index=True)
    combined_df  = pd.concat([orig, augmented_df], ignore_index=True)[ALL_COLS]
    combined_df.to_excel(COMBINED_OUTPUT, index=False)
    print(f"Combined dataset saved → '{COMBINED_OUTPUT}'  "
          f"({len(orig)} original + {len(augmented_df)} synthetic = {len(combined_df)} total rows)")

    # 7. Final summary
    oob = sum(
        (combined_df[m].dropna() > 100).sum() + (combined_df[m].dropna() < 0).sum()
        for m in TARGET_COLS
    )
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  Original rows       : {len(orig):,}")
    print(f"  Synthetic-only rows : {len(synthetic_df):,}  → '{OUTPUT_FILE}'")
    print(f"  Combined rows       : {len(combined_df):,}  → '{COMBINED_OUTPUT}'")
    print(f"  Out-of-bounds [0,100] in combined: {oob} rows")
    print("=" * 65)
    print("\nUse the COMBINED file as input to leaching_model.py")


if __name__ == '__main__':
    main()