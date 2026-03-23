"""
feature_pipeline.py
Step 1: Correlation pruning (54 -> ~25)
Step 2: RF importance ranking (~25 -> 15)
Step 3: Sanity check - ensure all signal types covered
Output: Final DAILY_FEATURES list ready for config.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

# =============================================================================
# CONFIG
# =============================================================================
PARQUET     = r"gp_output\gp_features_daily2.parquet"
TRAIN_END   = "2022-12-31"
CORR_THRESH = 0.85      # remove if corr > this
TOP_N       = 15        # final feature count
RF_TREES    = 300
RF_SEED     = 42

ALL_FEATURES = [
    "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
    "close_vs_sma5", "close_vs_sma10", "close_vs_sma20",
    "close_vs_sma50", "close_vs_sma200",
    "macd", "rsi_7", "rsi_14", "rsi_21",
    "bb_upper_20", "bb_lower_20", "bb_width_20", "bb_pos_20",
    "atr_pct_7", "atr_pct_14",
    "vol_ratio", "vol_ret",
    "mom_5", "mom_10", "mom_20",
    "pct_from_high20", "pct_from_low20",
    "pct_from_high52w", "pct_from_low52w",
    "body", "upper_wick", "lower_wick",
    "vol_20d", "vol_60d",
    "nifty_ret_1d", "nifty_ret_5d", "nifty_ret_20d",
    "nifty_vs_sma20", "nifty_rsi14", "rel_strength",
]

# Signal type categories for sanity check
CATEGORIES = {
    "Trend":       ["close_vs_sma20", "close_vs_sma50", "close_vs_sma200",
                    "macd", "mom_20"],
    "Momentum":    ["ret_1d", "ret_5d", "ret_20d", "rsi_14"],
    "Mean Rev":    ["bb_pos_20", "bb_width_20", "pct_from_high20",
                    "pct_from_low20"],
    "Volatility":  ["atr_pct_14", "vol_ratio", "vol_20d"],
    "52w Range":   ["pct_from_high52w", "pct_from_low52w"],
    "Candle":      ["body", "upper_wick", "lower_wick"],
    "Nifty":       ["nifty_ret_1d", "nifty_ret_5d", "rel_strength"],
}

# =============================================================================
# STEP 0: LOAD DATA
# =============================================================================
print("=" * 60)
print("FEATURE SELECTION PIPELINE")
print("=" * 60)

df = pd.read_parquet(PARQUET)
print(f"\n[0] Loaded {len(df):,} rows, {len(df.columns)} columns")

# Use train period only
train = df[df.index <= pd.Timestamp(TRAIN_END, tz='UTC')].copy()
print(f"    Train rows: {len(train):,}")

# Keep only available features
available = [f for f in ALL_FEATURES if f in train.columns]
print(f"    Available features: {len(available)}")

# =============================================================================
# STEP 1: CORRELATION PRUNING
# =============================================================================
print(f"\n[1] CORRELATION PRUNING (threshold={CORR_THRESH})")
print("-" * 40)

X_train = train[available].copy()
X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)

corr_matrix = X_train.corr().abs()

# Upper triangle only
upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

# Find pairs with high correlation
high_corr_pairs = []
for col in upper.columns:
    for row in upper.index:
        val = upper.loc[row, col]
        if pd.notna(val) and val > CORR_THRESH:
            high_corr_pairs.append((row, col, val))

high_corr_pairs.sort(key=lambda x: -x[2])

print(f"    High correlation pairs (>{CORR_THRESH}):")
for f1, f2, corr in high_corr_pairs:
    print(f"      {f1:25} ↔ {f2:25}  corr={corr:.3f}")

# Remove redundant features
# Strategy: for each correlated pair, remove the second one
to_remove = set()
for f1, f2, corr in high_corr_pairs:
    if f1 not in to_remove and f2 not in to_remove:
        to_remove.add(f2)
        print(f"      REMOVE: {f2} (correlated with {f1})")

after_corr = [f for f in available if f not in to_remove]
print(f"\n    Before: {len(available)} features")
print(f"    Removed: {len(to_remove)} features")
print(f"    After:  {len(after_corr)} features")
print(f"    Kept: {after_corr}")

# =============================================================================
# STEP 2: RF IMPORTANCE RANKING
# =============================================================================
print(f"\n[2] RF IMPORTANCE RANKING (trees={RF_TREES})")
print("-" * 40)

# Target = 5-day forward return
train = train.copy()
train['target'] = train.groupby('symbol')['ret_5d'].shift(-5)

X = train[after_corr].copy()
y = train['target'].copy()

X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
y = y.fillna(0)

mask = np.isfinite(X.values).all(axis=1) & np.isfinite(y.values)
X = X[mask]
y = y[mask]
print(f"    Clean rows for RF: {len(X):,}")

print(f"    Training RF ({RF_TREES} trees)...")
rf = RandomForestRegressor(
    n_estimators=RF_TREES,
    max_depth=6,
    min_samples_leaf=100,
    n_jobs=-1,
    random_state=RF_SEED
)
rf.fit(X, y)

importance = pd.Series(rf.feature_importances_, index=after_corr)
importance = importance.sort_values(ascending=False)

print(f"\n    Feature importance ranking:")
print(f"    {'Rank':<5} {'Feature':<25} {'Importance':<12} {'Bar'}")
print(f"    {'-'*60}")
for i, (feat, imp) in enumerate(importance.items(), 1):
    bar = '█' * int(imp * 300)
    marker = ' ← TOP 15' if i <= TOP_N else ''
    print(f"    {i:<5} {feat:<25} {imp:.4f}      {bar}{marker}")

top_features = importance.head(TOP_N).index.tolist()

# =============================================================================
# STEP 3: SANITY CHECK
# =============================================================================
print(f"\n[3] SANITY CHECK - Signal type coverage")
print("-" * 40)

missing_categories = []
final_features = top_features.copy()

for category, cat_features in CATEGORIES.items():
    covered = [f for f in cat_features if f in final_features]
    if covered:
        print(f"    ✅ {category:12}: {covered}")
    else:
        print(f"    ❌ {category:12}: NOT COVERED")
        # Add best available feature from this category
        available_in_cat = [f for f in cat_features if f in after_corr]
        if available_in_cat:
            # Pick highest RF importance from category
            cat_importance = importance[
                [f for f in available_in_cat if f in importance.index]
            ]
            if len(cat_importance) > 0:
                best = cat_importance.idxmax()
                final_features.append(best)
                print(f"       → Added: {best} (importance={importance.get(best, 0):.4f})")
                missing_categories.append(category)

# Deduplicate
final_features = list(dict.fromkeys(final_features))

# =============================================================================
# STEP 4: FINAL OUTPUT
# =============================================================================
print(f"\n[4] FINAL FEATURE LIST")
print("=" * 60)
print(f"    Started with:        {len(available)} features")
print(f"    After corr pruning:  {len(after_corr)} features")
print(f"    After RF top-{TOP_N}:     {len(top_features)} features")
print(f"    After sanity check:  {len(final_features)} features")

print(f"\n    Final {len(final_features)} features:")
for i, f in enumerate(final_features, 1):
    imp = importance.get(f, 0)
    cat = next((c for c, fs in CATEGORIES.items() if f in fs), "Other")
    print(f"    {i:2}. {f:25} importance={imp:.4f}  [{cat}]")

print(f"\n{'='*60}")
print(f"COPY THIS INTO config.py:")
print(f"{'='*60}")
print(f"DAILY_FEATURES = [")
for f in final_features:
    print(f'    "{f}",')
print(f"]")

# Also save to file
output = "DAILY_FEATURES = [\n"
for f in final_features:
    output += f'    "{f}",\n'
output += "]\n"
open("gp_output/selected_features.txt", "w").write(output)
print(f"\n[SAVED] gp_output/selected_features.txt")
