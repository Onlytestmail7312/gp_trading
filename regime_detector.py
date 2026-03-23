"""
regime_detector.py
==================
Complete regime detection pipeline using Hidden Markov Model.

Steps:
    1.  Load Nifty daily data from DuckDB snapshot
    2.  Compute features (returns + volatility + trend + drawdown)
    3.  Fit 4-state Gaussian HMM
    4.  Predict hard labels + soft probabilities
    5.  Print state statistics
    6.  Draw regime chart (price + probabilities + states)
    7.  Save regime CSV
    8.  Merge regime labels into main dataset
    9.  Add 5-day smoothed probabilities
    10. Save final dataset

States (data-driven, confirmed by user):
    S0 = BULL_QUIET     (green)
    S1 = CRISIS         (blue)
    S2 = SIDEWAYS_BEAR  (red)
    S3 = VOLATILE       (orange)

Output columns added to dataset:
    regime               - raw HMM label
    prob_bull_quiet      - raw probability
    prob_crisis          - raw probability
    prob_sideways_bear   - raw probability
    prob_volatile        - raw probability
    smooth_bull_quiet    - 5d rolling mean
    smooth_sideways_bear - 5d rolling mean
    smooth_volatile      - 5d rolling mean
    smooth_crisis        - 5d rolling mean
    regime_smooth        - smoothed hard label
"""

import numpy as np
import pandas as pd
import duckdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================
SNAPSHOT_DB  = r"gp_output\gp_snapshot2.duckdb"
PARQUET_IN   = r"gp_output\gp_features_daily2.parquet"
PARQUET_OUT  = r"gp_output\gp_features_daily2.parquet"
REGIME_CSV   = r"gp_output\nifty_regimes.csv"
CHART_FILE   = r"gp_output\regime_chart.png"

N_STATES     = 4
N_ITER       = 2000
RANDOM_SEED  = 42
SMOOTH_WINDOW = 5

# State labels confirmed by user after chart review
STATE_LABELS = {
    0: 'BULL_QUIET',
    1: 'CRISIS',
    2: 'SIDEWAYS_BEAR',
    3: 'VOLATILE',
}

# Colors for chart
STATE_COLORS = {
    0: '#2ecc71',   # green  = BULL_QUIET
    1: '#3498db',   # blue   = CRISIS
    2: '#e74c3c',   # red    = SIDEWAYS_BEAR
    3: '#f39c12',   # orange = VOLATILE
}

LABEL_COLORS = {
    'BULL_QUIET':    '#2ecc71',
    'CRISIS':        '#3498db',
    'SIDEWAYS_BEAR': '#e74c3c',
    'VOLATILE':      '#f39c12',
}

# =============================================================================
# STEP 1: LOAD NIFTY DATA
# =============================================================================
print("=" * 65)
print("  REGIME DETECTOR -- HMM 4-STATE + SMOOTHING")
print("=" * 65)

print("\n[1] Loading Nifty daily data...")
con = duckdb.connect(SNAPSHOT_DB, read_only=True)
nifty = con.execute("""
    SELECT date, open, high, low, close, volume
    FROM daily_1d
    WHERE symbol = 'NIFTY'
    ORDER BY date
""").fetchdf()
con.close()

nifty['date'] = pd.to_datetime(nifty['date'])
nifty = nifty.set_index('date').sort_index()
print(f"    Rows      : {len(nifty):,}")
print(f"    Date range: {nifty.index.min().date()} to {nifty.index.max().date()}")

# =============================================================================
# STEP 2: COMPUTE FEATURES
# =============================================================================
print("\n[2] Computing HMM features...")

nifty['ret_1d']   = nifty['close'].pct_change(1)
nifty['ret_5d']   = nifty['close'].pct_change(5)
nifty['ret_20d']  = nifty['close'].pct_change(20)
nifty['vol_5d']   = nifty['ret_1d'].rolling(5).std()
nifty['vol_20d']  = nifty['ret_1d'].rolling(20).std()
nifty['vol_60d']  = nifty['ret_1d'].rolling(60).std()
nifty['sma50']    = nifty['close'].rolling(50).mean()
nifty['sma200']   = nifty['close'].rolling(200).mean()
nifty['trend']    = (nifty['close'] - nifty['sma50']) / (nifty['sma50'] + 1e-9)
nifty['high_52w'] = nifty['close'].rolling(252).max()
nifty['drawdown'] = (nifty['close'] - nifty['high_52w']) / (nifty['high_52w'] + 1e-9)

nifty_clean = nifty.dropna().copy()
print(f"    Clean rows: {len(nifty_clean):,}")

feature_cols = ['ret_1d', 'ret_5d', 'vol_5d', 'vol_20d', 'trend', 'drawdown']
X = nifty_clean[feature_cols].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"    Features  : {feature_cols}")

# =============================================================================
# STEP 3: FIT HMM
# =============================================================================
print(f"\n[3] Fitting {N_STATES}-state Gaussian HMM...")

np.random.seed(RANDOM_SEED)
model = hmm.GaussianHMM(
    n_components=N_STATES,
    covariance_type='full',
    n_iter=N_ITER,
    random_state=RANDOM_SEED,
    verbose=False
)
model.fit(X_scaled)

print(f"    Converged      : {model.monitor_.converged}")
print(f"    Log-likelihood : {model.score(X_scaled):.2f}")

# =============================================================================
# STEP 4: PREDICT
# =============================================================================
print("\n[4] Predicting states...")

hard_labels = model.predict(X_scaled)
soft_probs  = model.predict_proba(X_scaled)

nifty_clean = nifty_clean.copy()
nifty_clean['state'] = hard_labels
for i in range(N_STATES):
    nifty_clean[f'prob_state_{i}'] = soft_probs[:, i]

nifty_clean['regime'] = nifty_clean['state'].map(STATE_LABELS)

# =============================================================================
# STEP 5: STATE STATISTICS
# =============================================================================
print("\n[5] State statistics")
print("=" * 75)
print(f"  {'State':<6} {'Label':<15} {'Days':>5} {'Pct':>6} "
      f"{'Ret1d':>8} {'Ret20d':>8} {'Vol20d':>8} {'AvgDD':>8}")
print("-" * 75)

state_stats = {}
for state in range(N_STATES):
    mask  = hard_labels == state
    label = STATE_LABELS[state]
    stats = {
        'label':    label,
        'days':     int(mask.sum()),
        'pct':      mask.sum() / len(hard_labels) * 100,
        'ret_1d':   nifty_clean.loc[mask, 'ret_1d'].mean(),
        'ret_20d':  nifty_clean.loc[mask, 'ret_20d'].mean(),
        'vol_20d':  nifty_clean.loc[mask, 'vol_20d'].mean(),
        'drawdown': nifty_clean.loc[mask, 'drawdown'].mean(),
        'color':    STATE_COLORS[state],
    }
    state_stats[state] = stats
    print(f"  S{state}     {label:<15} {stats['days']:>5} {stats['pct']:>5.1f}%"
          f"  {stats['ret_1d']:>8.4f}"
          f"  {stats['ret_20d']:>8.4f}"
          f"  {stats['vol_20d']:>8.4f}"
          f"  {stats['drawdown']:>8.4f}")
print("=" * 75)

# =============================================================================
# STEP 6: YEARLY BREAKDOWN
# =============================================================================
print("\n[6] State distribution by year...")
nifty_clean['year'] = nifty_clean.index.year
yearly = nifty_clean.groupby(['year', 'regime']).size().unstack(fill_value=0)
for label in STATE_LABELS.values():
    if label not in yearly.columns:
        yearly[label] = 0
yearly = yearly[['BULL_QUIET', 'VOLATILE', 'SIDEWAYS_BEAR', 'CRISIS']]
print(yearly.to_string())

# =============================================================================
# STEP 7: TRANSITION MATRIX
# =============================================================================
print("\n[7] Transition matrix...")
labels = [f'S{i}({STATE_LABELS[i][:4]})' for i in range(N_STATES)]
trans  = pd.DataFrame(model.transmat_, index=labels, columns=labels)
print(trans.round(3).to_string())

# =============================================================================
# STEP 8: COMPUTE SMOOTHED PROBABILITIES
# =============================================================================
print(f"\n[8] Computing {SMOOTH_WINDOW}-day smoothed probabilities...")

# Rename raw prob columns
nifty_clean = nifty_clean.rename(columns={
    'prob_state_0': 'prob_bull_quiet',
    'prob_state_1': 'prob_crisis',
    'prob_state_2': 'prob_sideways_bear',
    'prob_state_3': 'prob_volatile',
})

# Compute smoothed probabilities
prob_cols = ['prob_bull_quiet', 'prob_crisis',
             'prob_sideways_bear', 'prob_volatile']
smooth_map = {
    'prob_bull_quiet':    'smooth_bull_quiet',
    'prob_crisis':        'smooth_crisis',
    'prob_sideways_bear': 'smooth_sideways_bear',
    'prob_volatile':      'smooth_volatile',
}

for raw_col, smooth_col in smooth_map.items():
    nifty_clean[smooth_col] = nifty_clean[raw_col].rolling(
        SMOOTH_WINDOW, min_periods=1
    ).mean()

# Smoothed hard label
smooth_cols = list(smooth_map.values())
label_map = {
    'smooth_bull_quiet':    'BULL_QUIET',
    'smooth_sideways_bear': 'SIDEWAYS_BEAR',
    'smooth_volatile':      'VOLATILE',
    'smooth_crisis':        'CRISIS',
}
nifty_clean['regime_smooth'] = (
    nifty_clean[smooth_cols]
    .idxmax(axis=1)
    .map(label_map)
)

print(f"\n    Raw regime distribution:")
print(nifty_clean['regime'].value_counts().to_string())
print(f"\n    Smoothed regime distribution:")
print(nifty_clean['regime_smooth'].value_counts().to_string())

# =============================================================================
# STEP 9: DRAW REGIME CHART
# =============================================================================
print(f"\n[9] Drawing regime chart...")

fig, axes = plt.subplots(4, 1, figsize=(20, 16),
                          gridspec_kw={'height_ratios': [3, 1, 1, 1]})
fig.suptitle('Nifty Market Regimes -- HMM 4-State + 5-Day Smoothing (2015-2026)',
             fontsize=14, fontweight='bold')

dates = nifty_clean.index

# --- Panel 1: Nifty price with RAW regime background ---
ax1 = axes[0]
ax1.plot(dates, nifty_clean['close'],
         color='black', linewidth=0.8, zorder=5)

prev_state = hard_labels[0]
start_idx  = 0
for i in range(1, len(hard_labels)):
    if hard_labels[i] != prev_state or i == len(hard_labels) - 1:
        end_idx = i if hard_labels[i] != prev_state else i + 1
        ax1.axvspan(
            dates[start_idx],
            dates[min(end_idx, len(dates)-1)],
            alpha=0.25,
            color=STATE_COLORS[prev_state],
            linewidth=0
        )
        start_idx  = i
        prev_state = hard_labels[i]

ax1.set_ylabel('Nifty Close', fontsize=10)
ax1.set_title('Nifty Price -- Raw HMM Regime', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(dates[0], dates[-1])

patches = [
    mpatches.Patch(
        color=STATE_COLORS[i], alpha=0.6,
        label=f"{STATE_LABELS[i]}  "
              f"ret={state_stats[i]['ret_1d']:.4f}  "
              f"vol={state_stats[i]['vol_20d']:.4f}  "
              f"({state_stats[i]['pct']:.1f}%)"
    )
    for i in range(N_STATES)
]
ax1.legend(handles=patches, loc='upper left', fontsize=8)

# Add key event annotations
events = {
    '2020-03-23': 'COVID\nCrash',
    '2022-01-01': 'Rate\nHikes',
    '2024-06-01': 'Election\nRally',
}
for date_str, label in events.items():
    try:
        d = pd.Timestamp(date_str)
        nearest = nifty_clean.index[
            nifty_clean.index.get_indexer([d], method='nearest')[0]
        ]
        price = nifty_clean.loc[nearest, 'close']
        ax1.annotate(
            label,
            xy=(nearest, price),
            xytext=(nearest, price * 1.06),
            fontsize=7, ha='center',
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.8)
        )
    except Exception:
        pass

# --- Panel 2: Nifty price with SMOOTHED regime background ---
ax2 = axes[1]
ax2.plot(dates, nifty_clean['close'],
         color='black', linewidth=0.8, zorder=5)

smooth_labels = nifty_clean['regime_smooth'].values
prev_regime   = smooth_labels[0]
start_idx     = 0
for i in range(1, len(smooth_labels)):
    if smooth_labels[i] != prev_regime or i == len(smooth_labels) - 1:
        end_idx = i if smooth_labels[i] != prev_regime else i + 1
        ax2.axvspan(
            dates[start_idx],
            dates[min(end_idx, len(dates)-1)],
            alpha=0.25,
            color=LABEL_COLORS.get(prev_regime, 'gray'),
            linewidth=0
        )
        start_idx  = i
        prev_regime = smooth_labels[i]

ax2.set_ylabel('Nifty Close', fontsize=10)
ax2.set_title(f'Nifty Price -- Smoothed Regime ({SMOOTH_WINDOW}-day rolling)', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(dates[0], dates[-1])

# --- Panel 3: Raw soft probabilities ---
ax3 = axes[2]
for i in range(N_STATES):
    col = list(smooth_map.keys())[i]
    ax3.plot(dates,
             nifty_clean[col],
             color=STATE_COLORS[i],
             linewidth=0.7,
             alpha=0.9,
             label=STATE_LABELS[i])
ax3.set_ylabel('Probability', fontsize=10)
ax3.set_title('Raw Soft Probabilities', fontsize=11)
ax3.legend(loc='upper right', fontsize=8, ncol=4)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0, 1)
ax3.set_xlim(dates[0], dates[-1])

# --- Panel 4: Smoothed probabilities ---
ax4 = axes[3]
for i in range(N_STATES):
    smooth_col = smooth_cols[i]
    ax4.plot(dates,
             nifty_clean[smooth_col],
             color=STATE_COLORS[i],
             linewidth=0.7,
             alpha=0.9,
             label=STATE_LABELS[i])
ax4.set_ylabel('Probability', fontsize=10)
ax4.set_title(f'Smoothed Probabilities ({SMOOTH_WINDOW}-day rolling mean)', fontsize=11)
ax4.legend(loc='upper right', fontsize=8, ncol=4)
ax4.grid(True, alpha=0.3)
ax4.set_ylim(0, 1)
ax4.set_xlim(dates[0], dates[-1])

plt.tight_layout()
plt.savefig(CHART_FILE, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {CHART_FILE}")

# =============================================================================
# STEP 10: SAVE REGIME CSV
# =============================================================================
print(f"\n[10] Saving regime CSV...")
save_cols = [
    'close', 'state', 'regime',
    'ret_1d', 'ret_20d', 'vol_20d', 'drawdown',
    'prob_bull_quiet', 'prob_crisis',
    'prob_sideways_bear', 'prob_volatile',
    'smooth_bull_quiet', 'smooth_crisis',
    'smooth_sideways_bear', 'smooth_volatile',
    'regime_smooth',
]
regime_df = nifty_clean[save_cols].copy()
regime_df.index.name = 'date'
regime_df.to_csv(REGIME_CSV)
print(f"    Saved: {REGIME_CSV}")

# =============================================================================
# STEP 11: MERGE INTO MAIN DATASET
# =============================================================================
print(f"\n[11] Merging into main dataset...")

df = pd.read_parquet(PARQUET_IN)
print(f"    Rows before   : {len(df):,}")
print(f"    Columns before: {len(df.columns)}")

# Drop old regime columns
old_cols = [
    'regime', 'prob_bull_quiet', 'prob_crisis',
    'prob_sideways_bear', 'prob_volatile',
    'smooth_bull_quiet', 'smooth_crisis',
    'smooth_sideways_bear', 'smooth_volatile',
    'regime_smooth',
    'prob_bull', 'prob_bear', 'prob_sideways',
    'prob_state_0', 'prob_state_1',
    'prob_state_2', 'prob_state_3',
]
df = df.drop(columns=[c for c in old_cols if c in df.columns])

# Prepare merge columns
merge_cols = [
    'regime',
    'prob_bull_quiet', 'prob_crisis',
    'prob_sideways_bear', 'prob_volatile',
    'smooth_bull_quiet', 'smooth_crisis',
    'smooth_sideways_bear', 'smooth_volatile',
    'regime_smooth',
]
merge_df = regime_df[merge_cols].copy()
merge_df.index = pd.to_datetime(merge_df.index, utc=True)

# Join
df = df.join(merge_df, how='left')

# Fill missing
for col in merge_cols:
    if col in ['regime', 'regime_smooth']:
        df[col] = df[col].ffill().bfill()
    else:
        df[col] = df[col].ffill().bfill()

print(f"    Rows after    : {len(df):,}")
print(f"    Columns after : {len(df.columns)}")
print(f"    Missing values: {df[merge_cols].isna().sum().sum()}")

print(f"\n    Raw regime distribution:")
print(df['regime'].value_counts().to_string())
print(f"\n    Smoothed regime distribution:")
print(df['regime_smooth'].value_counts().to_string())

# Save
df.to_parquet(PARQUET_OUT, index=True)
print(f"\n    Saved: {PARQUET_OUT}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("  COMPLETE")
print("=" * 65)
print(f"  Chart        : {CHART_FILE}")
print(f"  CSV          : {REGIME_CSV}")
print(f"  Dataset      : {PARQUET_OUT}")
print(f"  Total columns: {len(df.columns)}")
print()
print("  Columns added:")
print("    regime               - raw HMM label")
print("    prob_bull_quiet      - raw HMM probability")
print("    prob_crisis          - raw HMM probability")
print("    prob_sideways_bear   - raw HMM probability")
print("    prob_volatile        - raw HMM probability")
print(f"   smooth_bull_quiet    - {SMOOTH_WINDOW}d rolling mean")
print(f"   smooth_sideways_bear - {SMOOTH_WINDOW}d rolling mean")
print(f"   smooth_volatile      - {SMOOTH_WINDOW}d rolling mean")
print(f"   smooth_crisis        - {SMOOTH_WINDOW}d rolling mean")
print("    regime_smooth        - smoothed hard label")
print()
print("  Next: use regime_smooth + smooth probs in V8 fitness function")
print("=" * 65)
