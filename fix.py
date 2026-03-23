results = {}

# ============================================================
# FIX 1: Update DAILY_FEATURES to 16 selected features
# ============================================================
content = open('config.py', encoding='utf-8').read()

old = 'DAILY_FEATURES = ['
idx = content.find(old)
end = content.find(']', idx) + 1
old_block = content[idx:end]

new_block = '''DAILY_FEATURES = [
    "vol_60d",
    "nifty_rsi14",
    "pct_from_high52w",
    "nifty_ret_20d",
    "nifty_ret_5d",
    "rel_strength",
    "vol_20d",
    "pct_from_low52w",
    "rsi_7",
    "close_vs_sma200",
    "bb_width_20",
    "atr_pct_7",
    "ret_20d",
    "ret_5d",
    "bb_upper_20",
    "upper_wick",
]'''

if old_block:
    content = content.replace(old_block, new_block)
    results[1] = 'FIXED features -> 16'
else:
    results[1] = 'NOT FOUND'

# ============================================================
# FIX 2: Update date range to recent regime (2019-2022)
# ============================================================
content = content.replace(
    'TRAIN_START = "2015-01-01"',
    'TRAIN_START = "2019-01-01"'
)
results[2] = 'FIXED TRAIN_START -> 2019'

# ============================================================
# FIX 3: GP parameters for V7
# ============================================================
replacements = [
    ('GP_POPULATION    = 6000', 'GP_POPULATION    = 5000'),
    ('GP_GENERATIONS   = 80',   'GP_GENERATIONS   = 100'),
    ('GP_ELITE         = 40',   'GP_ELITE         = 30'),
    ('GP_TOURNAMENT    = 5',    'GP_TOURNAMENT    = 4'),
    ('GP_CROSSOVER     = 0.70', 'GP_CROSSOVER     = 0.70'),
    ('GP_MUTATION      = 0.20', 'GP_MUTATION      = 0.25'),
    ('GP_MAX_DEPTH     = 6',    'GP_MAX_DEPTH     = 5'),
    ('GP_MAX_NODES     = 25',   'GP_MAX_NODES     = 20'),
    ('GP_EARLY_STOP    = 20',   'GP_EARLY_STOP    = 25'),
    ('COMPLEXITY_PENALTY = 0.3','COMPLEXITY_PENALTY = 0.05'),
]

fixed = []
not_found = []
for old, new in replacements:
    if old in content:
        content = content.replace(old, new)
        fixed.append(new.split('=')[0].strip())
    else:
        not_found.append(old.split('=')[0].strip())

results[3] = f'FIXED: {len(fixed)} params'
if not_found:
    results[3] += f' | NOT FOUND: {not_found}'

# ============================================================
# FIX 4: Linear parsimony in fitness.py
# ============================================================
fit_content = open('fitness.py', encoding='utf-8').read()

# Find and replace complexity penalty calculation
old_complexity = 'complexity = COMPLEXITY_PENALTY * (tree_size / GP_MAX_NODES) ** 2'
new_complexity = '''# Linear parsimony pressure (Koza 1992)
    # AdjustedFitness = RawFitness - (lambda * TreeSize)
    LAMBDA_PARSIMONY = 0.05
    complexity = LAMBDA_PARSIMONY * tree_size'''

if old_complexity in fit_content:
    fit_content = fit_content.replace(old_complexity, new_complexity)
    open('fitness.py', 'w', encoding='utf-8').write(fit_content)
    results[4] = 'FIXED linear parsimony in fitness.py'
else:
    results[4] = 'NOT FOUND in fitness.py'
    idx = fit_content.find('complexity')
    print(f'complexity context: {repr(fit_content[idx:idx+150])}')

open('config.py', 'w', encoding='utf-8').write(content)

# ============================================================
# SUMMARY
# ============================================================
print('\n' + '='*50)
print('FIX V7 RESULTS')
print('='*50)
for num, status in sorted(results.items()):
    icon = '✅' if 'FIXED' in status else '❌'
    print(f'  #{num}: {icon} {status}')

# Verify config
print('\n--- V7 CONFIG ---')
content = open('config.py').read()
for line in content.split('\n'):
    if any(x in line for x in [
        'TRAIN_START', 'TRAIN_END', 'GP_POPULATION',
        'GP_GENERATIONS', 'GP_TOURNAMENT', 'GP_MUTATION',
        'GP_MAX_DEPTH', 'GP_MAX_NODES', 'GP_EARLY_STOP',
        'COMPLEXITY_PENALTY', 'DAILY_FEATURES'
    ]):
        if '=' in line and not line.strip().startswith('#'):
            print(f'  {line.strip()}')

print('\n--- V7 FITNESS ---')
fit_content = open('fitness.py').read()
idx = fit_content.find('LAMBDA_PARSIMONY')
if idx >= 0:
    print(fit_content[idx:idx+100])
