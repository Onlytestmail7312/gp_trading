import os
import shutil

results = {}

# ============================================================
# FIX 7: colors defined before functions in evaluate_results.py
# ============================================================
content = open('evaluate_results.py', encoding='utf-8').read()
# Find where colors is currently defined
idx = content.find('colors')
if idx >= 0:
    # Add colors at top after imports
    old = 'import matplotlib.pyplot as plt'
    new = '''import matplotlib.pyplot as plt

# Color palette -- defined here so all functions can access it
COLORS = [
    "#2196F3", "#4CAF50", "#FF9800", "#F44336",
    "#9C27B0", "#00BCD4", "#FF5722", "#607D8B",
]'''
    if old in content:
        content = content.replace(old, new)
        # Replace all uses of colors[ with COLORS[
        content = content.replace('colors[', 'COLORS[')
        content = content.replace('color=colors', 'color=COLORS')
        open('evaluate_results.py', 'w', encoding='utf-8').write(content)
        results[7] = 'FIXED'
    else:
        results[7] = 'import pattern not found'
else:
    results[7] = 'colors not found in file'

# ============================================================
# FIX 10: Logger singleton -> per-name cache in utils.py
# ============================================================
content = open('utils.py', encoding='utf-8').read()
old = '_logger: Optional[logging.Logger] = None'
if old in content:
    # Replace singleton with dict cache
    content = content.replace(
        '_logger: Optional[logging.Logger] = None',
        '_loggers: dict = {}'
    )
    # Replace the function body
    old_func = '''def get_logger(name: str = "gp_system") -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger'''
    new_func = '''def get_logger(name: str = "gp_system") -> logging.Logger:
    global _loggers
    if name in _loggers:
        return _loggers[name]'''
    if old_func in content:
        content = content.replace(old_func, new_func)
        # Replace _logger = logger with _loggers[name] = logger
        content = content.replace(
            '    _logger = logger\n    return _logger',
            '    _loggers[name] = logger\n    return _loggers[name]'
        )
        open('utils.py', 'w', encoding='utf-8').write(content)
        results[10] = 'FIXED'
    else:
        results[10] = 'function body not found'
else:
    results[10] = 'singleton pattern not found'
    idx = content.find('get_logger')
    print(f'#10 context: {repr(content[idx:idx+200])}')

# ============================================================
# FIX 13: Move one-off scripts to scripts/ folder
# ============================================================
scripts_dir = 'scripts'
os.makedirs(scripts_dir, exist_ok=True)

# Write a README for the scripts folder
readme = """# scripts/

One-off utility scripts used during development.
These are NOT production modules.

| Script | Purpose |
|--------|---------|
| fix.py | Various patch scripts applied during development |
| fix_batch2.py | Batch 2 fixes |
| fix_batch3.py | Batch 3 fixes |
| fix_all_pending.py | Batch 1 pending fixes |
| fix_equity.py | Equity curve fix |
| fix_overfit.py | Overfit guard fix |
| fix_minstocks.py | Min stocks fix |
| fix_nextopen.py | Next day open entry fix |
| fix_lookahead.py | Look-ahead fix |
| fix_lookahead2.py | Look-ahead fix part 2 |
| fix_main_evaluate.py | main_evaluate_gp.py import fix |
| fix_19_15.py | Issues 19 and 15 fix |
| fix_v5config.py | V5 config patch |
| check_formula.py | Check saved model formula |
| update_config.py | Config update utility |
"""
open(os.path.join(scripts_dir, 'README.md'), 'w', encoding='utf-8').write(readme)

moved = []
scripts_to_move = [
    'fix.py', 'fix_batch2.py', 'fix_batch3.py',
    'fix_all_pending.py', 'fix_equity.py', 'fix_overfit.py',
    'fix_minstocks.py', 'fix_nextopen.py', 'fix_lookahead.py',
    'fix_lookahead2.py', 'fix_main_evaluate.py', 'fix_19_15.py',
    'fix_v5config.py', 'check_formula.py', 'update_config.py',
]
for script in scripts_to_move:
    if os.path.exists(script):
        shutil.move(script, os.path.join(scripts_dir, script))
        moved.append(script)

results[13] = f'FIXED -- moved {len(moved)} scripts to scripts/'

# ============================================================
# FIX 18: EarlyStopping docstring in gp_engine.py
# ============================================================
content = open('gp_engine.py', encoding='utf-8').read()
old = 'class EarlyStopping:'
new = '''class EarlyStopping:
    """
    Early stopping monitor for GP evolution.

    Parameters
    ----------
    patience : int
        Number of consecutive generations without improvement
        before stopping. E.g., patience=25 stops after 25 stagnant
        generations.
    min_delta : float
        Minimum improvement in fitness to reset the patience counter.
    """'''
if old in content and '"""' not in content[content.find('class EarlyStopping:'):content.find('class EarlyStopping:')+200]:
    content = content.replace(old, new)
    open('gp_engine.py', 'w', encoding='utf-8').write(content)
    results[18] = 'FIXED'
else:
    results[18] = 'ALREADY HAS DOCSTRING or pattern not found'

# ============================================================
# FIX 26: CLI seed argument in main_train_gp.py
# ============================================================
content = open('main_train_gp.py', encoding='utf-8').read()
old = 'import random'
new = 'import random\nimport argparse'
if 'argparse' not in content:
    content = content.replace(old, new)

# Find where random.seed is set and add argparse before it
old_seed = '''random.seed(42)
    np.random.seed(42)'''
new_seed = '''parser = argparse.ArgumentParser(description="GP Trading System Trainer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args, _ = parser.parse_known_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    log.info(f"  Random seed: {args.seed}")'''

if old_seed in content:
    content = content.replace(old_seed, new_seed)
    open('main_train_gp.py', 'w', encoding='utf-8').write(content)
    results[26] = 'FIXED'
else:
    results[26] = 'seed pattern not found'
    idx = content.find('random.seed')
    print(f'#26 context: {repr(content[idx-20:idx+80])}')

# ============================================================
# FIX 27: Infix formula notation in gp_primitives.py
# ============================================================
content = open('gp_primitives.py', encoding='utf-8').read()
infix_code = '''

# ===========================================================================
# INFIX NOTATION CONVERTER
# ===========================================================================

# Operator symbols for infix display
_INFIX_OPS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "max": "max",
    "min": "min",
    "neg": "-",
    "sqrt": "sqrt",
    "log": "log",
    "exp": "exp",
}

def tree_to_infix(individual) -> str:
    """
    Convert a GP tree from prefix (LISP) notation to human-readable infix.

    Example:
        add(sub(rsi_14, half), mul(close_vs_sma20, ret_5d))
        -> ((rsi_14 - 0.5) + (close_vs_sma20 * ret_5d))
    """
    try:
        tokens = str(individual).replace("(", " ( ").replace(")", " ) ").replace(",", " ").split()
        result, _ = _parse_infix(tokens, 0)
        return result
    except Exception:
        return str(individual)


def _parse_infix(tokens, pos):
    """Recursive parser for infix conversion."""
    if pos >= len(tokens):
        return "?", pos

    token = tokens[pos]

    # Terminal (leaf node)
    if token not in _INFIX_OPS and token not in ("(", ")"):
        return token, pos + 1

    # Function call: func(arg1, arg2, ...)
    if token in _INFIX_OPS:
        op = _INFIX_OPS[token]
        pos += 1  # skip function name

        # Skip opening paren if present
        if pos < len(tokens) and tokens[pos] == "(":
            pos += 1

        args = []
        while pos < len(tokens) and tokens[pos] != ")":
            if tokens[pos] == ",":
                pos += 1
                continue
            arg, pos = _parse_infix(tokens, pos)
            args.append(arg)

        # Skip closing paren
        if pos < len(tokens) and tokens[pos] == ")":
            pos += 1

        if len(args) == 1:
            return f"{op}({args[0]})", pos
        elif len(args) == 2:
            return f"({args[0]} {op} {args[1]})", pos
        else:
            return f"{op}({', '.join(args)})", pos

    return token, pos + 1
'''

if 'def tree_to_infix' not in content:
    content = content + infix_code
    open('gp_primitives.py', 'w', encoding='utf-8').write(content)
    results[27] = 'FIXED'
else:
    results[27] = 'ALREADY EXISTS'

# ============================================================
# SUMMARY
# ============================================================
print('\n' + '='*50)
print('FIX BATCH 3 RESULTS')
print('='*50)
for num, status in sorted(results.items()):
    icon = '✅' if 'FIXED' in status or 'EXISTS' in status else '❌'
    print(f'  #{num}: {icon} {status}')
