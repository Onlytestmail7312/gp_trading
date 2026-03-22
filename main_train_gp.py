import sys, os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""
main_train_gp.py -- run GP evolution for positional trading.
"""

import time
from pathlib import Path

from utils import get_logger, print_banner, fmt_seconds
from dataset_builder import load_features, split_dataset, prepare_stock_data
from gp_engine import run_evolution
from config import OUTPUT_DIR, ALL_SYMBOLS

log = get_logger()


def main():
    t0 = time.time()
    print_banner("GP POSITIONAL TRADING -- EVOLUTION")

    df = load_features(Path("gp_output/gp_features_daily.parquet"))

    train_df, val_df, test_df = split_dataset(df)

    log.info("Preparing per-stock data...")
    train_data = prepare_stock_data(train_df, ALL_SYMBOLS)
    val_data   = prepare_stock_data(val_df, ALL_SYMBOLS)

    log.info(f"  Train stocks: {list(train_data.keys())}")
    log.info(f"  Val stocks  : {list(val_data.keys())}")

    ckpt = Path("gp_output/checkpoint.pkl")
    if ckpt.exists():
        ckpt.unlink()
        log.info("  Old checkpoint deleted -- fresh evolution")

    best, stats = run_evolution(
        train_data = train_data,
        val_data   = val_data,
    )

    log.info("")
    log.info(f"[OK] Evolution complete in {fmt_seconds(time.time() - t0)}")
    log.info(f"     Best fitness : {best.fitness.values[0]:.4f}")
    log.info(f"     Tree size    : {len(best)} nodes")
    log.info(f"     Formula      : {best}")


if __name__ == "__main__":
    main()
