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
main_build_features.py -- build daily feature dataset and save to parquet.
"""

import time
from utils import get_logger, print_banner, fmt_seconds
from dataset_builder import build_full_feature_dataset, split_dataset, save_features
from config import DAILY_FEATURES

log = get_logger()


def main():
    t0 = time.time()
    print_banner("GP DAILY FEATURE BUILDER")

    df = build_full_feature_dataset()
    path = save_features(df)

    train, val, test = split_dataset(df)

    log.info("")
    log.info("Split summary:")
    log.info(f"  Train : {len(train):,} daily bars across all stocks")
    log.info(f"  Val   : {len(val):,} daily bars across all stocks")
    log.info(f"  Test  : {len(test):,} daily bars across all stocks")
    log.info("")
    log.info(f"Features ({len(DAILY_FEATURES)}):")
    for f in DAILY_FEATURES:
        log.info(f"  {f}")
    log.info("")
    log.info(f"[OK] Done in {fmt_seconds(time.time() - t0)}")
    log.info(f"     Saved to: {path}")


if __name__ == "__main__":
    main()
