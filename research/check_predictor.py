#!/usr/bin/env python3
"""Leakage smoke test for the Stock AI Predictor (or any module exposing train_ticker(ticker, hist_dataframe) -> metrics).

    python3 research/check_predictor.py --repo ../stock-ai-predictor

Trains the repo's own train_ticker on pure random-walk prices, where NO model can have an edge, and reads its reported
direction_accuracy. Honest model: about 50% (noise ranges roughly 40-62% on a single 20% split). A leaking model reports
80-100%. Exit code 1 = leakage detected, 0 = clean. Needs numpy + pandas + scikit-learn + joblib; a missing `yfinance` is stubbed.
"""

import argparse
import os
import sys
import tempfile
import types
import warnings


def random_walk(seed, n=700):
    import numpy as np
    import pandas as pd
    r = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(r.normal(0, 0.012, n)))
    open_ = np.r_[close[0], close[:-1]] * (1 + r.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(r.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(r.normal(0, 0.003, n)))
    return pd.DataFrame({"date": pd.bdate_range("2022-01-03", periods=n).strftime("%Y-%m-%d"), "open": open_, "high": high,
                         "low": low, "close": close, "volume": r.integers(1_000_000, 2_000_000, n)})


def check(app_dir, seeds=4):
    warnings.filterwarnings("ignore")
    sys.modules.setdefault("yfinance", types.ModuleType("yfinance"))
    sys.path.insert(0, app_dir)
    sys.modules.pop("ml_model", None)
    try:
        import ml_model as m
        if hasattr(m, "MODEL_DIR"):
            m.MODEL_DIR = tempfile.mkdtemp()
        accs = [float(m.train_ticker("CHK%d" % s, random_walk(s))["direction_accuracy"]) for s in range(seeds)]
    finally:
        sys.path.remove(app_dir)
        sys.modules.pop("ml_model", None)
    mean = sum(accs) / len(accs)
    return {"accuracies": accs, "mean": round(mean, 2), "leakage": mean >= 70 or max(accs) >= 90}


def main(argv=None):
    p = argparse.ArgumentParser(description="Random-walk leakage test for the Stock AI Predictor.")
    p.add_argument("--repo", required=True, help="path to the stock-ai-predictor checkout")
    a = p.parse_args(argv)
    app = os.path.join(a.repo, "stock_system", "app")
    if not os.path.exists(os.path.join(app, "ml_model.py")):
        print("no stock_system/app/ml_model.py under %s" % a.repo)
        return 2
    r = check(app)
    print("direction_accuracy on random walks (should be ~50): %s  mean %s" % (r["accuracies"], r["mean"]))
    print("LEAKAGE DETECTED: the model is being handed the answer. Apply predictor-fixes.patch." if r["leakage"]
          else "CLEAN: no sign of the answer leaking into the features.")
    return 1 if r["leakage"] else 0


if __name__ == "__main__":
    sys.exit(main())
