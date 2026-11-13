#!/usr/bin/env python3
"""Fit and store the deployable ORB filter: one model + its calibrator + its threshold.

    python3 research/model_store.py --bars-dir research/bars --out research/out/model

Procedure = the one walk-forward validated: fit on TRAIN, calibrate on CALIBRATION, pick model and threshold on
VALIDATION. The locked TEST segment is NOT used, so the deployed model's first honest out-of-sample evidence is paper
trading. Writes model.pkl + model.json (sha256 of the pickle, features, threshold, trained-through date, symbols and
`ledger_ref` = the newest REAL-data walk-forward entry at build time).

Security: the model is a pickle. load() verifies the sha256 in model.json first, but only load models you built
yourself; never one from someone else.
"""

import argparse
import datetime
import hashlib
import json
import os
import pickle
import sys

import numpy as np
from sklearn.metrics import brier_score_loss

import backtest as bt
import meta_models as mm
import orb_dataset as o
import strategies as ss


class ModelError(Exception):
    pass


class Deployable:
    def __init__(self, name, model, cal, threshold, cols, meta):
        self.name, self.model, self.cal, self.threshold, self.cols, self.meta = name, model, cal, threshold, cols, meta

    def score(self, row):
        """Calibrated P(target before stop) for ONE candidate row (features as of the trigger bar)."""
        X = mm.matrix([row], self.cols)
        return float(self.cal(self.model.proba(X))[0])

    def take(self, row):
        p = self.score(row)
        return p >= self.threshold, p


def latest_real_walkforward(ledger_path, strategy=None, bar_seconds=None):
    """Newest walk-forward ledger entry whose data was NOT synthetic (and, if given, for that strategy; entries
    written before strategies existed count as 'orb'), or None."""
    if not os.path.exists(ledger_path):
        return None
    best = None
    with open(ledger_path, encoding="utf-8") as fh:
        for line in fh:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if (e.get("kind") == "walk-forward" and not str(e.get("source", "")).upper().startswith("SYNTHETIC")
                    and (strategy is None or e.get("strategy", "orb") == strategy)
                    and (bar_seconds is None or e.get("bar_seconds", 300) == bar_seconds)):
                best = e
    return best


def fit_deployable(rows, seed=7, cost_bps=2.0, fractions=(0.5, 0.15, 0.15, 0.2), embargo_days=1, strategy="orb", bar_seconds=300, backtest_cfg=None):
    cols = o.feature_columns(rows)
    seg = o.chrono_split(rows, fractions=fractions, embargo_days=embargo_days)
    tr, ca, va = mm.binary(seg["train"]), mm.binary(seg["calibration"]), seg["validation"]
    if len(tr) < mm.MIN_FIT or len({r["label"] for r in tr}) < 2:
        raise ModelError("only %d resolved training trades (need %d): get more tickers / history"
                         % (len(tr), mm.MIN_FIT))
    Xtr, ytr = mm.matrix(tr, cols), np.array([r["label"] for r in tr])
    Xca, yca = mm.matrix(ca, cols), np.array([r["label"] for r in ca])
    Xva = mm.matrix(va, cols)
    bva = [i for i, r in enumerate(va) if r["label"] in (0, 1)]
    yva_b = np.array([va[i]["label"] for i in bva])
    min_taken = max(5, mm.MIN_TAKEN // 3)
    best = None
    for m in mm.make_models(seed):
        if m.name == "rate":
            continue
        m.fit(Xtr, ytr)
        raw_va = m.proba(Xva)
        cal = mm.Raw() if len(ca) == 0 else mm.pick_calibrator(m.proba(Xca), yca, raw_va[bva] if bva else np.array([]), yva_b)
        pva = cal(raw_va)
        pick = mm.choose_threshold(va, pva, cost_bps, min_taken, backtest_cfg=backtest_cfg) if len(va) else None
        if pick is None or not bva or len(set(yva_b)) < 2:
            continue
        brier = float(brier_score_loss(yva_b, np.clip(pva[bva], 1e-6, 1 - 1e-6)))
        if best is None or brier < best[0]:
            best = (brier, m, cal, pick[0])
    if best is None:
        raise ModelError("no model produced a usable threshold on the validation segment")
    brier, m, cal, th = best
    last_day = max(r["date"] for r in seg["validation"] + seg["calibration"] + seg["train"])
    meta = {"strategy": strategy, "bar_seconds": bar_seconds, "model": m.name, "calibration": cal.name, "threshold": th, "validation_brier": round(brier, 4),
            "features": cols, "trained_through": last_day, "symbols": sorted({r.get("symbol", "?") for r in rows}),
            "rows": {k: len(v) for k, v in seg.items()}, "seed": seed, "cost_bps": cost_bps,
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    return Deployable(m.name, m, cal, th, cols, meta)


def save(dep, out_dir, ledger_path=None):
    os.makedirs(out_dir, exist_ok=True)
    blob = pickle.dumps(dep)
    wf = latest_real_walkforward(ledger_path, dep.meta.get("strategy", "orb"), dep.meta.get("bar_seconds", 300)) if ledger_path else None
    meta = dict(dep.meta, sha256=hashlib.sha256(blob).hexdigest(),
                ledger_ref=wf["id"] if wf else None, ledger_result=wf["result"] if wf else None)
    with open(os.path.join(out_dir, "model.pkl"), "wb") as fh:
        fh.write(blob)
    with open(os.path.join(out_dir, "model.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    return meta


def load(out_dir):
    """-> (Deployable, meta). Verifies the pickle's sha256 against model.json before unpickling."""
    with open(os.path.join(out_dir, "model.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    with open(os.path.join(out_dir, "model.pkl"), "rb") as fh:
        blob = fh.read()
    if hashlib.sha256(blob).hexdigest() != meta.get("sha256"):
        raise ModelError("model.pkl does not match the sha256 in model.json: refusing to load it")
    return pickle.loads(blob), meta


def cli(argv=None):
    p = argparse.ArgumentParser(description="Fit and store the deployable ORB filter.")
    p.add_argument("--bars-dir", default=None)
    p.add_argument("--context", nargs="*", default=["SPY", "QQQ"])
    p.add_argument("--out", default="research/out/model")
    p.add_argument("--ledger", default="research/out/ledger.jsonl")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--strategy", default="orb", choices=ss.names())
    p.add_argument("--bar-seconds", type=int, default=300, help="bar size; != 300 needs --ticks-dir (recorded trades)")
    p.add_argument("--ticks-dir", help="recorded ticks (live_runner.py record) instead of --bars-dir")
    a = p.parse_args(argv)
    tm = o.load_pack()
    if a.ticks_dir:
        import tickbars
        rows, info = tickbars.load_rows(ss.get(a.strategy, a.bar_seconds), tm, a.ticks_dir)
    elif a.bar_seconds != 300:
        print("--bar-seconds other than 300 needs --ticks-dir (Alpaca has no free sub-5-minute history here)")
        return 2
    elif not a.bars_dir:
        print("give --bars-dir (5-minute bars) or --ticks-dir (recorded trades)")
        return 2
    else:
        rows, info = (o.load_dir(tm, a.bars_dir, tuple(a.context)) if a.strategy == "orb"
                      else ss.get(a.strategy).load_dir(tm, a.bars_dir, tuple(a.context)))
    try:
        cfg = bt.Config(latest_entry_slot=ss.scale(60, a.bar_seconds))
        dep = fit_deployable(rows, seed=a.seed, strategy=a.strategy, bar_seconds=a.bar_seconds, backtest_cfg=cfg)
    except ModelError as e:
        print("No model saved: %s" % e)
        return 1
    meta = save(dep, a.out, a.ledger)
    print("Saved %s (%s, calibration %s, threshold %.2f, validation Brier %.4f) trained through %s on %d symbols."
          % (a.out, meta["model"], meta["calibration"], meta["threshold"], meta["validation_brier"],
             meta["trained_through"], len(meta["symbols"])))
    print("Linked to ledger entry: %s (%s)" % (meta["ledger_ref"], meta["ledger_result"]))
    if not meta["ledger_ref"] or not str(meta["ledger_result"]).startswith("PASS"):
        print("The bot will NOT send paper orders with this model (no real-data walk-forward PASS to link to). "
              "Dry-run only.")
    return 0


if __name__ == "__main__":
    import model_store as _ms          # so the pickle records 'model_store.Deployable', not '__main__.Deployable'
    sys.exit(_ms.cli())
