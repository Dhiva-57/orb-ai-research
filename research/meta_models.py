#!/usr/bin/env python3
"""Phases 5-6: baseline meta-models and probability calibration on the ORB triple-barrier dataset.

The strategy proposes the trade (a long ORB candidate). A model estimates P(target before stop). A threshold
turns that into take / skip. Everything is judged against "take every ORB candidate".

Models, simplest first: `rate` (training target-first rate), `logistic`, and a shallow regularised `forest`.

Rules the code enforces:
  * fit on TRAIN only (imputation and scaling included); timeouts are excluded from fitting but stay in every
    trading result;
  * probabilities are calibrated on the CALIBRATION segment only (Platt; isotonic only with >= 150 trades) and a
    calibrator is adopted only if it beats raw on VALIDATION Brier;
  * the threshold is chosen on VALIDATION, then applied ONCE to the locked TEST set;
  * a filter only ADDS VALUE if the day-clustered bootstrap 95% interval of
    (expectancy of trades taken - expectancy of all candidates) is above zero on TEST, >= 30 trades were taken
    and the filtered expectancy is positive after costs.

Needs numpy and scikit-learn.
"""

import argparse
import math
import random
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import orb_dataset as o

MIN_FIT = 30          # fewer resolved training trades than this: no model is fitted
MIN_TAKEN = 30        # fewer TEST trades taken than this can never "add value"
MIN_PLATT = 40
MIN_ISO = 150
GRID = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]


def binary(rows):
    return [r for r in rows if r["label"] in (0, 1)]


def matrix(rows, cols):
    return np.array([[np.nan if r.get(c) is None else float(r[c]) for c in cols] for r in rows], dtype=float)


class Rate:
    name = "rate"

    def fit(self, X, y):
        self.p = float(np.mean(y))
        return self

    def proba(self, X):
        return np.full(len(X), self.p)


class Sk:
    def __init__(self, name, est):
        self.name = name
        self.pipe = make_pipeline(SimpleImputer(strategy="median"), est)

    def fit(self, X, y):
        self.pipe.fit(X, y)
        return self

    def proba(self, X):
        return self.pipe.predict_proba(X)[:, 1]


def make_models(seed=7):
    return [
        Rate(),
        Sk("logistic", make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=1000))),
        Sk("forest", RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=10,
                                            max_features="sqrt", random_state=seed, n_jobs=1)),
    ]


def ml_metrics(y, p):
    out = {"n": int(len(y))}
    if len(y) == 0:
        return out
    p = np.clip(p, 1e-6, 1 - 1e-6)
    out["brier"] = round(float(brier_score_loss(y, p)), 4)
    out["logloss"] = round(float(log_loss(y, p, labels=[0, 1])), 4)
    out["auc"] = round(float(roc_auc_score(y, p)), 3) if len(set(y)) == 2 else None
    return out


# ---------------------------------------------------------------- calibration

def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p)).reshape(-1, 1)


class Platt:
    name = "platt"

    def fit(self, p, y):
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p), y)
        return self

    def __call__(self, p):
        return self.lr.predict_proba(_logit(p))[:, 1]


class Iso:
    name = "isotonic"

    def fit(self, p, y):
        self.ir = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99).fit(p, y)
        return self

    def __call__(self, p):
        return self.ir.predict(p)


class Raw:
    name = "none"

    def fit(self, p, y):
        return self

    def __call__(self, p):
        return np.asarray(p, dtype=float)


def ece(y, p, bins=10):
    """Expected calibration error: bin-weighted gap between predicted and observed rate."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if len(y) == 0:
        return None
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return round(float(total), 4)


def calibration_table(y, p, bins=5):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    out, edges = [], np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            out.append({"range": "%.1f-%.1f" % (lo, hi), "n": int(m.sum()),
                        "predicted": round(float(p[m].mean()), 3), "observed": round(float(y[m].mean()), 3)})
    return out


def pick_calibrator(p_cal, y_cal, p_val, y_val):
    """Fit candidates on CALIBRATION only; choose among them by Brier on VALIDATION. Raw is always a candidate,
    so calibration is adopted only if it helps."""
    cands = [Raw()]
    if len(y_cal) >= MIN_PLATT and len(set(y_cal)) == 2:
        cands.append(Platt().fit(p_cal, y_cal))
        if len(y_cal) >= MIN_ISO:
            cands.append(Iso().fit(p_cal, y_cal))
    if len(y_val) == 0 or len(set(y_val)) < 2:
        return cands[0]
    return min(cands, key=lambda c: brier_score_loss(y_val, np.clip(c(p_val), 1e-6, 1 - 1e-6)))


# ---------------------------------------------------------------- threshold, significance

def choose_threshold(rows, probs, cost_bps, min_taken, backtest_cfg=None):
    best = None

    for th in GRID:
        take = [p >= th for p in probs]

        if backtest_cfg is not None:
            import backtest as bt

            res = bt.replay(rows, backtest_cfg, take)
            stats = bt.summarize(res, backtest_cfg)

            n = stats["trades"]
            if n < min_taken:
                continue

            e = stats.get("expectancy_R")
            if e is None:
                continue

        else:
            taken = [r for r, selected in zip(rows, take) if selected]

            if len(taken) < min_taken:
                continue

            e = float(np.mean([o.net_R(r, cost_bps) for r in taken]))
            n = len(taken)

        if best is None or e > best[1]:
            best = (th, e, n)

    return best

def bootstrap_gain(rows, probs, th, cost_bps, B=2000, seed=11):
    """95% interval of (mean net R of trades taken) - (mean net R of ALL candidates). Resamples whole DAYS:
    trades on one day share the market and are not independent, so a per-trade bootstrap is too narrow."""
    rng = random.Random(seed)
    by_day = {}
    for r, p in zip(rows, probs):
        by_day.setdefault(r["date"], []).append((o.net_R(r, cost_bps), p >= th))
    days = list(by_day.values())
    nd = len(days)
    diffs = []
    for _ in range(B):
        picked = [days[rng.randrange(nd)] for _ in range(nd)]
        allr = [x for d in picked for x, _t in d]
        taken = [x for d in picked for x, t in d if t]
        if not taken:
            continue
        diffs.append(sum(taken) / len(taken) - sum(allr) / len(allr))
    if len(diffs) < B // 2:
        return None
    diffs.sort()
    return [round(diffs[int(0.025 * len(diffs))], 3), round(diffs[int(0.975 * len(diffs))], 3)]


def bootstrap_replay_gain(rows, take, cfg, B=2000, seed=11):
    """Paired day-cluster bootstrap of actual replay outcomes.

    Returns the 95% CI for AI expectancy minus ORB-only expectancy,
    and the AI strategy's own expectancy CI. Trades are grouped by
    date and the same sampled dates are used for both strategies.
    """
    import backtest as bt

    if not rows or len(take) != len(rows):
        return {"gain_ci95_R": None, "ai_ci95_R": None}

    base = bt.replay(rows, cfg)["trades"]
    ai = bt.replay(rows, cfg, take)["trades"]

    base_by_day, ai_by_day = {}, {}
    for trade in base:
        base_by_day.setdefault(trade["date"], []).append(trade["R_net"])
    for trade in ai:
        ai_by_day.setdefault(trade["date"], []).append(trade["R_net"])

    days = sorted({r["date"] for r in rows})
    if not days or not ai:
        return {"gain_ci95_R": None, "ai_ci95_R": None}

    rng = random.Random(seed)
    gains, ai_means = [], []

    for _ in range(B):
        picked = [days[rng.randrange(len(days))] for _ in days]
        base_r = [x for day in picked for x in base_by_day.get(day, [])]
        ai_r = [x for day in picked for x in ai_by_day.get(day, [])]

        if not ai_r:
            continue

        ai_mean = sum(ai_r) / len(ai_r)
        ai_means.append(ai_mean)

        if base_r:
            gains.append(ai_mean - sum(base_r) / len(base_r))

    def interval(values):
        if len(values) < B // 2:
            return None
        values.sort()
        return [round(values[int(0.025 * len(values))], 3),
                round(values[int(0.975 * len(values))], 3)]

    return {"gain_ci95_R": interval(gains), "ai_ci95_R": interval(ai_means)}


def run(rows, cost_bps=2.0, embargo_days=1, seed=7, fractions=(0.5, 0.15, 0.15, 0.2), calibrate=True, segments=None, backtest_cfg=None):
    """Fit on TRAIN, calibrate on CALIBRATION, pick model + threshold on VALIDATION, score the locked TEST once."""
    cols = o.feature_columns(rows)
    seg = segments if segments is not None else o.chrono_split(rows, fractions=fractions, embargo_days=embargo_days)
    tr, ca, va, te = binary(seg["train"]), binary(seg["calibration"]), seg["validation"], seg["test"]
    report = {"counts": {k: len(v) for k, v in seg.items()}, "features": len(cols), "models": {}}
    report["execution_aware"] = backtest_cfg is not None
    if backtest_cfg is not None:
        import backtest as bt
        report["orb_only_test"] = bt.summarize(bt.replay(te, backtest_cfg), backtest_cfg)
    else:
        report["orb_only_test"] = o.stats(te, cost_bps)
    report["_test_rows"] = te        # even when no model can be fitted, the ORB-only baseline still runs
    if len(tr) < MIN_FIT or len({r["label"] for r in tr}) < 2:
        report["note"] = "only %d resolved training trades: no model is fitted (needs %d)" % (len(tr), MIN_FIT)
        return report
    Xtr, ytr = matrix(tr, cols), np.array([r["label"] for r in tr])
    Xca, yca = matrix(ca, cols), np.array([r["label"] for r in ca])
    Xva, Xte = matrix(va, cols), matrix(te, cols)
    bva = [i for i, r in enumerate(va) if r["label"] in (0, 1)]
    bte = [i for i, r in enumerate(te) if r["label"] in (0, 1)]
    yva_b = np.array([va[i]["label"] for i in bva])
    yte_b = np.array([te[i]["label"] for i in bte])
    min_taken_val = max(5, MIN_TAKEN // 3)
    for m in make_models(seed):
        m.fit(Xtr, ytr)
        raw_va, raw_te = m.proba(Xva), m.proba(Xte)
        if m.name == "rate" or not calibrate or len(ca) == 0:
            cal = Raw()
        else:
            cal = pick_calibrator(m.proba(Xca), yca, raw_va[bva] if bva else np.array([]), yva_b)
        pva, pte = cal(raw_va), cal(raw_te)
        entry = {"calibration": cal.name,
                 "validation_ml": ml_metrics(yva_b, pva[bva]) if bva else {},
                 "test_ml": ml_metrics(yte_b, pte[bte]) if bte else {}}
        if bte:
            entry["test_ece_raw"] = ece(yte_b, raw_te[bte])
            entry["test_ece"] = ece(yte_b, pte[bte])
            entry["test_calibration_table"] = calibration_table(yte_b, pte[bte])
        pick = choose_threshold(va, pva, cost_bps, min_taken_val, backtest_cfg=backtest_cfg) if len(va) else None
        if pick is None or m.name == "rate":
            entry["threshold"] = None
            entry["verdict"] = ("NOT PROVEN: no threshold keeps enough validation trades" if m.name != "rate"
                                else "baseline (every candidate gets the same probability)")
        else:
            th = pick[0]
            take = [pr >= th for pr in pte]
            taken = [r for r, selected in zip(te, take) if selected]
            entry["threshold"] = th

            if backtest_cfg is not None:
                import backtest as bt
                entry["test_taken"] = bt.summarize(
                    bt.replay(te, backtest_cfg, take), backtest_cfg)
                ci_info = bootstrap_replay_gain(te, take, backtest_cfg)
                gain = ci_info["gain_ci95_R"]
                entry["standalone_expectancy_ci95_R"] = ci_info["ai_ci95_R"]
                trade_count = entry["test_taken"]["trades"]
                expectancy = entry["test_taken"].get("expectancy_R")
                ai_ci = ci_info["ai_ci95_R"]
                entry["standalone_profitable"] = (
                    trade_count > 0 and ai_ci is not None and ai_ci[0] > 0)
            else:
                entry["test_taken"] = o.stats(taken, cost_bps)
                gain = bootstrap_gain(te, pte, th, cost_bps) if taken else None
                trade_count = len(taken)
                expectancy = entry["test_taken"]["expectancy_R"]
                ci = entry["test_taken"]["ci95_R"]
                entry["standalone_profitable"] = (
                    bool(taken) and ci is not None and ci[0] > 0)

            entry["gain_vs_all_ci95_R"] = gain
            ok = (trade_count >= MIN_TAKEN and gain is not None and gain[0] > 0
                  and expectancy is not None and expectancy > 0)
            entry["verdict"] = (
                "ADDS VALUE over ORB-only on the locked test (standalone profit %s)"
                % ("also clear" if entry["standalone_profitable"] else "NOT clear: its own interval crosses zero")
            ) if ok else ("NOT PROVEN: %s" % ("only %d executed trades (need %d)" % (trade_count, MIN_TAKEN)
                                              if trade_count < MIN_TAKEN else "the gain over ORB-only is not clearly above zero"))
        entry["_probs_test"] = [float(x) for x in pte]
        report["models"][m.name] = entry
    return report


def show(report):
    c = report["counts"]

    def stats_text(stats):
        if not report.get("execution_aware"):
            return o.show("", stats).strip()
        expectancy = stats.get("expectancy_R")
        expectancy_text = "%.3f" % expectancy if expectancy is not None else "n/a"
        return ("trades=%d return=%+.2f%% expectancy=%s R maxDD=%.2f%%"
                % (stats.get("trades", 0), stats.get("return_pct", 0.0),
                   expectancy_text, stats.get("max_drawdown_pct", 0.0)))

    lines = ["Rows per segment: train %d, calibration %d, validation %d, test %d. Features: %d."
             % (c["train"], c["calibration"], c["validation"], c["test"], report["features"]),
             "ORB only on the locked test: " + stats_text(report["orb_only_test"])]
    if report.get("note"):
        lines.append(report["note"])
        return "\n".join(lines)
    for name, e in report["models"].items():
        lines += ["", "[%s]" % name,
                  "  calibration: %s | test ECE raw %s -> %s" % (e["calibration"], e.get("test_ece_raw"), e.get("test_ece"))]
        for k in ("validation_ml", "test_ml"):
            if e.get(k):
                lines.append("  %-14s %s" % (k, e[k]))
        if e.get("threshold") is not None:
            lines.append("  threshold %.2f (chosen on validation)" % e["threshold"])
            lines.append("  test, trades taken: " + stats_text(e["test_taken"]))
            lines.append("  gain vs taking everything, 95%% interval: %s R" % (e.get("gain_vs_all_ci95_R"),))
            if report.get("execution_aware"):
                lines.append("  AI standalone expectancy, 95%% interval: %s R"
                             % (e.get("standalone_expectancy_ci95_R"),))
        lines.append("  verdict: %s" % e["verdict"])
    return "\n".join(lines)


# ---------------------------------------------------------------- synthetic data with a KNOWN answer

def datetime_days(n):
    import datetime
    d, out = datetime.date(2024, 1, 1), []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += datetime.timedelta(days=1)
    return out


def synthetic(n_days=400, per_day=3, edge=0.0, seed=1):
    """Candidate rows with the real schema. P(target first) depends on one feature only when edge > 0:
    edge=0 is pure noise (every candidate wins 46% of the time); edge>0 plants a real signal."""
    rng = random.Random(seed)
    rows = []
    for d in datetime_days(n_days):
        for _ in range(per_day):
            f = {name: rng.gauss(0, 1) for name in o.FEATURES}
            p = 1.0 / (1.0 + math.exp(-(math.log(0.46 / 0.54) + edge * f["trigger_vol_ratio"])))
            win = rng.random() < p
            es = rng.randrange(3, 40)
            rows.append(dict(f, date=d, symbol="S%02d" % rng.randrange(30), entry=100.0, risk=1.0, stop=99.0,
                             target=101.0, entry_slot=es, exit_slot=es + rng.randrange(1, 30),
                             label=1 if win else 0, R=1.0 if win else -1.0))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description="Baseline meta-models + calibration (blueprint phases 5-6).")
    p.add_argument("chart", nargs="?", help="saved 5-minute chart (omit with --synthetic or --bars-dir)")
    p.add_argument("--bars-dir")
    p.add_argument("--symbol", default="TSLA")
    p.add_argument("--synthetic", choices=["noise", "edge"], help="generated data with a known answer")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cost-bps", type=float, default=2.0)
    a = p.parse_args(argv)
    if a.synthetic:
        rows = synthetic(edge=1.0 if a.synthetic == "edge" else 0.0, seed=a.seed)
        print("SYNTHETIC %s data, %d candidates (the true answer is known)" % (a.synthetic.upper(), len(rows)))
    else:
        tm = o.load_pack()
        if a.bars_dir:
            rows, _ = o.load_dir(tm, a.bars_dir)
        elif a.chart:
            rows = o.build(tm, o.load_sessions(tm, a.chart, a.symbol), symbol=a.symbol.upper())
        else:
            p.error("give a chart file, --bars-dir or --synthetic")
        print("%d ORB candidates" % len(rows))
    print(show(run(rows, cost_bps=a.cost_bps, seed=a.seed)))
    print("\nNot financial advice.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
