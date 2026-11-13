#!/usr/bin/env python3
"""Honest daily-direction check, built from the same 5-minute bars the rest of the project uses (no extra data).

Question it answers: on YOUR tickers, is next-day direction predictable at all, out of sample? Pattern of the Stock AI
Predictor audit (see PREDICTOR_AUDIT.md): the old model scored ~100% only because it was handed the answer. This module is
the leak-proof version:
  * features of day t use data <= t only (stationary: returns, ratios, RSI, range, gap, volume ratio; no price levels)
  * the target is day t+1's direction; a model trained at the close of day t0 uses only rows whose label (close t+1) is
    already known by t0
  * walk-forward (expanding window, refit every `step` days), judged against two naive rivals: the commoner direction
    and the training base rate, with 95% intervals on the paired differences
  * the verdict is EDGE only if BOTH accuracy and Brier beat the naive rival with the whole interval above zero over at
    least 500 out-of-sample days; anything else is "NO EDGE DEMONSTRATED" (which is the expected answer for most tickers)

    python3 research/daily_signal.py --bars-dir research/bars [--symbols AAPL MSFT]

Also provides p_up_asof(), the as-of-yesterday P(up today) used by the *_daily strategy variants (strategies_daily.py).
numpy + scikit-learn only.
"""

import argparse
import math
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

WARMUP = 20            # first row with every feature defined
MIN_TRAIN = 150        # training rows needed before a model is trusted at all
LOOKBACK = 300         # p_up_asof trains on at most this many most-recent days (live and backtest must see the same window)
NAMES = ["ret1", "ret5", "ret10", "ret20", "vol10", "vol20", "close_vs_ma20", "ma5_vs_ma20", "rsi14", "range_pct",
         "gap", "close_loc", "vol_ratio", "atr_pct"]


def _shift(x, k):
    out = np.full(len(x), np.nan)
    if k < len(x):
        out[k:] = x[:len(x) - k]
    return out


def _rmean(x, n):
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        cs = np.cumsum(np.insert(x, 0, 0.0))
        out[n - 1:] = (cs[n:] - cs[:-n]) / n
    return out


def _rstd(x, n):
    m, m2 = _rmean(x, n), _rmean(x * x, n)
    return np.sqrt(np.maximum(m2 - m * m, 0.0) * n / (n - 1))


def features(o, h, l, c, v):
    """(n, len(NAMES)) matrix; row t uses bars <= t only (a prefix computation: changing later bars cannot change it)."""
    o, h, l, c, v = (np.asarray(a, dtype=float) for a in (o, h, l, c, v))
    prev = np.r_[c[0], c[:-1]]
    ret1 = c / prev - 1.0
    d = np.diff(c, prepend=c[0])
    g, ls = _rmean(np.clip(d, 0, None), 14), _rmean(np.clip(-d, 0, None), 14)
    rsi = np.where(ls > 0, 100 - 100 / (1 + g / np.where(ls > 0, ls, 1)), 100.0) - 50.0
    rng = np.maximum(h - l, 1e-12)
    tr = np.maximum.reduce([h - l, np.abs(h - prev), np.abs(l - prev)])
    vm = _rmean(v, 20)
    ma20 = _rmean(c, 20)
    cols = [ret1, c / _shift(c, 5) - 1, c / _shift(c, 10) - 1, c / _shift(c, 20) - 1, _rstd(ret1, 10), _rstd(ret1, 20),
            c / ma20 - 1, _rmean(c, 5) / ma20 - 1, rsi / 50.0, (h - l) / c, o / prev - 1, (c - l) / rng - 0.5,
            np.where(vm > 0, v / np.where(vm > 0, vm, 1) - 1, 0.0), _rmean(tr, 14) / c]
    return np.column_stack(cols)


def _fit(X, y):
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=0.1, max_iter=500).fit(sc.transform(X), y)
    return lambda Z: m.predict_proba(sc.transform(Z))[:, 1]


def p_up_asof(o, h, l, c, v):
    """P(the NEXT day closes up), from completed days only; None until there is enough history."""
    o, h, l, c, v = (np.asarray(a, dtype=float)[-LOOKBACK:] for a in (o, h, l, c, v))
    n = len(c)
    if n < WARMUP + MIN_TRAIN + 2:
        return None
    F = features(o, h, l, c, v)
    t = np.arange(WARMUP, n - 1)
    y = (c[t + 1] > c[t]).astype(int)
    if len(set(y)) < 2 or not np.isfinite(F[t]).all() or not np.isfinite(F[n - 1]).all():
        return None
    return float(_fit(F[t], y)(F[[n - 1]])[0])


def _ci(diffs):
    d = np.asarray(diffs, dtype=float)
    m = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")
    return m, m - 1.96 * se, m + 1.96 * se


def evaluate(o, h, l, c, v, min_train=250, step=20, min_oos=500):
    """Walk-forward verdict for ONE series of daily bars. Pure and deterministic."""
    o, h, l, c, v = (np.asarray(a, dtype=float) for a in (o, h, l, c, v))
    n = len(c)
    F = features(o, h, l, c, v)
    P, Y, B, BP, R = [], [], [], [], []
    for t0 in range(max(min_train, WARMUP + MIN_TRAIN), n - 1, step):
        tr = np.arange(WARMUP, t0)                       # rows whose label (close t+1 <= close t0) is known at t0
        y_tr = (c[tr + 1] > c[tr]).astype(int)
        if len(set(y_tr)) < 2:
            continue
        pred = _fit(F[tr], y_tr)
        te = np.arange(t0, min(t0 + step, n - 1))
        P += list(pred(F[te]))
        Y += list((c[te + 1] > c[te]).astype(int))
        B += [int(y_tr.mean() >= 0.5)] * len(te)
        BP += [float(y_tr.mean())] * len(te)
        R += list(c[te + 1] / c[te] - 1)
    out = {"oos_days": len(Y), "verdict": "NO EDGE DEMONSTRATED", "why": []}
    if len(Y) < 30:
        out["why"].append("fewer than 30 out-of-sample days")
        return out
    P, Y, B, BP, R = (np.asarray(a, dtype=float) for a in (P, Y, B, BP, R))
    hit_m, hit_b = ((P > 0.5) == (Y == 1)).astype(float), (B == Y).astype(float)
    br_m, br_b = (P - Y) ** 2, (BP - Y) ** 2
    acc_d, acc_lo, acc_hi = _ci(hit_m - hit_b)
    br_d, br_lo, br_hi = _ci(br_b - br_m)                # positive = model has the lower (better) Brier score
    long = P > 0.5
    out.update(accuracy=round(float(hit_m.mean()) * 100, 2), baseline_accuracy=round(float(hit_b.mean()) * 100, 2),
               accuracy_edge_pts=round(acc_d * 100, 2), accuracy_edge_ci95_pts=[round(acc_lo * 100, 2), round(acc_hi * 100, 2)],
               brier_improvement=round(br_d, 5), brier_improvement_ci95=[round(br_lo, 5), round(br_hi, 5)],
               mean_next_day_ret_pct_when_p_up=round(float(R[long].mean()) * 100, 3) if long.any() else None,
               mean_next_day_ret_pct_all_days=round(float(R.mean()) * 100, 3))
    if len(Y) < min_oos:
        out["why"].append("only %d out-of-sample days (need %d for a verdict)" % (len(Y), min_oos))
    elif acc_lo > 0 and br_lo > 0:
        out["verdict"] = "EDGE (investigate further)"
    else:
        out["why"].append("accuracy edge CI %s pts, Brier improvement CI %s: not both clearly above zero"
                          % (out["accuracy_edge_ci95_pts"], out["brier_improvement_ci95"]))
    return out


# ---------------------------------------------------------------- sessions adapter (5-minute bars -> daily bars)

def daily_from_sessions(sessions):
    """[(date, [bar dicts])] -> o, h, l, c, v arrays (one row per completed session)."""
    o = [D[0]["o"] for _, D in sessions]
    h = [max(b["h"] for b in D) for _, D in sessions]
    l = [min(b["l"] for b in D) for _, D in sessions]
    c = [D[-1]["c"] for _, D in sessions]
    v = [sum(b["v"] for b in D) for _, D in sessions]
    return np.array(o), np.array(h), np.array(l), np.array(c), np.array(v)


def p_up_from_sessions(prior):
    if len(prior) < WARMUP + MIN_TRAIN + 2:
        return None
    return p_up_asof(*daily_from_sessions(prior[-LOOKBACK:]))


def main(argv=None):
    import orb_dataset as o_
    p = argparse.ArgumentParser(description="Honest walk-forward check: is next-day direction predictable on your tickers?")
    p.add_argument("--bars-dir", required=True)
    p.add_argument("--symbols", nargs="*")
    a = p.parse_args(argv)
    tm = o_.load_pack()
    files = {n.split(".")[0].upper(): os.path.join(a.bars_dir, n) for n in sorted(os.listdir(a.bars_dir)) if n.endswith((".json", ".json.gz"))}
    syms = [s.upper() for s in (a.symbols or files)]
    flagged = 0
    for s in syms:
        if s not in files:
            print("%-6s no bar file" % s)
            continue
        r = evaluate(*daily_from_sessions(o_.load_sessions(tm, files[s], s)))
        flagged += r["verdict"].startswith("EDGE")
        print("%-6s %-22s oos_days=%-4d acc %s%% vs baseline %s%%  edge %s pts CI %s  %s" % (
            s, r["verdict"], r["oos_days"], r.get("accuracy"), r.get("baseline_accuracy"), r.get("accuracy_edge_pts"),
            r.get("accuracy_edge_ci95_pts"), "; ".join(r["why"])))
    print("\n%d of %d flagged. With 95%% intervals, about 2-3%% of edge-free tickers get flagged by luck: one flag among many "
          "tickers is NOT evidence. Trust only a flag that repeats on later data." % (flagged, len(syms)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
