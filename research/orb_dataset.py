#!/usr/bin/env python3
"""Phases 2-4: the ORB baseline, leakage-safe features and the triple-barrier dataset.

For every past session it finds the ORB candidate (the first 5-minute bar after the opening range that
CLOSES above the range high; long only), freezes features as of that bar's close, and labels the trade with
the triple barrier: 1 = target first, 0 = stop first, 2 = timeout at the flat-by bar. Barriers are read off
bar High / Low; a bar touching both counts as the STOP (pessimistic).

Leakage rule: a feature may use bars up to and including the trigger bar, plus COMPLETED earlier sessions.
The label uses only bars after the trigger. tests/test_orb_dataset.py recomputes every feature from bars cut
off at the trigger and demands identical numbers.

Reuses the day-trader pack's chart parsing (github.com/Jakeschincariol/day-trader-skill, trader-timemachine):
clone it next to this repo or set DAY_TRADER_DIR. Standard library only.
"""

import argparse
import csv
import datetime
import importlib.util
import json
import math
import os
import statistics
import sys

SESSION_BARS = 78
FEATURES = ["or_range_pct", "breakout_pct", "risk_pct", "trigger_vol_ratio", "rel_volume", "body_pct",
            "upper_wick_pct", "lower_wick_pct", "atr_pct", "ret_from_open_pct", "dist_vwap_pct",
            "gap_pct", "minutes_since_open", "weekday"]


def load_pack(name="trader-timemachine", file="timemachine.py"):
    roots = [os.environ.get("DAY_TRADER_DIR"), os.path.expanduser("~/.claude/skills"),
             os.path.join(os.getcwd(), "day-trader-skill"), os.path.join(os.getcwd(), "..", "day-trader-skill"),
             "/tmp/day-trader-skill"]
    for r in roots:
        p = r and os.path.join(r, name, file)
        if p and os.path.exists(p):
            spec = importlib.util.spec_from_file_location("pack_" + name.replace("-", "_"), p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit("cannot find the day-trader pack: clone github.com/Jakeschincariol/day-trader-skill "
                     "next to this repo or set DAY_TRADER_DIR")


def load_sessions(tm, path, symbol):
    res = tm.chart_result(tm.read_offline(path, symbol), symbol)
    _, bars, _ = tm.parse_bars(res, datetime.datetime(2100, 1, 1))     # naive far-future clock: keep every bar
    full, _short = tm.full_sessions(tm.split_sessions(bars), "9999-12-31")
    return full  # [(date, [78 bars])], oldest first


def load_dir(tm, directory, context=("SPY", "QQQ"), **kw):
    """Every <SYMBOL>.json[.gz] in a folder (as written by alpaca_bars.py) -> (rows, info).
    The context symbols are market-context features only; they are NOT traded."""
    files = {}
    for name in sorted(os.listdir(directory)):
        if name.endswith((".json", ".json.gz")):
            files[name.split(".")[0].upper()] = os.path.join(directory, name)
    if not files:
        raise SystemExit("orb_dataset.py: no .json / .json.gz bar files in %s" % directory)
    ctx_syms = [c.upper() for c in context if c.upper() in files]
    ctx = {c: {d: D for d, D in load_sessions(tm, files[c], c)} for c in ctx_syms}
    rows, per = [], {}
    for sym, path in files.items():
        if sym in ctx_syms:
            continue
        sessions = load_sessions(tm, path, sym)
        got = build(tm, sessions, context=ctx or None, symbol=sym, **kw)
        per[sym] = {"sessions": len(sessions), "candidates": len(got)}
        rows += got
    return rows, {"symbols": per, "context": ctx_syms}


# ---------------------------------------------------------------- features (as of the trigger bar)

def features_at(D, k, oh, ol, prior, bar_seconds=300):
    """Pure. Reads ONLY D[0..k] of today plus completed earlier sessions `prior` [(date, bars)]."""
    b = D[k]
    e = b["c"]
    upto = D[:k + 1]
    f = 300.0 / bar_seconds                       # 5-minute defaults expressed in this bar size (f == 1 for 5-minute bars)
    or_vols = [x["v"] for x in D[:max(1, round(3 * f))]]
    avg_or_v = (sum(or_vols) / len(or_vols)) if or_vols and sum(or_vols) else None
    cum_v = sum(x["v"] for x in upto)
    hist = [sum(x["v"] for x in bars[:k + 1]) for _, bars in prior[-20:]]
    rel = (cum_v / (sum(hist) / len(hist))) if hist and sum(hist) else None
    tp = [((x["h"] + x["l"] + x["c"]) / 3.0, x["v"]) for x in upto]
    vsum = sum(v for _, v in tp)
    vwap = (sum(p * v for p, v in tp) / vsum) if vsum else None
    window = upto[-max(1, round(14 * f)):]
    atr = sum((x["h"] - x["l"]) for x in window) / len(window)
    gap = None
    if prior:
        pc = prior[-1][1][-1]["c"]
        gap = (D[0]["o"] - pc) / pc * 100.0
    return {
        "or_range_pct": max(oh - ol, 0.0) / ol * 100.0,
        "breakout_pct": (e - oh) / oh * 100.0,
        "risk_pct": None,  # filled by the caller once the stop is known
        "trigger_vol_ratio": (b["v"] / avg_or_v) if avg_or_v else None,
        "rel_volume": rel,
        "body_pct": (b["c"] - b["o"]) / b["o"] * 100.0,
        "upper_wick_pct": (b["h"] - max(b["o"], b["c"])) / e * 100.0,
        "lower_wick_pct": (min(b["o"], b["c"]) - b["l"]) / e * 100.0,
        "atr_pct": atr / e * 100.0,
        "ret_from_open_pct": (e - D[0]["o"]) / D[0]["o"] * 100.0,
        "dist_vwap_pct": ((e - vwap) / vwap * 100.0) if vwap else None,
        "gap_pct": gap,
        "minutes_since_open": (k + 1) * bar_seconds / 60.0,
        "weekday": datetime.date.fromisoformat(b["date"]).weekday(),
    }


def context_features(context, date, k):
    """Market context (SPY, QQQ ...) as of the trigger bar: that symbol's bars 0..k of the SAME day only."""
    out = {}
    for sym, days in (context or {}).items():
        D = days.get(date)
        key = "ctx_%s_ret_pct" % sym.lower()
        out[key] = None if (not D or len(D) <= k or not D[0]["o"]) else (D[k]["c"] - D[0]["o"]) / D[0]["o"] * 100.0
    return out


# ---------------------------------------------------------------- the candidate and its triple-barrier label

def candidate(tm, date, D, prior, or_bars=3, last=76, stop_rule="or_mid", target_r=1.0, latest_entry=60,
              context=None):
    """The day's long ORB candidate with features and label, or None. last = the flat-by bar
    (76 = the bar that closes at 15:55). latest_entry = last bar allowed to trigger (60 ~ 14:30)."""
    oh, ol = tm.opening_range(D, or_bars)
    k = next((i for i in range(or_bars, latest_entry + 1) if D[i]["c"] > oh), None)
    if k is None:
        return None
    e = round(D[k]["c"], 2)
    stop = tm.rule_stop({"stop_rule": stop_rule, "stop_pct": 1.0}, oh, ol, e, True)
    risk = e - stop
    if risk <= 0:
        return None
    target = round(e + target_r * risk, 2)
    feats = features_at(D, k, oh, ol, prior)
    feats["risk_pct"] = risk / e * 100.0
    feats.update(context_features(context, date, k))
    label, exit_slot = 2, last
    for i in range(k + 1, last + 1):
        b = D[i]
        if b["l"] <= stop:          # stop first when one bar touches both
            label, exit_slot = 0, i
            break
        if b["h"] >= target:
            label, exit_slot = 1, i
            break
    R = target_r if label == 1 else -1.0 if label == 0 else (D[last]["c"] - e) / risk
    row = {"date": date, "entry_time": D[k]["start"], "entry_slot": k, "entry": e, "stop": stop,
           "target": target, "exit_slot": exit_slot, "label": label, "R": R, "risk": risk}
    row.update(feats)
    return row


def build(tm, sessions, context=None, symbol="?", **kw):
    rows = []
    for i, (d, D) in enumerate(sessions):
        r = candidate(tm, d, D, sessions[:i], context=context, **kw)
        if r:
            r["symbol"] = symbol
            rows.append(r)
    return rows


# ---------------------------------------------------------------- split with an embargo

def chrono_split(rows, fractions=(0.5, 0.15, 0.15, 0.2), embargo_days=1):
    """Chronological train / calibration / validation / locked-test split by WHOLE DAYS with `embargo_days`
    sessions dropped after each boundary. Every trade opens and closes inside one session, so no outcome
    window crosses a day and day-level splitting already purges overlapping labels."""
    dates = sorted({r["date"] for r in rows})
    n = len(dates)
    edges, acc = [], 0.0
    for f in fractions:
        acc += f
        edges.append(int(round(acc * n)))
    edges[-1] = n
    out, start = {}, 0
    for name, end in zip(("train", "calibration", "validation", "test"), edges):
        seg = dates[start:end]
        out[name] = set(seg if name == "train" else seg[embargo_days:])
        start = end
    return {name: [r for r in rows if r["date"] in days] for name, days in out.items()}


# ---------------------------------------------------------------- statistics

def net_R(r, cost_bps=0.0):
    """R after a round-trip cost in basis points of the entry price."""
    return r["R"] - (cost_bps / 1e4) * r["entry"] / r["risk"]


def feature_columns(rows):
    return FEATURES + sorted({k for r in rows for k in r if k.startswith("ctx_")})


def stats(rows, cost_bps=0.0):
    if not rows:
        return {"n": 0}
    Rs = [net_R(r, cost_bps) for r in rows]
    n = len(Rs)
    wins = [x for x in Rs if x > 0]
    losses = [x for x in Rs if x <= 0]
    eq, peak, dd = 0.0, 0.0, 0.0
    for x in Rs:
        eq += x
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    mean = statistics.mean(Rs)
    se = statistics.stdev(Rs) / math.sqrt(n) if n > 1 else None
    resolved = [r for r in rows if r["label"] in (0, 1)]
    return {
        "n": n, "target": sum(1 for r in rows if r["label"] == 1), "stop": sum(1 for r in rows if r["label"] == 0),
        "timeout": sum(1 for r in rows if r["label"] == 2),
        "target_first_rate": (sum(1 for r in resolved if r["label"] == 1) / len(resolved)) if resolved else None,
        "expectancy_R": round(mean, 3),
        "ci95_R": [round(mean - 1.96 * se, 3), round(mean + 1.96 * se, 3)] if se is not None else None,
        "profit_factor": round(sum(wins) / -sum(losses), 2) if losses and sum(losses) < 0 else None,
        "max_drawdown_R": round(dd, 2), "cost_bps": cost_bps,
    }


def show(title, s):
    if not s.get("n"):
        return "%-12s no trades" % title
    ci = s["ci95_R"]
    return ("%-12s n=%-3d target %d / stop %d / timeout %d | target-first %s | expectancy %+.3fR  95%% CI %s | PF %s | maxDD %.1fR"
            % (title, s["n"], s["target"], s["stop"], s["timeout"],
               "%.0f%%" % (100 * s["target_first_rate"]) if s["target_first_rate"] is not None else "n/a",
               s["expectancy_R"], "[%+.2f, %+.2f]" % tuple(ci) if ci else "n/a", s["profit_factor"], s["max_drawdown_R"]))


def write_outputs(rows, out):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    base = ["date", "symbol", "entry_time", "entry", "stop", "target", "label", "R"]
    cols = feature_columns(rows)
    with open(out + ".csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(base + cols)
        for r in rows:
            w.writerow([r.get(c) for c in base] + [r.get(c) for c in cols])
    with open(out + ".json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)


def main(argv=None):
    p = argparse.ArgumentParser(description="ORB baseline + triple-barrier dataset (blueprint phases 2-4).")
    p.add_argument("chart", nargs="?", help="a saved Yahoo-style 5-minute chart (.json/.json.gz)")
    p.add_argument("--bars-dir", help="a folder of <SYMBOL>.json.gz files from alpaca_bars.py (SPY/QQQ = context)")
    p.add_argument("--symbol", default="TSLA")
    p.add_argument("--target-r", type=float, default=1.0)
    p.add_argument("--stop", default="or_mid", choices=["or_mid", "or_low"])
    p.add_argument("--flat-by-bar", type=int, default=76, help="last bar held (76 = out at 15:55)")
    p.add_argument("--embargo-days", type=int, default=1)
    p.add_argument("--costs", default="0,2,5", help="round-trip cost in bps, comma separated")
    p.add_argument("--context", nargs="*", default=[], metavar="SYM=CHART")
    p.add_argument("--out", default="research/out/orb_dataset")
    a = p.parse_args(argv)
    tm = load_pack()
    kw = dict(last=a.flat_by_bar, stop_rule=a.stop, target_r=a.target_r)
    if a.bars_dir:
        rows, info = load_dir(tm, a.bars_dir, **kw)
        label = "%d symbols" % len(info["symbols"])
    elif a.chart:
        ctx = {}
        for item in a.context:
            sym, _, path = item.partition("=")
            ctx[sym.upper()] = {d: D for d, D in load_sessions(tm, path, sym)}
        rows = build(tm, load_sessions(tm, a.chart, a.symbol), context=ctx or None, symbol=a.symbol.upper(), **kw)
        label = a.symbol.upper()
    else:
        p.error("give a chart file or --bars-dir")
    if not rows:
        print("no ORB candidates found")
        return 1
    write_outputs(rows, a.out)
    print("%s: %d ORB candidates (%s to %s). Wrote %s.csv/.json" % (label, len(rows), rows[0]["date"], rows[-1]["date"], a.out))
    print("\nBaseline (ORB only, no AI), all candidates:")
    for c in [float(x) for x in a.costs.split(",")]:
        print("  " + show("cost %gbps" % c, stats(rows, c)))
    seg = chrono_split(rows, embargo_days=a.embargo_days)
    print("\nChronological split by whole days, %d-day embargo (test stays locked until the end):" % a.embargo_days)
    for name in ("train", "calibration", "validation", "test"):
        print("  " + show(name, stats(seg[name], 2.0)))
    print("\nA handful of trades cannot show an edge: read the confidence interval, not the mean. Not financial advice.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
