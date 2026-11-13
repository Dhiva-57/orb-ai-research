#!/usr/bin/env python3
"""Paper-vs-backtest comparison: for the days the bot ran, replay the SAME strategy on the (refreshed) historical bars and
line the two up, trade by trade.

    python3 research/alpaca_bars.py ... --force          # refresh bars so they include the paper days
    python3 research/compare.py --out research/out/bot --bars-dir research/bars [--strategy orb momentum]

Reads research/out/bot/trades.jsonl (one line per finished paper trade) and decisions.jsonl. For each paper trade it reports
signal parity (did history produce the same entry/stop/target?), entry slippage in bps, paper R vs backtest R, and the exit
reason. For each historical candidate the bot did NOT trade on a day it was running, it reports why (AI veto, risk veto,
stale data, missed trigger...). It lists CANDIDATE explanations for any gap; it never claims a cause it cannot see.

R here is GROSS (no commissions or modelled costs on either side) so the two are comparable. A trade whose exit price could
not be recovered is counted but excluded from R statistics. Fewer than --min-trades trades means "too early to say".
"""

import argparse
import json
import os
import sys

import journal as jn


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def paper_r(t):
    fill, ex, risk = _f(t.get("fill_price")), _f(t.get("exit_price")), (_f(t.get("entry")) or 0) - (_f(t.get("stop")) or 0)
    if fill is None or ex is None or risk <= 0:
        return None
    return (ex - fill) / risk


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def r3(x):
    return None if x is None else round(x, 3)


def compare(trades, bt_rows, decisions, run_dates, date_of, min_trades=30):
    """Pure. trades: trades.jsonl records. bt_rows: {(strategy, symbol, date): labelled row}. decisions: decisions.jsonl
    records. run_dates: set of ET dates the bot ticked on. date_of(decision) -> ET date string."""
    per, traded = [], set()
    for t in trades:
        if _f(t.get("fill_price")) is None:
            continue                                                    # never filled: not a trade
        strat = t.get("strategy", "orb")
        key = (strat, t["symbol"], t["date"])
        traded.add(key)
        bt = bt_rows.get(key)
        entry = _f(t["entry"])
        row = {"strategy": strat, "symbol": t["symbol"], "date": t["date"], "exit_reason": t.get("exit_reason"),
               "slippage_bps": r3((_f(t["fill_price"]) - entry) / entry * 1e4), "paper_R": r3(paper_r(t)),
               "bt_in_history": bt is not None, "bt_R": r3(bt["R"]) if bt else None, "bt_label": bt["label"] if bt else None}
        if bt:
            row["signal_ok"] = all(abs(_f(t[k]) - bt[k]) <= 0.011 for k in ("entry", "stop", "target"))
            row["delta_R"] = r3(row["paper_R"] - row["bt_R"]) if row["paper_R"] is not None else None
        per.append(row)

    by_day = {}
    for d in decisions:
        if d.get("symbol"):
            by_day.setdefault((d["symbol"], date_of(d)), []).append(d)
    missed = []
    for key, bt in bt_rows.items():
        strat, sym, date = key
        if date not in run_dates or key in traded:
            continue
        ds = [d for d in by_day.get((sym, date), []) if d.get("strategy", strat) == strat]
        reason = (ds[-1].get("why") or ds[-1]["kind"]) if ds else "no_decision_logged"
        missed.append({"strategy": strat, "symbol": sym, "date": date, "reason": reason, "bt_R": r3(bt["R"]), "bt_label": bt["label"]})

    matched = [p for p in per if p["bt_in_history"]]
    both = [p for p in matched if p["delta_R"] is not None]
    rep = {"n_paper_trades": len(per), "n_with_history": len(matched), "n_with_R_both": len(both),
           "signal_parity_pct": r3(100.0 * sum(p["signal_ok"] for p in matched) / len(matched)) if matched else None,
           "paper_trades_without_history": len(per) - len(matched),
           "mean_slippage_bps": r3(mean([p["slippage_bps"] for p in per])),
           "paper_expectancy_R": r3(mean([p["paper_R"] for p in both])),
           "backtest_expectancy_R_same_trades": r3(mean([p["bt_R"] for p in both])),
           "mean_delta_R": r3(mean([p["delta_R"] for p in both])),
           "exit_reasons": _count(p["exit_reason"] for p in per),
           "missed_candidates": len(missed), "missed_by_reason": _count(m["reason"] for m in missed),
           "missed_backtest_expectancy_R": r3(mean([m["bt_R"] for m in missed])),
           "missed_ai_veto_backtest_expectancy_R": r3(mean([m["bt_R"] for m in missed if m["reason"] == "ai_veto"])),
           "trades": per, "missed": missed, "notes": []}
    n = rep["notes"]
    if len(both) < min_trades:
        n.append("Only %d trades with a usable R on both sides (need %d): too early to conclude anything. Keep collecting." % (len(both), min_trades))
    if rep["signal_parity_pct"] is not None and rep["signal_parity_pct"] < 90:
        n.append("Signal parity is %.0f%%: live and historical bars often disagree on the entry/stop/target. Candidate causes: "
                 "different feed or consolidation of the latest bar, a bar that changed after it was first seen, volume differences." % rep["signal_parity_pct"])
    if rep["paper_trades_without_history"]:
        n.append("%d paper trade(s) have NO historical counterpart: the bot traded a setup the backtest did not produce "
                 "(or the bars file does not cover that day: refresh it)." % rep["paper_trades_without_history"])
    if rep["mean_slippage_bps"] is not None and rep["mean_slippage_bps"] > 5:
        n.append("Mean entry slippage is %.1f bps (the backtest assumes 1): the market order after the bar closes fills worse than the bar's close." % rep["mean_slippage_bps"])
    if rep["mean_delta_R"] is not None and len(both) >= min_trades and rep["mean_delta_R"] < -0.1:
        n.append("Paper trails the backtest by %.2f R per trade. Candidate causes, in the order to check them: entry slippage, "
                 "exits (see exit_reasons), signal mismatches, then luck (compare against the spread of the R values)." % -rep["mean_delta_R"])
    unknown = sum(1 for p in per if p["paper_R"] is None)
    if unknown:
        n.append("%d trade(s) have no recoverable exit price and are excluded from the R statistics." % unknown)
    return rep


def _count(items):
    out = {}
    for i in items:
        out[i] = out.get(i, 0) + 1
    return out


def render(rep):
    L = ["PAPER vs BACKTEST", "  paper trades: %d (with history %d, usable R on both sides %d)" %
         (rep["n_paper_trades"], rep["n_with_history"], rep["n_with_R_both"]),
         "  signal parity: %s%%   mean entry slippage: %s bps" % (rep["signal_parity_pct"], rep["mean_slippage_bps"]),
         "  expectancy (gross R): paper %s vs backtest %s on the same trades (delta %s)" %
         (rep["paper_expectancy_R"], rep["backtest_expectancy_R_same_trades"], rep["mean_delta_R"]),
         "  exits: %s" % rep["exit_reasons"],
         "  historical candidates not traded: %d %s" % (rep["missed_candidates"], rep["missed_by_reason"])]
    if rep["missed_ai_veto_backtest_expectancy_R"] is not None:
        L.append("  AI-vetoed candidates averaged %s R in the backtest (negative = the veto helped)" % rep["missed_ai_veto_backtest_expectancy_R"])
    L += ["  - " + x for x in rep["notes"]] or []
    return "\n".join(L)


def main(argv=None):
    import orb_dataset as o
    import strategies as ss
    p = argparse.ArgumentParser(description="Compare paper trades with a backtest replay of the same days.")
    p.add_argument("--out", default="research/out/bot")
    p.add_argument("--bars-dir", required=True)
    p.add_argument("--strategy", nargs="*", help="default: every strategy seen in trades.jsonl")
    p.add_argument("--context", nargs="*", default=["SPY", "QQQ"])
    p.add_argument("--min-trades", type=int, default=30)
    a = p.parse_args(argv)
    J = jn.Journal(a.out)
    trades, decisions = jn.read_jsonl(J.trades_path), jn.read_jsonl(J.decisions_path)
    if not trades and not decisions:
        print("nothing to compare: %s has no trades.jsonl / decisions.jsonl yet" % a.out)
        return 1
    tm = o.load_pack()

    def date_of(rec):
        import datetime
        return tm.to_eastern(datetime.datetime.fromisoformat(rec["at"]).replace(tzinfo=None)).date().isoformat()

    run_dates = {date_of(d) for d in decisions if d.get("at")}
    strategies = a.strategy or sorted({t.get("strategy", "orb") for t in trades}) or ["orb"]
    bt_rows = {}
    for name in strategies:
        rows, _info = (o.load_dir(tm, a.bars_dir, tuple(a.context)) if name == "orb"
                       else ss.get(name).load_dir(tm, a.bars_dir, tuple(a.context)))
        for r in rows:
            if r["date"] in run_dates:
                bt_rows[(name, r["symbol"], r["date"])] = r
    rep = compare([t for t in trades if t.get("strategy", "orb") in strategies], bt_rows, decisions, run_dates, date_of, a.min_trades)
    jn.save_json_atomic(os.path.join(a.out, "compare.json"), rep)
    print(render(rep))
    print("\nDetails: %s" % os.path.join(a.out, "compare.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
