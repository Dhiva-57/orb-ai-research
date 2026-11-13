#!/usr/bin/env python3
"""Phase 8: walk-forward validation of the whole ORB + AI + risk pipeline.

One train/validate/test split can be lucky. Here the entire pipeline is repeated over rolling windows and ONLY the
out-of-sample test periods are combined:

    window k:  train | gap | calibration | gap | validation | gap | TEST_k      (window k+1 starts `test` days later)

Inside every window the model is fitted on its own train days, calibrated on its own calibration days, and its
model choice and threshold come from its own validation days. TEST_k is touched once. The combined out-of-sample
candidates are then replayed through the same risk engine and cost model as phase 7, for ORB-only and ORB+AI.
A window with no usable model trades NOTHING for the AI strategy (reported, never hidden).
"""

import argparse
import sys
from dataclasses import asdict

import backtest as bt
import meta_models as mm
import orb_dataset as o


def window_plan(dates, train, cal, val, test, embargo=1):
    """Pure. Layout train E cal E val E test, stepping forward by `test` days. Returns a list of {segment: dates}."""
    total = train + cal + val + test + 3 * embargo
    out, s = [], 0
    while s + total <= len(dates):
        i, seg = s, {}
        for name, n in (("train", train), ("calibration", cal), ("validation", val), ("test", test)):
            seg[name] = dates[i:i + n]
            i += n + embargo
        out.append(seg)
        s += test
    return out


def pick(rep):
    """The window's model: lowest VALIDATION Brier among fitted models that produced a threshold."""
    cand = [(n, e) for n, e in rep.get("models", {}).items() if n != "rate" and e.get("threshold") is not None]
    return min(cand, key=lambda ne: ne[1]["validation_ml"].get("brier", 9)) if cand else None


def walk_forward(rows, cfg, train=120, cal=30, val=30, test=40, embargo=1, seed=7, cost_bps=2.0):
    dates = sorted({r["date"] for r in rows})
    plan = window_plan(dates, train, cal, val, test, embargo)
    if not plan:
        need = train + cal + val + test + 3 * embargo
        return {"error": "not enough data: %d candidate days, a single window needs %d" % (len(dates), need),
                "days": len(dates), "needed": need}
    oos_rows, oos_take, windows = [], [], []
    for k, seg_dates in enumerate(plan):
        seg = {name: [r for r in rows if r["date"] in set(ds)] for name, ds in seg_dates.items()}
        rep = mm.run(rows, cost_bps=cost_bps, seed=seed + k, segments=seg, backtest_cfg=cfg)
        te = rep["_test_rows"]
        chosen = pick(rep)
        if chosen:
            name, e = chosen
            take = [p >= e["threshold"] for p in e["_probs_test"]]
            info = {"model": name, "threshold": e["threshold"], "calibration": e["calibration"],
                    "test_auc": e["test_ml"].get("auc")}
        else:
            take, info = [False] * len(te), {"model": None, "threshold": None, "calibration": None, "test_auc": None}
        base = bt.summarize(bt.replay(te, cfg), cfg) if te else {"trades": 0}
        ai = bt.summarize(bt.replay(te, cfg, take), cfg) if te else {"trades": 0}
        windows.append(dict(info, k=k + 1, test_from=seg_dates["test"][0], test_to=seg_dates["test"][-1],
                            candidates=len(te), ai_takes=sum(take), orb_expectancy_R=base.get("expectancy_R"),
                            ai_expectancy_R=ai.get("expectancy_R"), orb_return_pct=base.get("return_pct"),
                            ai_return_pct=ai.get("return_pct")))
        oos_rows += te
        oos_take += take
    return {"windows": windows, "oos_candidates": len(oos_rows),
            "orb_only": bt.summarize(bt.replay(oos_rows, cfg), cfg),
            "orb_ai": bt.summarize(bt.replay(oos_rows, cfg, oos_take), cfg),
            "gain_ci95_R": mm.bootstrap_replay_gain(oos_rows, oos_take, cfg)["gain_ci95_R"]
            if any(oos_take) else None,
            "shape": {"train": train, "calibration": cal, "validation": val, "test": test, "embargo": embargo},
            "days": len(dates)}


def wf_gate(res, min_trades=30, min_windows=3, min_win_share=0.6):
    """Only says whether a human should look. Never approves trading."""
    if res.get("error"):
        return "FAIL", [res["error"]]
    ai, base, w = res["orb_ai"], res["orb_only"], res["windows"]
    reasons = []
    if len(w) < min_windows:
        reasons.append("only %d windows (need %d)" % (len(w), min_windows))
    if ai["trades"] < min_trades:
        reasons.append("only %d out-of-sample AI trades (need %d)" % (ai["trades"], min_trades))
    ci = res.get("gain_ci95_R")
    if ci is None or ci[0] <= 0:
        reasons.append("the gain over ORB-only is not clearly above zero (95%% interval %s)" % (ci,))
    if ai.get("expectancy_R", -1) <= 0:
        reasons.append("net expectancy is not positive after costs")
    if ai["return_pct"] <= base["return_pct"]:
        reasons.append("no better than ORB-only")
    if ai.get("sharpe") is None or ai["sharpe"] <= 0:
        reasons.append("Sharpe is not positive")
    comparable = [x for x in w if x["ai_expectancy_R"] is not None and x["orb_expectancy_R"] is not None]
    if comparable:
        share = sum(1 for x in comparable if x["ai_expectancy_R"] > x["orb_expectancy_R"]) / len(comparable)
        if share < min_win_share:
            reasons.append("AI beat ORB-only in only %.0f%% of windows (need %.0f%%): one lucky period, not a pattern"
                           % (100 * share, 100 * min_win_share))
    return ("PASS: send to human review" if not reasons else "FAIL"), reasons


def show(res):
    if res.get("error"):
        return res["error"]
    L = ["Walk-forward: %d windows over %d candidate days; shape %s" % (len(res["windows"]), res["days"], res["shape"]),
         "%-3s %-23s %5s %5s %-9s %5s %7s %9s %9s" % ("#", "test period", "cand", "AI", "model", "thr", "AUC", "ORB E[R]", "AI E[R]")]
    for w in res["windows"]:
        L.append("%-3d %-23s %5d %5d %-9s %5s %7s %9s %9s" % (
            w["k"], "%s..%s" % (w["test_from"][5:], w["test_to"][5:]), w["candidates"], w["ai_takes"], w["model"] or "-",
            "%.2f" % w["threshold"] if w["threshold"] is not None else "-", w["test_auc"] if w["test_auc"] is not None else "-",
            w["orb_expectancy_R"] if w["orb_expectancy_R"] is not None else "-",
            w["ai_expectancy_R"] if w["ai_expectancy_R"] is not None else "-"))
    for k, title in (("orb_only", "ORB only"), ("orb_ai", "ORB + AI")):
        s = res[k]
        L.append("COMBINED OOS  %-8s trades %-4d return %+.2f%%  maxDD %.2f%%  win %s  expectancy %s R  PF %s  Sharpe %s"
                 % (title, s["trades"], s["return_pct"], s["max_drawdown_pct"], s.get("win_rate"), s.get("expectancy_R"),
                    s.get("profit_factor"), s.get("sharpe")))
    L.append("Gain of AI over taking everything, day-clustered 95%% interval: %s R" % (res.get("gain_ci95_R"),))
    return "\n".join(L)


def main(argv=None):
    p = argparse.ArgumentParser(description="Walk-forward validation of ORB + AI + risk (blueprint phase 8).")
    p.add_argument("charts", nargs="*", help="SYMBOL=chart.json.gz files (or use --bars-dir / --synthetic)")
    p.add_argument("--bars-dir", help="a folder of <SYMBOL>.json.gz files from alpaca_bars.py")
    p.add_argument("--ticks-dir", help="a folder of recorded trades (live_runner.py record) for sub-5-minute bars")
    p.add_argument("--bar-seconds", type=int, default=300, help="bar size; != 300 needs --ticks-dir")
    p.add_argument("--context", nargs="*", default=["SPY", "QQQ"], help="symbols in --bars-dir used only as market context")
    p.add_argument("--synthetic", choices=["noise", "edge"])
    p.add_argument("--strategy", default="orb", help="strategy name from strategies.py (default orb)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--train", type=int, default=120)
    p.add_argument("--cal", type=int, default=30)
    p.add_argument("--val", type=int, default=30)
    p.add_argument("--test", type=int, default=40)
    p.add_argument("--embargo", type=int, default=1)
    p.add_argument("--ledger", default="research/out/ledger.jsonl")
    p.add_argument("--no-ledger", action="store_true")
    a = p.parse_args(argv)
    import strategies as _ss
    cfg = bt.Config(latest_entry_slot=_ss.scale(60, a.bar_seconds))          # the risk engine counts slots of THIS bar size
    if a.synthetic:
        rows = mm.synthetic(n_days=700, edge=2.0 if a.synthetic == "edge" else 0.0, seed=a.seed)
        src = "SYNTHETIC %s (the true answer is known)" % a.synthetic.upper()
    elif a.ticks_dir:
        import strategies as ss
        import tickbars
        tm = o.load_pack()
        rows, info = tickbars.load_rows(ss.get(a.strategy, a.bar_seconds), tm, a.ticks_dir)
        src = "%d symbols of recorded ticks from %s, %ds bars (no market context)" % (len(info["symbols"]), a.ticks_dir, a.bar_seconds)
    elif a.bar_seconds != 300:
        p.error("--bar-seconds other than 300 needs --ticks-dir")
    elif a.bars_dir:
        import strategies as ss
        tm = o.load_pack()
        rows, info = (o.load_dir(tm, a.bars_dir, tuple(a.context)) if a.strategy == "orb"
                      else ss.get(a.strategy).load_dir(tm, a.bars_dir, tuple(a.context)))
        src = "%d symbols from %s (context: %s)" % (len(info["symbols"]), a.bars_dir, ", ".join(info["context"]) or "none")
    elif a.charts:
        tm = o.load_pack()
        rows = []
        for item in a.charts:
            sym, _, path = item.partition("=")
            rows += o.build(tm, o.load_sessions(tm, path, sym), symbol=sym.upper())
        src = "%d chart(s)" % len(a.charts)
    else:
        p.error("give --bars-dir, SYMBOL=chart files or --synthetic")
    res = walk_forward(rows, cfg, a.train, a.cal, a.val, a.test, a.embargo, seed=a.seed)
    verdict = wf_gate(res)
    print("%s, %d candidates" % (src, len(rows)))
    print(show(res))
    print("Gate: %s%s" % (verdict[0], (" - " + "; ".join(verdict[1])) if verdict[1] else ""))
    if not a.no_ledger and not res.get("error"):
        eid = bt.ledger_append(a.ledger, {"source": src, "kind": "walk-forward", "strategy": a.strategy, "bar_seconds": a.bar_seconds, "config": asdict(cfg),
                                          "shape": res["shape"], "result": verdict[0], "reasons": verdict[1],
                                          "orb_only": res["orb_only"], "orb_ai": res["orb_ai"],
                                          "gain_ci95_R": res["gain_ci95_R"], "windows": res["windows"]})
        print("Logged %s to %s" % (eid, a.ledger))
    print("\nSlippage is modelled; partial fills, halts and rejected orders are not. A PASS only means a human should "
          "look. Not financial advice.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
