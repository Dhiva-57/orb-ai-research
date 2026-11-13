#!/usr/bin/env python3
"""Phase 7: ORB + AI filter + hard risk engine, in an execution-aware backtest.

    ORB candidate -> AI probability >= threshold -> RISK ENGINE (veto) -> sizing -> fills

The risk engine is independent of the model: it sees only the proposed trade and the account state and can say
NO TRADE whatever the AI thinks. ORB-only and ORB+AI are replayed on the SAME locked test period with the same
costs and risk rules, so the only difference is the filter.

Execution model (deliberately pessimistic): entries slip against you; a stop fills worse than its price; a target
is a resting limit order that fills at its price (slightly optimistic); a timeout exits at the flat-by close with
slippage; decisions are made in time order and never read a later trade; sizing uses realised P&L only.
Not modelled (real results will be worse): partial fills, rejected orders, halts, latency, borrow, taxes.
"""

import argparse
import datetime
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass, asdict

import orb_dataset as o


@dataclass
class Config:
    equity: float = 10000.0
    risk_pct: float = 0.5               # of equity risked per trade, at the stop
    max_position_pct: float = 25.0      # one position's notional, % of equity
    max_trades_day: int = 3
    max_open: int = 2
    daily_loss_pct: float = 1.5         # stop opening trades once today's losses + open risk reach this
    slip_bps: float = 1.0
    stop_slip_bps: float = 3.0
    commission_per_share: float = 0.0
    kill_switch: bool = False
    min_reward_risk: float = 1.0
    latest_entry_slot: int = 60         # no entries after ~14:30
    allow_duplicate_symbol: bool = False


def risk_per_share(cfg, cand):
    """Estimated loss per share at the stop, including entry/stop slippage and round-trip commission."""
    entry_fill = cand["entry"] * (1 + cfg.slip_bps / 1e4)
    stop_fill = cand["stop"] * (1 - cfg.stop_slip_bps / 1e4)
    return entry_fill - stop_fill + 2 * cfg.commission_per_share


def risk_check(cfg, acct, cand, shares):
    """Pure. (ok, reason). acct = {equity, start_equity, trades_today, open: [{symbol, exit_slot, risk_usd,
    notional}], realized_today}."""
    if cfg.kill_switch:
        return False, "kill_switch"
    if cand["entry_slot"] > cfg.latest_entry_slot:
        return False, "too_late"
    if cand["risk"] <= 0 or cand["entry"] <= 0:
        return False, "bad_prices"
    if (cand["target"] - cand["entry"]) / cand["risk"] < cfg.min_reward_risk - 1e-9:
        return False, "reward_risk"
    if acct["trades_today"] >= cfg.max_trades_day:
        return False, "max_trades"
    if len(acct["open"]) >= cfg.max_open:
        return False, "max_open"
    if not cfg.allow_duplicate_symbol and any(p["symbol"] == cand["symbol"] for p in acct["open"]):
        return False, "duplicate_symbol"
    if shares < 1:
        return False, "size_zero"
    loss_today = max(0.0, -acct["realized_today"])
    open_risk = sum(p["risk_usd"] for p in acct["open"])
    if loss_today + open_risk + shares * risk_per_share(cfg, cand) > cfg.daily_loss_pct / 100.0 * acct["start_equity"] + 1e-9:
        return False, "daily_loss_limit"
    if sum(p["notional"] for p in acct["open"]) + shares * cand["entry"] > acct["equity"] + 1e-9:
        return False, "buying_power"
    return True, "ok"


def size(cfg, equity, cand):
    """Whole shares so the loss AT THE STOP (with entry and stop slippage) is <= risk_pct of equity, capped by the
    position limit. 0 if one share is already too much (a too-wide stop gives no trade, not a smaller bet)."""
    entry_fill = cand["entry"] * (1 + cfg.slip_bps / 1e4)
    risk_ps = risk_per_share(cfg, cand)
    if risk_ps <= 0:
        return 0
    return max(0, min(math.floor(cfg.risk_pct / 100.0 * equity / risk_ps),
                      math.floor(cfg.max_position_pct / 100.0 * equity / entry_fill)))


def pnl(cfg, cand, shares):
    entry_fill = cand["entry"] * (1 + cfg.slip_bps / 1e4)
    if cand["label"] == 1:
        exit_fill = cand["target"]
    elif cand["label"] == 0:
        exit_fill = cand["stop"] * (1 - cfg.stop_slip_bps / 1e4)
    else:
        exit_fill = (cand["entry"] + cand["R"] * cand["risk"]) * (1 - cfg.slip_bps / 1e4)
    return shares * (exit_fill - entry_fill) - 2 * cfg.commission_per_share * shares


def replay(rows, cfg, take=None):
    """Replay candidates in time order. take[i] False = the AI vetoed row i before the risk engine."""
    n = len(rows)
    take = [True] * n if take is None else take
    order = sorted(range(n), key=lambda i: (rows[i]["date"], rows[i]["entry_slot"], rows[i].get("symbol", "")))
    by_day = {}
    for i in order:
        by_day.setdefault(rows[i]["date"], []).append(i)
    equity = cfg.equity
    trades, refused, ai_vetoed = [], {}, 0
    day_returns, peak, max_dd = [], equity, 0.0
    curve = [(None, equity)]
    for day in sorted(by_day):
        start = equity
        acct = {"equity": equity, "start_equity": start, "trades_today": 0, "open": [], "realized_today": 0.0}
        for i in by_day[day]:
            c = dict(rows[i])
            c.setdefault("symbol", "?")
            still = []
            for p in acct["open"]:                 # settle positions that exited by this entry bar (realised only)
                if p["exit_slot"] <= c["entry_slot"]:
                    acct["realized_today"] += p["pnl"]
                    acct["equity"] += p["pnl"]
                else:
                    still.append(p)
            acct["open"] = still
            if not take[i]:
                ai_vetoed += 1
                continue
            shares = size(cfg, acct["equity"], c)
            ok, why = risk_check(cfg, acct, c, shares)
            if not ok:
                refused[why] = refused.get(why, 0) + 1
                continue
            p = pnl(cfg, c, shares)
            acct["trades_today"] += 1
            acct["open"].append({"symbol": c["symbol"], "exit_slot": c.get("exit_slot", 10 ** 6),
                                 "risk_usd": shares * risk_per_share(cfg, c), "notional": shares * c["entry"], "pnl": p})
            trades.append({"date": day, "symbol": c["symbol"], "shares": shares, "pnl": p,
                           "R_net": p / (shares * c["risk"]), "label": c["label"]})
        for p in acct["open"]:                      # everything is flat by the close
            acct["realized_today"] += p["pnl"]
            acct["equity"] += p["pnl"]
        equity = acct["equity"]
        day_returns.append((equity - start) / start)
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)
        curve.append((day, equity))
    return {"final_equity": equity, "trades": trades, "refused": refused, "ai_vetoed": ai_vetoed,
            "day_returns": day_returns, "max_drawdown": max_dd, "curve": curve}


def summarize(res, cfg):
    t = res["trades"]
    out = {"trades": len(t), "final_equity": round(res["final_equity"], 2),
           "return_pct": round((res["final_equity"] / cfg.equity - 1) * 100, 2),
           "max_drawdown_pct": round(res["max_drawdown"] * 100, 2),
           "refused_by_risk": res["refused"], "vetoed_by_ai": res["ai_vetoed"]}
    if t:
        wins = sum(x["pnl"] for x in t if x["pnl"] > 0)
        loss = -sum(x["pnl"] for x in t if x["pnl"] <= 0)
        out.update(win_rate=round(sum(1 for x in t if x["pnl"] > 0) / len(t), 3),
                   expectancy_R=round(statistics.mean(x["R_net"] for x in t), 3),
                   profit_factor=round(wins / loss, 2) if loss > 0 else None)
    dr = res["day_returns"]
    if len(dr) > 2 and statistics.pstdev(dr) > 0:
        out["sharpe"] = round(statistics.mean(dr) / statistics.pstdev(dr) * math.sqrt(252), 2)
        down = [x for x in dr if x < 0]
        if len(down) > 1 and statistics.pstdev(down) > 0:
            out["sortino"] = round(statistics.mean(dr) / math.sqrt(sum(x * x for x in down) / len(dr)) * math.sqrt(252), 2)
    return out


def gate(ai, base, verdict, min_trades=30):
    """Single-split gate. Never 'approves' trading: it only says whether a human should look."""
    reasons = []
    if not str(verdict).startswith("ADDS VALUE"):
        reasons.append("the model's own locked-test verdict is not 'ADDS VALUE'")
    if ai["trades"] < min_trades:
        reasons.append("only %d trades (need %d)" % (ai["trades"], min_trades))
    if ai.get("expectancy_R", -1) <= 0:
        reasons.append("net expectancy is not positive after costs")
    if ai["return_pct"] <= base["return_pct"]:
        reasons.append("no better than ORB-only")
    if ai.get("sharpe") is None or ai["sharpe"] <= 0:
        reasons.append("Sharpe is not positive")
    return ("PASS: send to human review" if not reasons else "FAIL"), reasons


def ledger_append(path, entry):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = 1
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            n = sum(1 for _ in fh) + 1
    entry = dict({"id": "EXP-%03d" % n, "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}, **entry)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry["id"]


def experiment(rows, cfg, seed=7, cost_bps=2.0):
    """Train / calibrate / pick the threshold as in phases 5-6, then replay the LOCKED TEST period for ORB-only
    and ORB+AI under identical risk rules."""
    import meta_models as mm
    rep = mm.run(rows, cost_bps=cost_bps, seed=seed, backtest_cfg=cfg)
    te = rep.get("_test_rows", [])
    out = {"split": rep["counts"], "orb_only": summarize(replay(te, cfg), cfg) if te else None}
    cand = [(n, e) for n, e in rep.get("models", {}).items() if n != "rate" and e.get("threshold") is not None]
    if not cand:
        out["orb_ai"] = None
        out["note"] = rep.get("note") or "no model produced a usable threshold on validation"
        out["gate"] = ("FAIL", [out["note"]])
        return out, rep
    name, e = min(cand, key=lambda ne: ne[1]["validation_ml"].get("brier", 9))   # chosen on VALIDATION only
    take = [p >= e["threshold"] for p in e["_probs_test"]]
    out.update(model=name, threshold=e["threshold"], calibration=e["calibration"], model_verdict=e["verdict"])
    out["orb_ai"] = summarize(replay(te, cfg, take), cfg)
    out["gate"] = gate(out["orb_ai"], out["orb_only"], e["verdict"])
    return out, rep


def show(out):
    L = ["Rows per segment: %s" % out["split"]]
    for k, title in (("orb_only", "ORB only"), ("orb_ai", "ORB + AI")):
        s = out.get(k)
        if not s:
            L.append("%-9s not run" % title)
            continue
        L.append("%-9s trades %-3d return %+.2f%%  maxDD %.2f%%  win %s  expectancy %s R  PF %s  Sharpe %s  "
                 "| risk engine refused %s, AI vetoed %d"
                 % (title, s["trades"], s["return_pct"], s["max_drawdown_pct"], s.get("win_rate"), s.get("expectancy_R"),
                    s.get("profit_factor"), s.get("sharpe"), s["refused_by_risk"] or "nothing", s["vetoed_by_ai"]))
    if out.get("model"):
        L.append("Model: %s, %s calibration, threshold %.2f (all chosen without the test set)"
                 % (out["model"], out["calibration"], out["threshold"]))
    g = out["gate"]
    L.append("Gate: %s%s" % (g[0], (" - " + "; ".join(g[1])) if g[1] else ""))
    return "\n".join(L)


def main(argv=None):
    import meta_models as mm
    p = argparse.ArgumentParser(description="ORB + AI + risk engine backtest (blueprint phase 7).")
    p.add_argument("chart", nargs="?", help="saved 5-minute chart; omit with --synthetic or --bars-dir")
    p.add_argument("--bars-dir")
    p.add_argument("--symbol", default="TSLA")
    p.add_argument("--synthetic", choices=["noise", "edge"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--equity", type=float, default=Config.equity)
    p.add_argument("--risk-pct", type=float, default=Config.risk_pct)
    p.add_argument("--slip-bps", type=float, default=Config.slip_bps)
    p.add_argument("--stop-slip-bps", type=float, default=Config.stop_slip_bps)
    p.add_argument("--kill-switch", action="store_true")
    p.add_argument("--ledger", default="research/out/ledger.jsonl")
    p.add_argument("--no-ledger", action="store_true")
    a = p.parse_args(argv)
    cfg = Config(equity=a.equity, risk_pct=a.risk_pct, slip_bps=a.slip_bps, stop_slip_bps=a.stop_slip_bps,
                 kill_switch=a.kill_switch)
    if a.synthetic:
        rows = mm.synthetic(edge=2.0 if a.synthetic == "edge" else 0.0, seed=a.seed)
        src = "SYNTHETIC %s (the true answer is known)" % a.synthetic.upper()
    else:
        tm = o.load_pack()
        if a.bars_dir:
            rows, _ = o.load_dir(tm, a.bars_dir)
        elif a.chart:
            rows = o.build(tm, o.load_sessions(tm, a.chart, a.symbol), symbol=a.symbol.upper())
        else:
            p.error("give a chart file, --bars-dir or --synthetic")
        src = "real bars"
    out, _ = experiment(rows, cfg, seed=a.seed)
    print("%s, %d candidates" % (src, len(rows)))
    print(show(out))
    if not a.no_ledger:
        eid = ledger_append(a.ledger, {"source": src, "config": asdict(cfg), "result": out["gate"][0],
                                       "reasons": out["gate"][1], "split": out["split"], "orb_only": out["orb_only"],
                                       "orb_ai": out["orb_ai"], "model": out.get("model"),
                                       "threshold": out.get("threshold"), "calibration": out.get("calibration")})
        print("Logged %s to %s" % (eid, a.ledger))
    print("\nSlippage is modelled; partial fills, halts and rejected orders are not. A PASS only means a human should look. "
          "Not financial advice.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
