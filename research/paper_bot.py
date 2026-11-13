#!/usr/bin/env python3
"""Alpaca PAPER bot, phase 1: long-only 15-minute ORB + AI filter + the backtest's own risk engine.

    python3 research/paper_bot.py tick      [--paper-orders]     # run every 5 minutes through the US session
    python3 research/paper_bot.py status
    python3 research/paper_bot.py flatten                       # close the bot's own positions now
    python3 research/paper_bot.py kill | unkill                 # stop / allow NEW entries (positions still managed)

PAPER ONLY: the broker class is hard-wired to Alpaca's paper host. Default is DRY-RUN: decisions are logged, nothing is
sent. Orders need --paper-orders AND all of: a loaded model, the newest REAL-data walk-forward ledger entry is a PASS,
that entry is the one the model was built against, and the model is not stale.

Every tick: market open? -> reconcile + manage own positions (the trade manager) -> entries only if allowed:
  no kill file, not within the flatten window, not an early-close day, account not blocked.
Entry per symbol: fresh 5-minute bars -> today's FIRST close above the opening range must be the LATEST completed bar
(otherwise 'missed_trigger') -> features frozen at that bar -> AI probability >= threshold -> size + risk engine (veto)
-> price-drift guard -> write intent to disk -> bracket order (market buy + take-profit + stop). Orders are never retried.

Known gaps (real results will be worse): the backtest enters at the trigger bar's close plus 1 bp, the bot sends a
market order after that bar completes; paper fills are kinder than real fills; nothing was run against live Alpaca.
"""

import argparse
import datetime
import json
import os
import sys

import alpaca_bars as ab
import backtest as bt
import broker as bk
import journal as jn
import model_store as ms
import orb_dataset as o
import router as rt
import strategies as ss
import trade_manager as tmgr

SESSION_BARS, GRID, OPEN_MIN = 78, 300, 9 * 60 + 30


class BotConfig:
    def __init__(self, symbols, bars_dir="research/bars", out_dir="research/out/bot", model_dir="research/out/model",
                 ledger="research/out/ledger.jsonl", context=("SPY", "QQQ"), risk=None, or_bars=3, target_r=1.0,
                 stop_rule="or_mid", max_gap_bars=6, max_drift_bps=15.0, flatten_minutes=5, max_hist_age_days=6,
                 max_model_age_days=45, max_hold_minutes=None, strategy=None):
        self.symbols = [s.upper() for s in symbols]
        self.bars_dir, self.out_dir, self.model_dir, self.ledger = bars_dir, out_dir, model_dir, ledger
        self.context = [c.upper() for c in context]
        self.risk = risk or bt.Config()
        self.or_bars, self.target_r, self.stop_rule = or_bars, target_r, stop_rule
        self.max_gap_bars, self.max_drift_bps, self.flatten_minutes = max_gap_bars, max_drift_bps, flatten_minutes
        self.max_hist_age_days, self.max_model_age_days, self.max_hold_minutes = max_hist_age_days, max_model_age_days, max_hold_minutes
        self.strategy = strategy or ss.ORB(or_bars=or_bars, target_r=target_r, stop_rule=stop_rule, last=SESSION_BARS - 2,
                                           latest_entry=self.risk.latest_entry_slot)
        if risk is None:
            self.risk.latest_entry_slot = self.strategy.latest_entry          # the risk engine counts slots of THIS bar size

    @property
    def kill_file(self):
        return os.path.join(self.out_dir, "KILL")


# ---------------------------------------------------------------- data

def utc_iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def session_open_utc(tm, today):
    d = datetime.date.fromisoformat(today)
    return tm.eastern_to_utc(datetime.datetime(d.year, d.month, d.day, 9, 30))


def live_bars(raw, tm, now_utc, today, max_gap=6):
    """Today's COMPLETED regular-session 5-minute bars -> (list of bar dicts slot 0..n-1 | [], last_real_slot, why).
    Interior gaps are filled flat (zero volume) exactly as the dataset does; no opening bar or too many gaps -> []."""
    slots = {}
    for b in raw:
        try:
            ts, px = ab.epoch(b["t"]), [float(b[k]) for k in ("o", "h", "l", "c")]
        except (KeyError, ValueError, TypeError):
            continue
        if ts % GRID:
            continue
        start = tm.utc_of(ts)
        local = tm.to_eastern(start)
        mins = local.hour * 60 + local.minute - OPEN_MIN
        if local.date().isoformat() != today or mins < 0 or mins >= SESSION_BARS * 5:
            continue
        if start + datetime.timedelta(seconds=GRID) > now_utc:
            continue                                   # bar not complete yet
        slots[mins // 5] = {"date": today, "slot": mins // 5, "start": local.strftime("%H:%M"),
                            "o": px[0], "h": px[1], "l": px[2], "c": px[3], "v": float(b.get("v") or 0)}
    if 0 not in slots:
        return [], None, "no_opening_bar"
    last = max(slots)
    missing = [i for i in range(last + 1) if i not in slots]
    if len(missing) > max_gap:
        return [], last, "gappy_data"
    out, prev = [], None
    for i in range(last + 1):
        if i in slots:
            prev = slots[i]
            out.append(slots[i])
        else:
            out.append(dict(prev, slot=i, o=prev["c"], h=prev["c"], l=prev["c"], c=prev["c"], v=0.0))
    return out, last, "ok"


def load_history(tm, bars_dir, symbols, n=25):
    hist = {}
    for s in symbols:
        for ext in (".json.gz", ".json"):
            p = os.path.join(bars_dir, s + ext)
            if os.path.exists(p):
                hist[s] = o.load_sessions(tm, p, s)[-n:]
                break
    return hist


def live_candidate(tm, symbol, today, D, prior, context, bcfg):
    """(row | None, reason): delegated to the strategy, so the bot can never disagree with the backtest."""
    return bcfg.strategy.live_candidate(tm, symbol, today, D, prior, context)


# ---------------------------------------------------------------- gate

def gate_ok(bcfg, meta, today, name=None):
    name = name or bcfg.strategy.name
    bs = bcfg.strategy.bar_seconds
    wf = ms.latest_real_walkforward(bcfg.ledger, name, bs)
    if wf is None:
        return False, "no real-data walk-forward result for strategy %s in the ledger" % name
    if meta.get("strategy", "orb") != name:
        return False, "model was built for strategy %s, not %s" % (meta.get("strategy", "orb"), name)
    if meta.get("bar_seconds", 300) != bs:
        return False, "model was built for %ss bars, the strategy runs %ss" % (meta.get("bar_seconds", 300), bs)
    if not str(wf.get("result", "")).startswith("PASS"):
        return False, "newest real-data walk-forward (%s) is %s" % (wf.get("id"), wf.get("result"))
    if meta.get("ledger_ref") != wf.get("id"):
        return False, "model was built against %s but the newest walk-forward is %s: rebuild the model" % (meta.get("ledger_ref"), wf.get("id"))
    try:
        age = (datetime.date.fromisoformat(today) - datetime.date.fromisoformat(meta["trained_through"])).days
    except (KeyError, ValueError):
        return False, "model has no trained_through date"
    if age > bcfg.max_model_age_days:
        return False, "model is %d days old (limit %d)" % (age, bcfg.max_model_age_days)
    return True, "ok"


def parse_utc(text):
    dt = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
    return (dt - dt.utcoffset()).replace(tzinfo=None) if dt.tzinfo else dt


# ---------------------------------------------------------------- one tick

def tick(broker, tm, bcfg, model, meta, now_utc, paper_orders=False, history=None, router=None, bars_source=None):
    """bars_source(symbol) -> (bars, last_real_slot, why): live bars from the tick aggregator (live_runner.py). Required for
    strategies on bars other than 5 minutes; without it the bot reads Alpaca's 5-minute REST bars."""
    if not getattr(broker, "paper", False):
        raise SystemExit("refusing to run: the broker is not a paper broker")
    grid = (router.options[0].strategy if router else bcfg.strategy).bar_seconds
    if grid != GRID and bars_source is None:
        raise ValueError("strategies on %ds bars need a live bar source (live_runner.py); the REST path is 5-minute only" % grid)
    J = jn.Journal(bcfg.out_dir)
    today = tm.to_eastern(now_utc).date().isoformat()
    clock = broker.clock()
    if not clock.get("is_open"):
        J.decision("skip", why="market_closed")
        return {"skipped": "market_closed"}
    close_utc = parse_utc(clock["next_close"])
    close_et = tm.to_eastern(close_utc)
    early_close = close_et.hour * 60 + close_et.minute != 16 * 60
    flatten = now_utc >= close_utc - datetime.timedelta(minutes=bcfg.flatten_minutes)

    state = J.load_state()
    if state.get("date") != today:
        state.update(date=today, start_equity=float(broker.account()["equity"]), trades_today=0, closed=[], dry_seen=[])
    state.setdefault("positions", {})
    carry = [s for s, p in state["positions"].items() if p.get("date") != today]
    if carry:
        J.decision("overnight_carry", symbols=carry)
    J.save_state(state)

    summary = {"date": today, "flatten": flatten, "entries": []}
    mgr = tmgr.manage(broker, J, state, now_utc, flatten=flatten or bool(carry), max_hold_minutes=bcfg.max_hold_minutes)
    J.save_state(state)
    summary["manager"] = mgr

    blocked = None
    if os.path.exists(bcfg.kill_file):
        blocked = "kill_switch"
    elif flatten or carry:
        blocked = "flatten_window"
    elif early_close:
        blocked = "early_close_day"
    acct = broker.account()
    if blocked is None and (acct.get("trading_blocked") or acct.get("account_blocked")):
        blocked = "account_blocked"
    options = router.options if router else [rt.Option(bcfg.strategy, model, meta)]
    gates = {}
    for opt in options:                               # each strategy needs ITS OWN PASS tied to ITS OWN model
        if not paper_orders:
            gates[opt.name] = (False, "dry_run")
        else:
            gates[opt.name] = (False, "no model") if opt.model is None else gate_ok(bcfg, opt.meta, today, opt.name)
    summary["orders_enabled"] = any(g[0] for g in gates.values())
    summary["gate"] = list(gates.values())[0][1] if len(gates) == 1 else {n: g[1] for n, g in gates.items()}
    if blocked:
        J.decision("no_entries", why=blocked)
        summary["no_entries"] = blocked
        return summary

    equity = float(acct["equity"])
    held = {p["symbol"]: p for p in broker.positions()}
    own = state["positions"]
    open_list = [{"symbol": s, "exit_slot": 10 ** 6, "risk_usd": p["shares"] * p["risk"], "notional": p["shares"] * p["entry"]}
                 for s, p in own.items()]
    for s, p in held.items():                          # foreign positions still count against the limits
        if s not in own:
            open_list.append({"symbol": s, "exit_slot": 10 ** 6, "risk_usd": 0.0,
                              "notional": abs(float(p["qty"])) * float(p.get("current_price") or p["avg_entry_price"])})
    own_unreal = sum(float(held[s]["unrealized_pl"]) for s in own if s in held)
    risk_acct = {"equity": equity, "start_equity": state["start_equity"], "trades_today": state["trades_today"],
                 "open": open_list, "realized_today": (equity - state["start_equity"]) - own_unreal}

    history = history if history is not None else {}
    ctx_syms = [c for c in bcfg.context if any(opt.model is not None and ("ctx_%s_ret_pct" % c.lower()) in opt.model.cols
                                               for opt in options)]
    context, open_utc = {}, session_open_utc(tm, today)
    for c in (ctx_syms if bars_source is None else []):          # no market-context features for tick bars
        try:
            D, _l, _w = live_bars(broker.bars_5m(c, utc_iso(open_utc), utc_iso(now_utc)), tm, now_utc, today, bcfg.max_gap_bars)
            if D:
                context[c] = {today: D}
        except bk.BrokerError as e:
            J.decision("context_missing", symbol=c, error=str(e)[:200])
    expected_last = int((now_utc - open_utc).total_seconds() // grid) - 1

    for sym in bcfg.symbols:
        if sym in own or sym in state["dry_seen"] or any(c["symbol"] == sym for c in state["closed"]):
            continue
        try:
            res = _entry(broker, tm, bcfg, J, options, gates, state, risk_acct, sym, today, now_utc, open_utc, expected_last,
                         equity, history, context, bars_source)
        except Exception as e:                           # one bad symbol must not stop the pass; state is already on disk
            J.decision("error", symbol=sym, error="%s: %s" % (type(e).__name__, str(e)[:200]))
            res = {"symbol": sym, "result": "error"}
        J.save_state(state)
        if res:
            summary["entries"].append(res)
    return summary


def _entry(broker, tm, bcfg, J, options, gates, state, risk_acct, sym, today, now_utc, open_utc, expected_last, equity,
           history, context, bars_source=None):
    if bars_source is not None:
        D, last_real, why = bars_source(sym)
        tol = -(-30 // options[0].strategy.bar_seconds)           # fast bars: tolerate ~30 s without a trade
    else:
        raw = broker.bars_5m(sym, utc_iso(open_utc), utc_iso(now_utc))
        D, last_real, why = live_bars(raw, tm, now_utc, today, bcfg.max_gap_bars)
        tol = 0
    if not D:
        J.decision("skip", symbol=sym, why=why)
        return None
    if last_real < expected_last - tol:
        J.decision("skip", symbol=sym, why="stale_data", last_bar=last_real, expected=expected_last)
        return None
    prior = history.get(sym, [])
    if not prior or (datetime.date.fromisoformat(today) - datetime.date.fromisoformat(prior[-1][0])).days > bcfg.max_hist_age_days:
        J.decision("skip", symbol=sym, why="stale_history")
        return None
    scored, vetoes = [], []
    for opt in options:                                  # 1-3: every strategy proposes, its own model must agree
        cand, why = opt.strategy.live_candidate(tm, sym, today, D, prior, context)
        if cand is None:
            if why != "no_trigger":
                J.decision("skip", symbol=sym, why=why, strategy=opt.name)
            continue
        if opt.model is None:
            take, p = True, None                         # strategy-only dry-run
        else:
            if sym not in opt.meta.get("symbols", []):
                J.decision("skip", symbol=sym, why="symbol_not_in_model", strategy=opt.name)
                continue
            take, p = opt.model.take(cand)
        if not take:
            J.decision("ai_veto", symbol=sym, prob=p, threshold=opt.model.threshold, strategy=opt.name)
            vetoes.append(p)
            continue
        scored.append({"ev": rt.ev_r(p, cand) if p is not None else 0.0, "opt": opt, "cand": cand, "p": p})
    if not scored:
        if vetoes:
            state["dry_seen"].append(sym)
            return {"symbol": sym, "result": "ai_veto", "prob": vetoes[0]}
        return None
    ranked = rt.rank(scored)
    if len(ranked) > 1:
        J.decision("router_ranking", symbol=sym, order=[(r["opt"].name, round(r["ev"], 3), r["p"]) for r in ranked])
    live = [n for n, g in gates.items() if g[0]]
    shadow = set(gates) - set(live) if live else set()    # real orders requested: ungated strategies only watch
    last = None
    for item in ranked:                                  # 4: first one the risk engine accepts
        opt, cand, p = item["opt"], item["cand"], item["p"]
        if opt.name in shadow:
            J.decision("shadow_candidate", symbol=sym, strategy=opt.name, prob=p, why=gates[opt.name][1])
            continue
        shares = bt.size(bcfg.risk, equity, cand)
        ok, reason = bt.risk_check(bcfg.risk, risk_acct, cand, shares)
        if not ok:
            J.decision("risk_veto", symbol=sym, why=reason, shares=shares, strategy=opt.name)
            last = {"symbol": sym, "result": "risk_veto", "why": reason}
            continue
        price = broker.last_price(sym)
        if price <= cand["stop"] or price > cand["entry"] * (1 + bcfg.max_drift_bps / 1e4):
            J.decision("skip", symbol=sym, why="price_drift", price=price, entry=cand["entry"], stop=cand["stop"], strategy=opt.name)
            last = {"symbol": sym, "result": "price_drift"}
            continue
        return _place(broker, J, state, risk_acct, sym, today, now_utc, opt, cand, p, shares, gates[opt.name][0])
    state["dry_seen"].append(sym)
    return last


def _place(broker, J, state, risk_acct, sym, today, now_utc, opt, cand, p, shares, orders_on):
    plan = {"symbol": sym, "strategy": opt.name, "shares": shares, "entry": cand["entry"], "stop": cand["stop"],
            "target": cand["target"], "prob": p}
    if not orders_on:
        J.decision("would_order", **plan)
        state["dry_seen"].append(sym)
        return dict(plan, result="would_order")
    cid = "%s-%s-%s" % (opt.name, today.replace("-", ""), sym)
    if broker.order_by_client_id(cid) is not None:
        J.decision("skip", symbol=sym, why="already_sent_today", strategy=opt.name)
        return None
    pos = dict(plan, cid=cid, status="sent", risk=cand["risk"], date=today, sent_at=now_utc.isoformat(timespec="seconds"))
    state["positions"][sym] = pos                          # on disk BEFORE the order leaves
    state["trades_today"] += 1
    J.save_state(state)
    J.order("intent", **dict(plan, cid=cid))
    risk_acct["open"].append({"symbol": sym, "exit_slot": 10 ** 6, "risk_usd": shares * cand["risk"], "notional": shares * cand["entry"]})
    risk_acct["trades_today"] += 1
    try:
        res = broker.submit_bracket(cid, sym, shares, cand["stop"], cand["target"])
        pos.update(status="submitted", order_id=res.get("id"))
        J.order("accepted", cid=cid, order_id=res.get("id"), status=res.get("status"))
        return dict(plan, result="submitted")
    except bk.OrderRejected as e:
        del state["positions"][sym]
        state["trades_today"] -= 1
        J.order("rejected", cid=cid, error=str(e)[:300])
        return dict(plan, result="rejected")
    except bk.OrderUnknown as e:
        pos["status"] = "unknown"                          # resolved next tick by client_order_id lookup, never re-sent
        J.order("unknown", cid=cid, error=str(e)[:300])
        return dict(plan, result="unknown")


# ---------------------------------------------------------------- CLI

def build_cfg(a):
    return BotConfig(a.symbols, a.bars_dir, a.out, a.model, a.ledger,
                     strategy=ss.get(a.strategies[0] if a.strategies else a.strategy))


def main(argv=None):
    p = argparse.ArgumentParser(description="Alpaca PAPER ORB bot (dry-run unless --paper-orders).")
    p.add_argument("command", choices=["tick", "status", "flatten", "kill", "unkill"])
    p.add_argument("--symbols", nargs="*", default=["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN", "GOOGL"])
    p.add_argument("--bars-dir", default="research/bars")
    p.add_argument("--out", default="research/out/bot")
    p.add_argument("--model", default="research/out/model")
    p.add_argument("--ledger", default="research/out/ledger.jsonl")
    p.add_argument("--paper-orders", action="store_true", help="actually send PAPER orders (default: dry-run)")
    p.add_argument("--no-model", action="store_true", help="strategy-only DRY-RUN, no AI filter (never sends orders)")
    p.add_argument("--strategy", default="orb", choices=ss.names())
    p.add_argument("--strategies", nargs="*", choices=ss.names(),
                   help="ROUTER mode: several strategies, each with its own model in <model-root>/<name>/")
    p.add_argument("--model-root", default="research/out/models")
    a = p.parse_args(argv)
    bcfg = build_cfg(a)
    J = jn.Journal(bcfg.out_dir)
    if a.command in ("kill", "unkill"):
        os.makedirs(bcfg.out_dir, exist_ok=True)
        if a.command == "kill":
            open(bcfg.kill_file, "w").close()
        elif os.path.exists(bcfg.kill_file):
            os.remove(bcfg.kill_file)
        print("kill switch %s" % ("ON: no new entries" if os.path.exists(bcfg.kill_file) else "off"))
        return 0
    if a.command == "status":
        st = J.load_state()
        print(json.dumps({"kill_switch": os.path.exists(bcfg.kill_file), "state": st}, indent=1))
        return 0
    try:
        broker = bk.AlpacaPaperBroker()
    except bk.MissingKeys as e:
        print(e)
        return 2
    tm = o.load_pack()
    now = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0, tzinfo=None)
    if a.command == "flatten":
        state = J.load_state()
        state.setdefault("positions", {})
        out = tmgr.manage(broker, J, state, now, flatten=True)
        J.save_state(state)
        print(json.dumps(out))
        return 0
    model, meta, router = None, {}, None
    if a.strategies:
        router = rt.Router(rt.load_options(a.strategies, a.model_root))
        if a.paper_orders and not any(opt.model for opt in router.options):
            print("--paper-orders needs at least one strategy with a model; refusing")
            return 2
    elif not a.no_model:
        try:
            model, meta = ms.load(bcfg.model_dir)
        except (OSError, ms.ModelError) as e:
            print("no usable model (%s): running ORB-only DRY-RUN" % e)
    if a.paper_orders and model is None and router is None:
        print("--paper-orders needs a model; refusing")
        return 2
    need = max(opt.strategy.history_sessions for opt in router.options) if router else bcfg.strategy.history_sessions
    hist = load_history(tm, bcfg.bars_dir, bcfg.symbols + bcfg.context, need)
    print(json.dumps(tick(broker, tm, bcfg, model, meta, now, a.paper_orders, hist, router), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
