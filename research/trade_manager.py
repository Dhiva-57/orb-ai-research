#!/usr/bin/env python3
"""Trade manager: runs every tick (5 minutes today; the logic is bar-size agnostic) over the bot's OWN positions.

    reconcile  - make the bot's records match what the broker really has (fills, bracket-leg exits, vanished orders)
    evaluate   - pure: should this open position be closed now?   (stop / target / timeout / session end)
    manage     - reconcile, then close what evaluate says, flatten everything at the session-end cut-off

Rules that matter:
  * It only ever touches symbols the bot itself opened (state["positions"]). Foreign positions are never closed.
  * A bar that touches BOTH stop and target counts as the STOP (same pessimism as the dataset labels).
  * With broker-side bracket orders (default) the stop/target legs do the exiting; the manager is the backstop for
    session end, timeout, and a stop leg that failed to trigger. In software mode it exits on stop/target itself.
  * An order of unknown fate is resolved by looking it up by its client_order_id, never by re-sending it.
"""

import datetime

from broker import BrokerError

LIVE = ("sent", "unknown", "submitted", "open")
DEAD_ORDER = ("canceled", "cancelled", "expired", "rejected", "done_for_day")


def evaluate(pos, bar=None, flatten=False, bracket_mode=True, last_price=None, minutes_held=None,
             max_hold_minutes=None, backstop_bps=30.0):
    """Pure. -> {"action": None | "close", "reason": str}. `bar` = the latest COMPLETED bar {h, l, ...} or None."""
    if flatten:
        return {"action": "close", "reason": "session_end"}
    if max_hold_minutes is not None and minutes_held is not None and minutes_held >= max_hold_minutes:
        return {"action": "close", "reason": "timeout"}
    touched_stop = bar is not None and bar["l"] <= pos["stop"]
    touched_target = bar is not None and bar["h"] >= pos["target"]
    if touched_stop:                                  # both touched -> stop (pessimistic)
        which = "stop"
    elif touched_target:
        which = "target"
    else:
        which = None
    if not bracket_mode:
        return {"action": "close", "reason": which} if which else {"action": None, "reason": "hold"}
    # bracket mode: the broker's legs exit. Backstop only if price is clearly through the stop and we still hold.
    if last_price is not None and last_price <= pos["stop"] * (1 - backstop_bps / 1e4):
        return {"action": "close", "reason": "stop_backstop"}
    return {"action": None, "reason": "hold" if not which else "bracket_leg_expected_" + which}


def _num(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _minutes_since(iso, now_utc):
    try:
        t = datetime.datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return None
    return (now_utc - t).total_seconds() / 60.0


def _close(broker, journal, state, symbol, reason, now_utc):
    """Cancel the symbol's open orders (bracket legs hold the shares), then close. True when the bot is done."""
    pos = state["positions"][symbol]
    px = None
    try:
        px = broker.last_price(symbol)
    except (BrokerError, KeyError):
        pass
    try:
        for o in broker.open_orders(symbol):
            broker.cancel_order(o["id"])
        broker.close_position(symbol)
    except BrokerError as e:
        journal.decision("close_failed", symbol=symbol, reason=reason, error=str(e)[:200])
        return False                                   # stays in state; the next tick retries
    pos["status"] = "closed"
    state.setdefault("closed", []).append({"symbol": symbol, "reason": reason, "at": now_utc.isoformat(timespec="seconds"),
                                           "entry": pos.get("fill_price") or pos.get("entry"), "shares": pos.get("shares")})
    del state["positions"][symbol]
    journal.decision("closed", symbol=symbol, reason=reason)
    _trade_record(journal, pos, reason, now_utc, px, "last_price_before_close")
    return True


def reconcile(broker, journal, state, now_utc, grace_s=30, max_pending_s=600):
    held = {p["symbol"]: p for p in broker.positions()}
    for symbol in list(state.get("positions", {})):
        pos = state["positions"][symbol]
        status = pos["status"]
        if status == "open":
            if symbol not in held:
                _gone(state, journal, symbol, "broker_exit", now_utc, broker=broker)
            continue
        if status not in LIVE:
            continue
        age = _minutes_since(pos.get("sent_at"), now_utc)
        age_s = age * 60 if age is not None else 10 ** 9
        order = broker.order_by_client_id(pos["cid"])
        if order is None:
            if age_s >= grace_s:                       # never arrived at the broker
                _gone(state, journal, symbol, "order_not_found", now_utc, released=True)
            continue
        st = str(order.get("status", "")).lower()
        filled = _num(order.get("filled_qty"))
        pos["order_id"] = order.get("id")
        if symbol in held and filled > 0:
            pos.update(status="open", shares=int(_num(held[symbol]["qty"])),
                       fill_price=_num(order.get("filled_avg_price")) or _num(held[symbol]["avg_entry_price"]))
            journal.decision("position_open", symbol=symbol, shares=pos["shares"], fill=pos["fill_price"])
        elif st == "filled" and symbol not in held:
            _gone(state, journal, symbol, "broker_exit", now_utc, broker=broker)
        elif st in DEAD_ORDER and filled == 0:
            _gone(state, journal, symbol, "order_" + st, now_utc, released=True)
        elif age_s >= max_pending_s:
            try:
                broker.cancel_order(order.get("id"))
                journal.decision("cancel_stale_entry", symbol=symbol)
            except BrokerError as e:
                journal.decision("cancel_failed", symbol=symbol, error=str(e)[:200])
    return held


def _leg_exit(order):
    """(reason, price) of the filled bracket leg in a broker order, else (None, None). take-profit leg = limit."""
    for leg in (order or {}).get("legs") or []:
        px = _num(leg.get("filled_avg_price"), None)
        if str(leg.get("status", "")).lower() == "filled" and px:
            return ("target" if str(leg.get("type", "")).lower() == "limit" else "stop"), px
    return None, None


def _trade_record(journal, pos, reason, now_utc, exit_price, exit_source):
    """One permanent line per finished trade (trades.jsonl): what was planned, what filled, how it ended."""
    journal.trade(symbol=pos.get("symbol"), strategy=pos.get("strategy", "orb"), date=pos.get("date"), cid=pos.get("cid"),
                  shares=pos.get("shares"), entry=pos.get("entry"), stop=pos.get("stop"), target=pos.get("target"),
                  prob=pos.get("prob"), fill_price=pos.get("fill_price"), exit_price=exit_price, exit_source=exit_source,
                  exit_reason=reason, sent_at=pos.get("sent_at"), closed_at=now_utc.isoformat(timespec="seconds"))


def _gone(state, journal, symbol, reason, now_utc, released=False, broker=None):
    pos = state["positions"].pop(symbol)
    if not released:
        exit_price, source = None, None
        if broker is not None:
            try:
                why, px = _leg_exit(broker.order_by_client_id(pos.get("cid")))
                if px:
                    reason, exit_price, source = why, px, "bracket_leg"
            except BrokerError:
                pass
        _trade_record(journal, pos, reason, now_utc, exit_price, source)
    if not released:
        state.setdefault("closed", []).append({"symbol": symbol, "reason": reason,
                                               "at": now_utc.isoformat(timespec="seconds"),
                                               "entry": pos.get("fill_price") or pos.get("entry"),
                                               "shares": pos.get("shares")})
    journal.decision("released" if released else "closed", symbol=symbol, reason=reason)


def manage(broker, journal, state, now_utc, flatten=False, bar_fn=None, bracket_mode=True, max_hold_minutes=None,
           backstop_bps=30.0):
    """One manager pass. Returns {"closed": [...], "open": [...]}."""
    state.setdefault("positions", {})
    held = reconcile(broker, journal, state, now_utc)
    closed_now = []
    for symbol in list(state["positions"]):
        pos = state["positions"][symbol]
        if pos["status"] == "open" or flatten:
            price = None
            if pos["status"] == "open" and bracket_mode and not flatten:
                try:
                    price = broker.last_price(symbol)
                except (BrokerError, KeyError):
                    price = _num(held.get(symbol, {}).get("current_price"), None)
            bar = bar_fn(symbol) if (bar_fn and pos["status"] == "open") else None
            d = evaluate(pos, bar, flatten, bracket_mode, price,
                         _minutes_since(pos.get("opened_at") or pos.get("sent_at"), now_utc), max_hold_minutes, backstop_bps)
            if d["action"] == "close" and _close(broker, journal, state, symbol, d["reason"], now_utc):
                closed_now.append((symbol, d["reason"]))
            elif d["action"] is None and d["reason"] != "hold":
                journal.decision("note", symbol=symbol, reason=d["reason"])
    return {"closed": closed_now, "open": sorted(state["positions"])}
