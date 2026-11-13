#!/usr/bin/env python3
"""Alpaca market-data websocket (trades). READ-ONLY market data: this module can never place an order.

!! The connection code below was written from Alpaca's documented protocol and has NEVER been run against the live service
!! (the build sandbox cannot reach it). The pure helpers (auth/subscribe payloads, message parsing, back-off schedule,
!! staleness watchdog) are unit-tested. Test the stream yourself with `live_runner.py record` before trusting anything on it.

Free (basic) plans: IEX feed only, and typically ONE concurrent connection per account: a second connection (another script,
the same script twice) is refused. Needs `pip install websocket-client`.
"""

import json
import time

import tickbars

STREAM_URL = "wss://stream.data.alpaca.markets/v2/iex"
STALE_AFTER_S = 45          # no message at all for this long during the session -> reconnect


def auth_payload(key, secret):
    return json.dumps({"action": "auth", "key": key, "secret": secret})


def subscribe_payload(symbols):
    return json.dumps({"action": "subscribe", "trades": sorted({s.upper() for s in symbols})})


def parse_messages(text):
    """Raw websocket text -> (trades, events). trades = [(symbol, epoch, price, size)], events = [(type, detail)]
    for success / error / subscription messages. Malformed items are skipped, never raised."""
    trades, events = [], []
    try:
        items = json.loads(text)
    except ValueError:
        return trades, [("bad_json", text[:100])]
    for m in items if isinstance(items, list) else [items]:
        if not isinstance(m, dict):
            continue
        t = m.get("T")
        if t == "t":
            try:
                trades.append((m["S"], tickbars.parse_rfc3339(m["t"]), float(m["p"]), float(m["s"])))
            except (KeyError, ValueError, TypeError):
                events.append(("bad_trade", str(m)[:100]))
        elif t in ("success", "error", "subscription"):
            events.append((t, m.get("msg") or m.get("code") or m.get("trades")))
    return trades, events


def backoff_schedule(attempt, base=1.0, cap=60.0):
    """Seconds to wait before reconnect attempt n (0-based): 1, 2, 4 ... capped."""
    return min(cap, base * (2 ** attempt))


def is_stale(last_message_epoch, now_epoch, in_session=True, limit=STALE_AFTER_S):
    return in_session and last_message_epoch is not None and now_epoch - last_message_epoch > limit


class TradeStream:
    """Blocking client: run() calls on_trade(symbol, epoch, price, size) for every trade and reconnects with back-off.
    UNTESTED against the live service."""

    def __init__(self, key, secret, symbols, on_trade, on_event=None, url=STREAM_URL):
        self.key, self.secret, self.symbols, self.on_trade, self.on_event, self.url = key, secret, symbols, on_trade, on_event, url
        self.last_message = None
        self.stop = False

    def run(self):
        try:
            import websocket                               # pip install websocket-client
        except ImportError:
            raise SystemExit("pip install websocket-client  (needed for the live stream)")
        attempt = 0
        while not self.stop:
            def on_open(ws):
                ws.send(auth_payload(self.key, self.secret))
                ws.send(subscribe_payload(self.symbols))

            def on_message(ws, text):
                self.last_message = time.time()
                trades, events = parse_messages(text)
                for ev in events:
                    if self.on_event:
                        self.on_event(*ev)
                for tr in trades:
                    self.on_trade(*tr)

            ws = websocket.WebSocketApp(self.url, on_open=on_open, on_message=on_message)
            started = time.time()
            ws.run_forever(ping_interval=20, ping_timeout=10)
            attempt = 0 if time.time() - started > 120 else attempt + 1       # a long healthy run resets the back-off
            if self.stop:
                break
            if self.on_event:
                self.on_event("reconnect", backoff_schedule(attempt))
            time.sleep(backoff_schedule(attempt))
