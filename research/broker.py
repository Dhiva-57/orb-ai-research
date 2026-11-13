#!/usr/bin/env python3
"""Broker layer. PAPER ONLY by construction.

AlpacaPaperBroker talks to https://paper-api.alpaca.markets (orders, account) and https://data.alpaca.markets (bars,
latest trade). Every request is checked against those two hosts; there is no flag, argument or environment variable
that points it anywhere else (APCA_API_BASE_URL is deliberately ignored). Keys come from ALPACA_API_KEY and
ALPACA_SECRET_KEY in the environment only.

POST /v2/orders is NEVER retried. A timeout or 5xx raises OrderUnknown ("may or may not have reached the broker");
the bot resolves that by looking the order up by its client_order_id, which is deterministic.

Written against Alpaca's documented formats; NOT run against the live service (the build sandbox blocks it).
FakeBroker is an in-memory stand-in with the same interface, for tests and dry experiments.
"""

import json
import os
import urllib.error
import urllib.parse
import urllib.request

PAPER_HOST = "https://paper-api.alpaca.markets"
DATA_HOST = "https://data.alpaca.markets"


class BrokerError(Exception):
    pass


class MissingKeys(BrokerError):
    pass


class Refused(BrokerError):
    """Wrong host or bad credentials."""


class OrderRejected(BrokerError):
    """The broker answered 4xx: the order does NOT exist."""


class OrderUnknown(BrokerError):
    """No clear answer (timeout / 5xx): the order may or may not exist."""


class TransportError(BrokerError):
    pass


def urllib_transport(method, url, headers, body=None, timeout=30):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    hdrs = dict(headers)
    if data is not None:
        hdrs["content-type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            doc = json.loads(raw)
        except ValueError:
            doc = {"message": raw[:200].decode("utf-8", "replace")}
        return e.code, doc
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise TransportError(str(e))


class AlpacaPaperBroker:
    paper = True

    def __init__(self, env=os.environ, transport=urllib_transport, feed="iex"):
        key, secret = env.get("ALPACA_API_KEY"), env.get("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise MissingKeys("set ALPACA_API_KEY and ALPACA_SECRET_KEY (paper-account keys) in the environment")
        self._h = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret, "accept": "application/json"}
        self._t = transport
        self.feed = feed

    # -------------------------------------------------------------- plumbing
    def _call(self, method, path, body=None, data=False, query=None):
        url = (DATA_HOST if data else PAPER_HOST) + path
        if query:
            url += "?" + urllib.parse.urlencode(query)
        if not url.startswith((PAPER_HOST + "/", DATA_HOST + "/")):     # belt and braces
            raise Refused("refusing a request outside the Alpaca paper/data hosts: %s" % url)
        try:
            status, doc = self._t(method, url, self._h, body)
        except TransportError:
            raise
        if status in (401, 403) and not data and method == "GET":
            raise Refused("Alpaca refused the keys (HTTP %d): %s" % (status, doc))
        return status, doc

    def _get(self, path, data=False, query=None, allow_404=False):
        status, doc = self._call("GET", path, data=data, query=query)
        if status == 404 and allow_404:
            return None
        if status != 200:
            raise BrokerError("GET %s -> HTTP %d %s" % (path, status, doc))
        return doc

    # -------------------------------------------------------------- reads
    def account(self):
        return self._get("/v2/account")

    def positions(self):
        return self._get("/v2/positions")

    def clock(self):
        return self._get("/v2/clock")

    def last_price(self, symbol):
        doc = self._get("/v2/stocks/%s/trades/latest" % urllib.parse.quote(symbol), data=True,
                        query={"feed": self.feed})
        return float(doc["trade"]["p"])

    def bars_5m(self, symbol, start, end):
        bars, token = [], None
        while True:
            q = {"timeframe": "5Min", "start": start, "end": end, "limit": 10000, "adjustment": "split",
                 "feed": self.feed, "sort": "asc"}
            if token:
                q["page_token"] = token
            doc = self._get("/v2/stocks/%s/bars" % urllib.parse.quote(symbol), data=True, query=q)
            got = doc.get("bars") or []
            if isinstance(got, dict):
                got = got.get(symbol) or []
            bars += got
            token = doc.get("next_page_token")
            if not token:
                return bars

    def order_by_client_id(self, cid):
        return self._get("/v2/orders:by_client_order_id", query={"client_order_id": cid}, allow_404=True)

    def open_orders(self, symbol):
        return self._get("/v2/orders", query={"status": "open", "symbols": symbol, "nested": "true"})

    # -------------------------------------------------------------- writes (paper)
    def submit_bracket(self, cid, symbol, qty, stop, target):
        """Market BUY with a resting take-profit and stop-loss (day orders). Never retried."""
        body = {"symbol": symbol, "qty": str(int(qty)), "side": "buy", "type": "market", "time_in_force": "day",
                "order_class": "bracket", "client_order_id": cid,
                "take_profit": {"limit_price": "%.2f" % target}, "stop_loss": {"stop_price": "%.2f" % stop}}
        try:
            status, doc = self._call("POST", "/v2/orders", body)
        except TransportError as e:
            raise OrderUnknown("no answer from Alpaca: %s" % e)
        if status in (200, 201):
            return doc
        if 400 <= status < 500:
            raise OrderRejected("HTTP %d %s" % (status, doc))
        raise OrderUnknown("HTTP %d %s" % (status, doc))

    def cancel_order(self, order_id):
        status, doc = self._call("DELETE", "/v2/orders/%s" % urllib.parse.quote(str(order_id)))
        if status not in (200, 204, 404, 422):
            raise BrokerError("cancel %s -> HTTP %d %s" % (order_id, status, doc))

    def close_position(self, symbol):
        status, doc = self._call("DELETE", "/v2/positions/%s" % urllib.parse.quote(symbol))
        if status in (200, 204):
            return doc
        if status == 404:
            return None                                   # already flat
        raise BrokerError("close %s -> HTTP %d %s" % (symbol, status, doc))


class FakeBroker:
    """In-memory paper broker with the same interface. Tests drive it: set .prices, .raw_bars, .now; call
    .hit_leg(symbol, 'stop'|'target') to simulate a bracket leg filling."""

    paper = True

    def __init__(self, equity=10000.0, is_open=True, next_close="2026-03-02T16:00:00-05:00"):
        self.equity = equity
        self.is_open = is_open
        self.next_close = next_close
        self.timestamp = "2026-03-02T10:00:00-05:00"
        self.prices, self.raw_bars = {}, {}
        self.pos, self.orders, self.calls = {}, {}, []
        self.fail_submit = None            # None | "rejected" | "unknown_not_placed" | "unknown_but_placed"
        self.fill_orders = True
        self.blocked = False
        self.close_fails = 0

    def account(self):
        self.calls.append("account")
        return {"equity": str(self.equity), "last_equity": str(self.equity), "trading_blocked": self.blocked,
                "account_blocked": False, "buying_power": str(self.equity * 4)}

    def positions(self):
        self.calls.append("positions")
        return [dict(symbol=s, qty=str(p["qty"]), avg_entry_price=str(p["entry"]),
                     current_price=str(self.prices.get(s, p["entry"])),
                     unrealized_pl=str(round((self.prices.get(s, p["entry"]) - p["entry"]) * p["qty"], 2)))
                for s, p in self.pos.items()]

    def clock(self):
        return {"is_open": self.is_open, "timestamp": self.timestamp, "next_close": self.next_close}

    def last_price(self, symbol):
        return float(self.prices[symbol])

    def bars_5m(self, symbol, start, end):
        return list(self.raw_bars.get(symbol, []))

    def order_by_client_id(self, cid):
        return self.orders.get(cid)

    def open_orders(self, symbol):
        return [o for o in self.orders.values() if o["symbol"] == symbol and o["status"] in ("new", "accepted")]

    def submit_bracket(self, cid, symbol, qty, stop, target):
        self.calls.append(("submit", cid, symbol, qty, stop, target))
        if self.fail_submit == "rejected":
            raise OrderRejected("fake: rejected")
        if self.fail_submit == "unknown_not_placed":
            raise OrderUnknown("fake: timeout, order never arrived")
        if cid in self.orders:
            raise OrderRejected("fake: duplicate client_order_id")
        px = float(self.prices.get(symbol, 0))
        order = {"id": "ord-%d" % (len(self.orders) + 1), "client_order_id": cid, "symbol": symbol, "qty": str(qty),
                 "status": "filled" if self.fill_orders else "accepted",
                 "filled_qty": str(qty) if self.fill_orders else "0",
                 "filled_avg_price": str(px) if self.fill_orders else None, "_stop": stop, "_target": target}
        self.orders[cid] = order
        if self.fill_orders:
            self.pos[symbol] = {"qty": int(qty), "entry": px}
        if self.fail_submit == "unknown_but_placed":
            raise OrderUnknown("fake: timeout AFTER the broker accepted it")
        return order

    def cancel_order(self, order_id):
        self.calls.append(("cancel", order_id))
        for o in self.orders.values():
            if o["id"] == order_id and o["status"] in ("new", "accepted"):
                o["status"] = "canceled"

    def close_position(self, symbol):
        self.calls.append(("close", symbol))
        if self.close_fails > 0:
            self.close_fails -= 1
            raise BrokerError("fake: close failed")
        return self.pos.pop(symbol, None)

    def hit_leg(self, symbol, leg):
        """Simulate the stop or target bracket leg filling at the broker (the order then shows the filled leg)."""
        for o in self.orders.values():
            if o["symbol"] == symbol:
                o["legs"] = [{"type": "limit" if leg == "target" else "stop", "status": "filled",
                              "filled_avg_price": str(o["_target"] if leg == "target" else o["_stop"])}]
        return self.pos.pop(symbol, None)
