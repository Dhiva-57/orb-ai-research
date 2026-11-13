#!/usr/bin/env python3
"""Download 5-minute stock bars from Alpaca's Market Data API into <SYMBOL>.json.gz files (Yahoo-style chart JSON,
the format the research code reads).

    export ALPACA_API_KEY=...        # keys come from the ENVIRONMENT only: never as arguments, never in a chat,
    export ALPACA_SECRET_KEY=...     # never committed
    python3 research/alpaca_bars.py --symbols SPY QQQ AAPL MSFT NVDA --start 2025-01-01 --out research/bars

READ-ONLY by construction: GET requests to the historical-bars endpoint and nothing else (a test greps for any
order / account call). Per Alpaca's docs (docs.alpaca.markets/us/reference/stockbarsingle-1):
  GET https://data.alpaca.markets/v2/stocks/{symbol}/bars   headers APCA-API-KEY-ID, APCA-API-SECRET-KEY
  params timeframe=5Min start end limit<=10000 adjustment feed page_token
  reply  {"bars": [{"t","o","h","l","c","v","vw"}], "next_page_token": ...}
Free plan: feed `iex` only (one exchange's volume; quiet stocks have missing bars). `sip` needs a paid plan.

Cleaning: regular-session (09:30-16:00 New York) 5-minute bars only. A session missing up to --max-missing interior
bars has them filled flat at the previous close with zero volume; a session missing its opening bar, its last bar,
or more than that is DROPPED, never repaired. Splits are adjusted.
Written against the documented format and tested against a local fake server; NOT run against the live service.
"""

import argparse
import calendar
import datetime
import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE_URL = "https://data.alpaca.markets"
PAGE_LIMIT = 10000
RETRIES = 5
GRID = 300
SESSION_BARS = 78
OPEN_MIN = 9 * 60 + 30


class AlpacaError(Exception):
    pass


class MissingKeys(AlpacaError):
    pass


class Refused(AlpacaError):
    pass


def keys_from_env(env=os.environ):
    k, s = env.get("ALPACA_API_KEY", "").strip(), env.get("ALPACA_SECRET_KEY", "").strip()
    if not k or not s:
        raise MissingKeys("set ALPACA_API_KEY and ALPACA_SECRET_KEY in your environment (never as arguments or in a chat)")
    return k, s


def http_get(url, headers, timeout=60):
    """-> (status, headers, body). HTTP errors are returned, not raised."""
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), e.read()
    except (urllib.error.URLError, OSError) as e:
        raise AlpacaError("network error: %s" % getattr(e, "reason", e))


def get_json(url, headers, fetch=http_get, sleep=time.sleep):
    """Retried on 429 / 5xx. 400/401/403/404 stop at once with a message that never contains the keys."""
    for attempt in range(RETRIES):
        status, hdrs, body = fetch(url, headers)
        if status == 200:
            try:
                return json.loads(body.decode("utf-8"))
            except ValueError:
                raise AlpacaError("Alpaca returned something that is not JSON")
        if status == 429 or status >= 500:
            wait = 0.0
            try:
                wait = float({k.lower(): v for k, v in hdrs.items()}.get("retry-after", ""))
            except ValueError:
                pass
            sleep(max(wait, min(2 ** attempt, 30)))
            continue
        if status in (401, 403):
            raise Refused("Alpaca refused the keys (HTTP %d). Check the key pair, that it is for market data, and "
                          "that your plan allows this feed." % status)
        if status == 404:
            raise AlpacaError("Alpaca says that symbol or endpoint was not found (HTTP 404)")
        detail = ""
        try:
            detail = json.loads(body.decode("utf-8")).get("message", "")
        except Exception:
            pass
        raise AlpacaError("Alpaca rejected the request (HTTP %d) %s" % (status, detail))
    raise AlpacaError("Alpaca kept answering 429/5xx after %d tries; try again later" % RETRIES)


def fetch_bars(symbol, start, end, feed, adjustment, headers, base_url=BASE_URL, fetch=http_get, sleep=time.sleep):
    """All 5-minute bars for one symbol, oldest first, following next_page_token."""
    bars, token = [], None
    while True:
        q = {"timeframe": "5Min", "start": start, "end": end, "limit": PAGE_LIMIT, "adjustment": adjustment,
             "feed": feed, "sort": "asc"}
        if token:
            q["page_token"] = token
        doc = get_json("%s/v2/stocks/%s/bars?%s" % (base_url.rstrip("/"), urllib.parse.quote(symbol),
                                                      urllib.parse.urlencode(q)), headers, fetch, sleep)
        got = doc.get("bars") or []
        if isinstance(got, dict):                   # tolerate a multi-symbol shape {"bars": {"AAPL": [...]}}
            got = got.get(symbol) or []
        bars += got
        token = doc.get("next_page_token")
        if not token:
            return bars


def epoch(rfc3339):
    return calendar.timegm(datetime.datetime.strptime(rfc3339.replace("Z", "").split(".")[0], "%Y-%m-%dT%H:%M:%S").timetuple())


def convert(bars, symbol, tm, fill_gaps=True, max_missing=6):
    """Alpaca bars -> (Yahoo-style chart JSON or None, stats). Keeps only complete regular sessions. `tm` is the
    day-trader pack's timemachine module (New York time handling)."""
    stats = {"sessions": 0, "kept": 0, "filled_bars": 0, "dropped_gappy": 0, "dropped_no_open": 0, "dropped_short": 0}
    days = {}
    for b in bars:
        try:
            if any(b.get(k) is None for k in ("t", "o", "h", "l", "c")):
                continue
            ts = epoch(b["t"])
            px = (float(b["o"]), float(b["h"]), float(b["l"]), float(b["c"]))
        except (KeyError, ValueError, TypeError, AttributeError):
            continue
        if ts % GRID:
            continue                                  # off the 5-minute grid
        local = tm.to_eastern(tm.utc_of(ts))
        mins = local.hour * 60 + local.minute - OPEN_MIN
        if mins < 0 or mins >= SESSION_BARS * 5:
            continue                                  # pre-market or after-hours
        days.setdefault(local.date().isoformat(), {})[mins // 5] = (ts,) + px + (b.get("v") or 0,)
    stamps, o, h, l, c, v = [], [], [], [], [], []
    for d in sorted(days):
        slots = days[d]
        stats["sessions"] += 1
        if 0 not in slots:
            stats["dropped_no_open"] += 1
            continue
        if SESSION_BARS - 1 not in slots:
            stats["dropped_short"] += 1               # early close, or the data ends mid-day
            continue
        missing = [i for i in range(SESSION_BARS) if i not in slots]
        if missing and (not fill_gaps or len(missing) > max_missing):
            stats["dropped_gappy"] += 1
            continue
        open_ts, prev_close = slots[0][0], None
        for i in range(SESSION_BARS):
            if i in slots:
                _, bo, bh, bl, bc, bv = slots[i]
            else:
                bo = bh = bl = bc = prev_close         # flat at the last close, no volume
                bv = 0.0
                stats["filled_bars"] += 1
            stamps.append(open_ts + GRID * i)
            o.append(bo); h.append(bh); l.append(bl); c.append(bc); v.append(bv)
            prev_close = bc
        stats["kept"] += 1
    if not stamps:
        return None, stats
    return {"chart": {"result": [{"meta": {"symbol": symbol, "dataGranularity": "5m", "source": "alpaca",
                                           "regularMarketTime": stamps[-1] + GRID},
                                  "timestamp": stamps,
                                  "indicators": {"quote": [{"open": o, "high": h, "low": l, "close": c, "volume": v}]}}],
                      "error": None}}, stats


def write_chart(path, doc):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(doc, fh)
    os.replace(tmp, path)


def download(symbols, start, end, out, feed="iex", adjustment="split", force=False, base_url=BASE_URL,
             fetch=http_get, sleep=time.sleep, env=os.environ, log=print, pause=0.35, fill_gaps=True, max_missing=6):
    key, secret = keys_from_env(env)                  # before any request or file
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret, "accept": "application/json"}
    import orb_dataset as o
    tm = o.load_pack()
    results = {}
    for sym in symbols:
        sym = sym.upper()
        path = os.path.join(out, "%s.json.gz" % sym)
        if os.path.exists(path) and not force:
            log("%-6s skipped (exists; --force to refresh)" % sym)
            results[sym] = "exists"
            continue
        try:
            bars = fetch_bars(sym, start, end, feed, adjustment, headers, base_url, fetch, sleep)
        except Refused:
            raise                                     # the keys are wrong for everything: stop, do not hammer
        except AlpacaError as e:
            log("%-6s FAILED: %s" % (sym, e))
            results[sym] = "failed"
            continue
        if not bars:
            log("%-6s no bars returned for %s..%s" % (sym, start, end))
            results[sym] = "empty"
            continue
        doc, st = convert(bars, sym, tm, fill_gaps, max_missing)
        if doc is None:
            log("%-6s %d bars but no complete session (%s)" % (sym, len(bars), st))
            results[sym] = "empty"
            continue
        write_chart(path, doc)
        warn = "  WARNING: many sessions dropped for missing bars (thin volume on the iex feed?)" \
            if st["sessions"] and (st["sessions"] - st["kept"]) / st["sessions"] > 0.2 else ""
        log("%-6s %6d bars, %d/%d sessions kept, %d gap bars filled%s"
            % (sym, len(bars), st["kept"], st["sessions"], st["filled_bars"], warn))
        results[sym] = "ok"
        sleep(pause)
    return results


def main(argv=None, sleep=time.sleep):
    p = argparse.ArgumentParser(description="Download 5-minute bars from Alpaca (keys from environment variables only).")
    p.add_argument("--symbols", nargs="*", default=[])
    p.add_argument("--symbols-file", help="a text file with one ticker per line")
    p.add_argument("--start", default=(datetime.date.today() - datetime.timedelta(days=730)).isoformat())
    p.add_argument("--end", default=(datetime.date.today() - datetime.timedelta(days=1)).isoformat())
    p.add_argument("--feed", default="iex", choices=["iex", "sip"], help="free plan: iex")
    p.add_argument("--adjustment", default="split")
    p.add_argument("--out", default="research/bars")
    p.add_argument("--base-url", default=BASE_URL)
    p.add_argument("--max-missing", type=int, default=6)
    p.add_argument("--no-fill", action="store_true", help="drop any session with a missing bar instead of filling")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    syms = list(a.symbols)
    if a.symbols_file:
        with open(a.symbols_file, encoding="utf-8") as fh:
            syms += [x.strip().upper() for x in fh if x.strip() and not x.startswith("#")]
    if not syms:
        p.error("give --symbols or --symbols-file")
    try:
        res = download(syms, a.start, a.end, a.out, a.feed, a.adjustment, a.force, a.base_url, sleep=sleep,
                       fill_gaps=not a.no_fill, max_missing=a.max_missing)
    except MissingKeys as e:
        print("alpaca_bars.py: %s" % e, file=sys.stderr)
        return 1
    except AlpacaError as e:
        print("alpaca_bars.py: %s" % e, file=sys.stderr)
        return 2
    ok = sum(1 for v in res.values() if v in ("ok", "exists"))
    print("\n%d of %d symbols ready in %s/. Next: python3 research/walkforward.py --bars-dir %s" % (ok, len(res), a.out, a.out))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
