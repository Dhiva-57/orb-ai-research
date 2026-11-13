import contextlib
import datetime
import gzip
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
import alpaca_bars as ab  # noqa: E402
import orb_dataset as o  # noqa: E402

TM = o.load_pack()
FIXTURE = None
for base in (os.environ.get("DAY_TRADER_DIR"), "/tmp/day-trader-skill", os.path.join(ROOT, "..", "day-trader-skill")):
    p = base and os.path.join(base, "tests", "fixtures", "trader-timemachine", "TSLA-60d5m.json.gz")
    if p and os.path.exists(p):
        FIXTURE = p
        break

KEY, SECRET = "PKTESTKEY123456", "SECRETvalue987654321"


def iso(ts):
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fixture_as_alpaca():
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as fh:
        res = json.load(fh)["chart"]["result"][0]
    q, out = res["indicators"]["quote"][0], []
    for i, ts in enumerate(res["timestamp"]):
        vals = [q[k][i] for k in ("open", "high", "low", "close", "volume")]
        if ts % 300 or None in vals[:4]:
            continue
        out.append({"t": iso(ts), "o": vals[0], "h": vals[1], "l": vals[2], "c": vals[3], "v": vals[4] or 0})
    return out


def day_bars(date="2026-03-03", drop=(), extra=()):
    """One regular session in Alpaca format (EST in early March: 09:30 ET = 14:30Z)."""
    base = int(datetime.datetime.fromisoformat(date + "T14:30:00+00:00").timestamp())
    bars, px = [], 100.0
    for i in range(78):
        if i in drop:
            px += 0.01
            continue
        bars.append({"t": iso(base + 300 * i), "o": px, "h": px + 0.2, "l": px - 0.2, "c": px + 0.05, "v": 1000 + i})
        px += 0.05
    for off in extra:                                            # pre / post market bars
        bars.append({"t": iso(base + off), "o": 1, "h": 1, "l": 1, "c": 1, "v": 1})
    return bars


class Fake(BaseHTTPRequestHandler):
    bars, page, seen, mode, throttled = {}, 1000, [], "ok", {}

    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        Fake.seen.append((u.path, dict(self.headers)))
        sym = u.path.split("/")[3]
        if Fake.mode == "forbidden":
            return self._send(403, {"message": "forbidden"})
        if Fake.mode == "throttle_once" and not Fake.throttled.get(sym):
            Fake.throttled[sym] = True
            return self._send(429, {"message": "slow down"}, {"Retry-After": "0"})
        if not (self.headers.get("APCA-API-KEY-ID") and self.headers.get("APCA-API-SECRET-KEY")):
            return self._send(401, {"message": "no keys"})
        off = int(q.get("page_token", ["0"])[0])
        allb = Fake.bars.get(sym, [])
        self._send(200, {"bars": allb[off:off + Fake.page], "symbol": sym,
                         "next_page_token": str(off + Fake.page) if off + Fake.page < len(allb) else None})

    def _send(self, code, doc, extra=None):
        body = json.dumps(doc).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)


class Convert(unittest.TestCase):
    def conv(self, bars, **kw):
        return ab.convert(bars, "TST", TM, **kw)

    def test_complete_session_kept_and_short_gaps_filled_flat(self):
        doc, st = self.conv(day_bars())
        self.assertEqual((st["kept"], st["filled_bars"], len(doc["chart"]["result"][0]["timestamp"])), (1, 0, 78))
        doc, st = self.conv(day_bars(drop={10, 11, 40}))
        self.assertEqual((st["kept"], st["filled_bars"]), (1, 3))
        q = doc["chart"]["result"][0]["indicators"]["quote"][0]
        for i in (10, 11):
            self.assertEqual((q["open"][i], q["high"][i], q["low"][i], q["close"][i], q["volume"][i]),
                             (q["close"][9],) * 4 + (0.0,))

    def test_bad_sessions_are_dropped_never_repaired(self):
        self.assertEqual(self.conv(day_bars(drop=set(range(20, 27))))[1]["dropped_gappy"], 1)        # 7 missing > 6
        self.assertEqual(self.conv(day_bars(drop={0}))[1]["dropped_no_open"], 1)
        self.assertIsNone(self.conv(day_bars(drop={30}), fill_gaps=False)[0])
        self.assertIsNone(self.conv([b for i, b in enumerate(day_bars()) if i < 42])[0])              # early close

    def test_extended_hours_off_grid_and_garbage_never_enter(self):
        bars = day_bars(extra=(-5 * 3600, 6.5 * 3600 + 300, 7)) + [{"t": "nope"}, {}, {"t": "2026-03-03T15:00:00Z", "o": None}]
        doc, st = self.conv(bars)
        self.assertEqual((st["kept"], len(doc["chart"]["result"][0]["timestamp"])), (1, 78))


@unittest.skipUnless(FIXTURE, "needs the day-trader pack's TSLA fixture")
class Download(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), Fake)
        cls.url = "http://127.0.0.1:%d" % cls.srv.server_port
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        Fake.mode, Fake.seen, Fake.throttled, Fake.page = "ok", [], {}, 1000
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def run_main(self, *argv, env=True):
        old = {k: os.environ.get(k) for k in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY")}
        for k, v in (("ALPACA_API_KEY", KEY), ("ALPACA_SECRET_KEY", SECRET)):
            os.environ[k] = v
        if not env:
            os.environ.pop("ALPACA_API_KEY"); os.environ.pop("ALPACA_SECRET_KEY")
        out, err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = ab.main(list(argv) + ["--base-url", self.url, "--out", self.tmp], sleep=lambda s: None)
        finally:
            for k, v in old.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        return code, out.getvalue(), err.getvalue()

    def test_round_trip_gives_the_identical_dataset_across_several_pages(self):
        Fake.bars = {"TSLA": fixture_as_alpaca()}
        code, out, err = self.run_main("--symbols", "TSLA", "--start", "2026-07-01", "--end", "2026-10-30")
        self.assertEqual(code, 0, err)
        self.assertGreater(len([s for s in Fake.seen if s[0].endswith("/TSLA/bars")]), 3)
        got = o.build(TM, o.load_sessions(TM, os.path.join(self.tmp, "TSLA.json.gz"), "TSLA"), symbol="TSLA")
        want = o.build(TM, o.load_sessions(TM, FIXTURE, "TSLA"), symbol="TSLA")
        self.assertEqual((len(want), got), (33, want))

    def test_keys_are_sent_as_headers_and_never_leak(self):
        Fake.bars = {"TSLA": day_bars()}
        code, out, err = self.run_main("--symbols", "TSLA", "--start", "2026-03-01", "--end", "2026-03-05")
        self.assertEqual(code, 0, err)
        hdrs = {k.lower(): v for k, v in Fake.seen[0][1].items()}
        self.assertEqual((hdrs["apca-api-key-id"], hdrs["apca-api-secret-key"]), (KEY, SECRET))
        blob = out + err
        for name in os.listdir(self.tmp):
            with gzip.open(os.path.join(self.tmp, name), "rt") as fh:
                blob += fh.read()
        self.assertNotIn(SECRET, blob)
        self.assertNotIn(KEY, blob)

    def test_no_keys_means_a_clear_refusal_and_no_request(self):
        code, out, err = self.run_main("--symbols", "TSLA", "--start", "2026-03-01", "--end", "2026-03-05", env=False)
        self.assertEqual((code, Fake.seen, os.listdir(self.tmp)), (1, [], []))
        self.assertIn("ALPACA_API_KEY", err)

    def test_429_is_retried_and_403_stops_at_once_writing_nothing(self):
        Fake.bars, Fake.mode = {"TSLA": day_bars()}, "throttle_once"
        self.assertEqual(self.run_main("--symbols", "TSLA", "--start", "2026-03-01", "--end", "2026-03-05")[0], 0)
        self.assertEqual(len(Fake.seen), 2)
        shutil.rmtree(self.tmp); os.makedirs(self.tmp)
        Fake.bars, Fake.mode, Fake.seen = {"A": day_bars(), "B": day_bars()}, "forbidden", []
        code, out, err = self.run_main("--symbols", "A", "B", "--start", "2026-03-01", "--end", "2026-03-05")
        self.assertEqual((code, len(Fake.seen), os.listdir(self.tmp)), (2, 1, []))
        self.assertIn("refused", err)

    def test_only_get_requests_to_the_bars_endpoint_and_no_trading_code(self):
        Fake.bars = {"TSLA": day_bars()}
        self.run_main("--symbols", "TSLA", "--start", "2026-03-01", "--end", "2026-03-05")
        for path, _ in Fake.seen:
            self.assertRegex(path, r"^/v2/stocks/[A-Z.]+/bars$")
        src = open(os.path.join(ROOT, "research", "alpaca_bars.py"), encoding="utf-8").read()
        for forbidden in ("/v2/orders", "api.alpaca.markets", "paper-api", '"POST"'):
            self.assertNotIn(forbidden, src)


@unittest.skipUnless(FIXTURE, "needs the day-trader pack's TSLA fixture")
class Directory(unittest.TestCase):
    def test_context_symbols_become_features_and_are_not_traded(self):
        with tempfile.TemporaryDirectory() as d:
            for sym in ("TSLA", "SPY", "QQQ"):
                shutil.copy(FIXTURE, os.path.join(d, sym + ".json.gz"))
            rows, info = o.load_dir(TM, d)
            self.assertEqual((sorted(info["context"]), list(info["symbols"])), (["QQQ", "SPY"], ["TSLA"]))
            self.assertTrue(rows and all(r["symbol"] == "TSLA" for r in rows))
            self.assertTrue(any(r["ctx_spy_ret_pct"] is not None for r in rows))

    def test_empty_folder_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                o.load_dir(TM, d)


if __name__ == "__main__":
    unittest.main()
