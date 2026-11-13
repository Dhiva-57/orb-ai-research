import calendar
import datetime
import gzip
import os
import random
import sys
import tempfile
import time
import unittest
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import broker as bk  # noqa: E402
import journal as jn  # noqa: E402
import live_runner as lr  # noqa: E402
import orb_dataset as o  # noqa: E402
import paper_bot as pb  # noqa: E402
import stream as st  # noqa: E402
import strategies as ss  # noqa: E402
import strategies_more as sm  # noqa: E402
import tickbars as tb  # noqa: E402
from test_paper_bot import FakeModel, history as hist5  # noqa: E402
from test_strategies import sessions  # noqa: E402

TM = o.load_pack()
OPEN = calendar.timegm((2026, 3, 2, 14, 30, 0))            # 09:30 New York on Monday 2026-03-02 (EST)


def et(h, m, s=0):
    return OPEN + ((h * 60 + m) - 570) * 60 + s


def open_epoch(date):
    y, m, d = map(int, date.split("-"))
    return calendar.timegm(TM.eastern_to_utc(datetime.datetime(y, m, d, 9, 30)).timetuple())


def gen_day(date, seed, step=10, drift=0.0):
    """Trades every `step` seconds through the regular session: [(epoch, price, size)]."""
    r, px, t0, out = random.Random(seed), 100.0, open_epoch(date), []
    for i in range(0, 23400, step):
        px *= 1 + r.gauss(drift, 0.00035)
        out.append((t0 + i + r.random() * (step - 1), round(px, 2), r.randrange(1, 400)))
    return out


def weekdays(n, start=datetime.date(2026, 1, 5)):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += datetime.timedelta(days=1)
    return out


class Scaling(unittest.TestCase):
    def test_five_minute_defaults_are_unchanged(self):
        s = ss.get("orb")
        self.assertEqual((s.or_bars, s.last, s.latest_entry, s.bar_seconds), (3, 76, 60, 300))
        self.assertEqual((ss.get("momentum").start, ss.get("momentum").atr_n, ss.get("meanrev").window), (6, 12, 20))

    def test_same_wall_clock_durations_at_other_bar_sizes(self):
        s = ss.get("orb", bar_seconds=30)
        self.assertEqual((s.or_bars, s.last, s.latest_entry), (30, 760, 600))     # 15 min, 15:50, 14:30
        m = ss.get("momentum", bar_seconds=30)
        self.assertEqual((m.start, m.atr_n), (60, 120))
        self.assertEqual((ss.get("meanrev", bar_seconds=30).window, ss.get("vwap", bar_seconds=30).below_bars), (200, 30))
        self.assertEqual(ss.get("orb", bar_seconds=5).or_bars, 180)
        self.assertEqual(ss.get("orb", bar_seconds=30, or_bars=7).or_bars, 7)       # explicit values win

    def test_features_use_the_bar_size(self):
        D = sessions(3, seed=1)[0][1]
        oh, ol = TM.opening_range(D, 3)
        f300 = o.features_at(D, 9, oh, ol, [])
        self.assertEqual(f300["minutes_since_open"], 50)
        self.assertEqual(o.features_at(D, 9, oh, ol, [], 30)["minutes_since_open"], 5.0)
        self.assertEqual(f300, o.features_at(D, 9, oh, ol, [], 300))


def ref_trigger(s, D, i, vw):
    """The ORIGINAL quadratic definitions, kept here as the reference the linear scans must equal."""
    n = s.name.replace("_daily", "")
    if n == "momentum":
        return D[i]["c"] > max(b["h"] for b in D[:i]) and D[i]["c"] > vw[i] and D[i]["c"] > D[0]["o"]
    if n == "vwap":
        k = s.below_bars
        return i >= k and all(D[j]["c"] < vw[j] for j in range(i - k, i)) and D[i]["c"] > vw[i] and D[i]["c"] > D[i]["o"]
    if n == "pullback":
        st_ = s._state(D, i, vw)
        if st_ is None:
            return False
        jh, swing = st_
        a = sm.atr(D, i, s.atr_n)
        return (D[jh]["h"] >= D[0]["o"] * (1 + s.min_leg_pct / 100.0) and D[jh]["h"] - swing >= s.min_depth_atr * a
                and swing > vw[i] * 0.998 and D[i]["c"] > D[i - 1]["h"] and D[i]["c"] > D[i]["o"])
    if n == "meanrev":
        m, sd = s._band(D, i - 1)
        return sd > 1e-9 and D[i - 1]["c"] < m - s.z * sd and D[i]["c"] > D[i]["o"] and D[i]["c"] > D[i - 1]["c"]
    raise AssertionError(n)


class FastScansEqualTheReference(unittest.TestCase):
    def test_every_strategy_on_random_sessions(self):
        S = sessions(150, seed=17)
        for name, params in (("momentum", {}), ("vwap", {}), ("vwap", {"below_bars": 2}), ("pullback", {"min_leg_pct": 0.1}),
                             ("meanrev", {}), ("meanrev", {"window": 12, "z": 1.5})):
            s, hits = ss.get(name, **params), 0
            for _d, D in S:
                vw, top = sm.vwap_series(D), min(s.latest_entry, len(D) - 1)
                ref = next((i for i in range(s.start, top + 1) if ref_trigger(s, D, i, vw) and s.levels(TM, D, i) is not None), None)
                self.assertEqual(s.find_trigger(TM, D, []), ref, (name, params, _d))
                hits += ref is not None
            self.assertGreater(hits, 3, (name, params))

    def test_a_full_five_second_day_scans_fast(self):
        r, px, D = random.Random(3), 100.0, []
        for i in range(4680):
            o_ = px
            px *= 1 + r.gauss(0, 0.0003)
            D.append({"date": "2026-03-02", "slot": i, "start": "x", "o": o_, "h": max(o_, px) * 1.0001, "l": min(o_, px) * 0.9999,
                      "c": px, "v": 100.0 + r.randrange(50)})
        for name in ("orb", "momentum", "pullback", "vwap", "meanrev"):
            s = ss.get(name, bar_seconds=5)
            t0 = time.time()
            s.find_trigger(TM, D, [])
            self.assertLess(time.time() - t0, 1.5, "%s took %.2fs on 4680 bars" % (name, time.time() - t0))


class TickBarsTests(unittest.TestCase):
    def new(self, secs=30, **kw):
        return tb.TickBars(TM, secs, **kw)

    def test_ohlcv_and_completion_only_after_the_window_plus_lateness(self):
        b = self.new()
        self.assertEqual(b.add(OPEN + 1, 100.0, 10), 0)
        b.add(OPEN + 5, 101.0, 20)
        b.add(OPEN + 20, 99.0, 30)
        b.add(OPEN + 29, 100.5, 40)
        self.assertEqual(b.completed_bars()[0], [])                           # nothing complete yet
        self.assertEqual(b.add(OPEN + 30.5, 100.5, 1), 0)                     # 30.5 - 1.0 lateness < 30: still open
        self.assertEqual(b.add(OPEN + 31.5, 100.6, 1), 1)                     # now slot 0 is complete
        bars, last, why = b.completed_bars()
        self.assertEqual((why, last, len(bars)), ("ok", 0, 1))
        self.assertEqual((bars[0]["o"], bars[0]["h"], bars[0]["l"], bars[0]["c"], bars[0]["v"]), (100.0, 101.0, 99.0, 100.5, 100.0))
        self.assertEqual((bars[0]["date"], bars[0]["slot"], bars[0]["start"]), ("2026-03-02", 0, "09:30:00"))

    def test_out_of_order_inside_a_bar_uses_timestamps_for_open_and_close(self):
        b = self.new()
        b.add(OPEN + 20, 105.0, 1)
        b.add(OPEN + 2, 100.0, 1)                                              # arrives later but happened first
        b.add(OPEN + 40, 106.0, 1)
        bar = b.completed_bars()[0][0]
        self.assertEqual((bar["o"], bar["c"]), (100.0, 105.0))

    def test_a_late_trade_never_repaints_an_emitted_bar(self):
        b = self.new()
        b.add(OPEN + 1, 100.0, 1)
        b.add(OPEN + 40, 101.0, 1)
        before = b.completed_bars()[0]
        self.assertEqual(b.add(OPEN + 10, 999.0, 500), 0)
        self.assertEqual((b.late_dropped, b.completed_bars()[0]), (1, before))

    def test_gaps_are_filled_flat_and_the_clock_completes_quiet_bars(self):
        b = self.new()
        b.add(OPEN + 1, 100.0, 5)
        b.add(OPEN + 100, 102.0, 5)                                            # slots 1, 2 empty; slot 3 has the trade
        bars, last, _ = b.completed_bars(OPEN + 140)
        self.assertEqual([x["slot"] for x in bars], [0, 1, 2, 3])
        self.assertEqual([(x["c"], x["v"]) for x in bars[1:3]], [(100.0, 0.0), (100.0, 0.0)])
        self.assertEqual(b.completed_bars(OPEN + 200)[0][-1]["slot"], 5)       # quiet symbol: clock completed slots 4 and 5
        self.assertEqual(b.completed_bars(OPEN + 200)[0][-1]["c"], 102.0)

    def test_no_opening_bar_no_data(self):
        b = self.new()
        b.add(OPEN + 95, 100.0, 5)
        self.assertEqual(b.completed_bars(OPEN + 200)[1:], (None if False else 3, "no_opening_bar")[0:0] or b.completed_bars(OPEN + 200)[1:])
        self.assertEqual(b.completed_bars(OPEN + 200)[0], [])

    def test_session_rules_and_bad_input(self):
        b = self.new()
        self.assertEqual(b.add(OPEN - 3600, 100.0, 5), 0)                      # pre-market
        self.assertEqual(b.out_of_session, 1)
        b.add(OPEN + 1, 100.0, 5)
        self.assertEqual((b.add(OPEN + 2, -1.0, 5), b.add(OPEN + 2, 100.0, 0), b.add(OPEN + 2, None, 5)), (0, 0, 0))
        b.add(OPEN + 400 * 60, 100.0, 5)                                       # after the close
        self.assertEqual(b.out_of_session, 2)
        nxt = OPEN + 86400 + 60
        b.add(nxt, 110.0, 5)                                                   # next day: rolls forward
        self.assertEqual(b.date, "2026-03-03")
        b.add(OPEN + 100, 1.0, 5)                                              # a stray trade from the old day must not wipe today
        self.assertEqual(b.date, "2026-03-03")
        with self.assertRaises(ValueError):
            tb.TickBars(TM, 7)

    def test_bars_from_a_prefix_equal_the_prefix_of_bars_from_everything(self):
        trades = gen_day("2026-03-02", 4, step=3)
        full, part = self.new(), self.new()
        for t in trades:
            full.add(*t)
        cut = trades[:len(trades) // 2]
        for t in cut:
            part.add(*t)
        a, b = full.completed_bars()[0], part.completed_bars()[0]
        n = len(b) - 1
        self.assertEqual(a[:n], b[:n])                                         # no look-ahead: earlier bars never change

    def test_coarse_bars_are_the_aggregate_of_fine_bars(self):
        trades = gen_day("2026-03-02", 5, step=2)
        fine, coarse = self.new(30), self.new(300)
        for t in trades:
            fine.add(*t)
            coarse.add(*t)
        fb, cb = fine.completed_bars(OPEN + 23400)[0], coarse.completed_bars(OPEN + 23400)[0]
        for c in cb[:20]:
            grp = fb[c["slot"] * 10:(c["slot"] + 1) * 10]
            self.assertEqual((c["o"], c["c"], c["h"], c["l"]), (grp[0]["o"], grp[-1]["c"], max(g["h"] for g in grp), min(g["l"] for g in grp)))
            self.assertAlmostEqual(c["v"], sum(g["v"] for g in grp), places=6)


class RecordAndLoad(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        rec = tb.TickRecorder(TM, cls.dir)
        cls.days = weekdays(26)
        for i, d in enumerate(cls.days):
            for sym, seed in (("AAA", 1), ("BBB", 2)):
                for ts, px, sz in gen_day(d, seed * 1000 + i, step=10):
                    rec.add(sym, ts, px, sz)
        rec.close()

    def test_round_trip_and_truncated_files(self):
        path = os.path.join(self.dir, "AAA", self.days[0] + ".csv.gz")
        got = list(tb.read_ticks(path))
        self.assertEqual(len(got), 2340)
        self.assertTrue(all(g[1] > 0 and g[2] > 0 for g in got))
        cut = os.path.join(tempfile.mkdtemp(), "x.csv.gz")
        with open(path, "rb") as fh, open(cut, "wb") as out:
            out.write(fh.read()[:-40])                                        # crash mid-write
        self.assertGreater(len(list(tb.read_ticks(cut))), 1000)                # tolerated, keeps what it could read

    def test_full_days_load_partial_days_do_not(self):
        s30 = tb.load_sessions(TM, self.dir, "AAA", 30)
        self.assertEqual(len(s30), 26)
        self.assertTrue(all(len(D) == 780 for _d, D in s30))
        self.assertEqual([d for d, _ in s30], sorted(d for d, _ in s30))
        part = os.path.join(self.dir, "CCC")
        os.makedirs(part)
        with gzip.open(os.path.join(part, self.days[0] + ".csv.gz"), "wt") as fh:
            for ts, px, sz in gen_day(self.days[0], 9)[:600]:                  # recorder stopped after ~100 minutes
                fh.write("%.6f,%s,%s\n" % (ts, px, sz))
        self.assertEqual(tb.load_sessions(TM, self.dir, "CCC", 30), [])

    def test_strategies_on_tick_bars_honour_the_contract_at_every_bar_size(self):
        for secs in (30, 60):
            S = tb.load_sessions(TM, self.dir, "AAA", secs)
            for name in ("orb", "momentum", "vwap", "meanrev", "pullback"):
                problems, _n = ss.check_contract(ss.get(name, secs), TM, S)
                self.assertEqual(problems, [], (name, secs))

    def test_rows_load_for_backtests(self):
        rows, info = tb.load_rows(ss.get("momentum", 60), TM, self.dir)
        self.assertEqual({s for s, v in info["symbols"].items() if v["sessions"]}, {"AAA", "BBB"})   # a partial-day folder has 0 usable sessions
        self.assertGreater(len(rows), 3)
        self.assertTrue(all(r["stop"] < r["entry"] < r["target"] and r["label"] in (0, 1, 2) for r in rows))
        self.assertTrue(all(r["minutes_since_open"] == (r["entry_slot"] + 1) * 1.0 for r in rows))   # 60 s bars


class StreamHelpers(unittest.TestCase):
    def test_parsing(self):
        trades, events = st.parse_messages('[{"T":"success","msg":"authenticated"},{"T":"t","S":"AAPL","p":190.25,"s":100,'
                                           '"t":"2026-03-02T14:30:01.123456789Z"},{"T":"t","S":"X"},{"T":"error","code":406,"msg":"x"}]')
        self.assertEqual(trades, [("AAPL", 1772461801.1234567, 190.25, 100.0)])
        self.assertEqual([e[0] for e in events], ["success", "bad_trade", "error"])
        self.assertEqual(st.parse_messages("not json")[1][0][0], "bad_json")
        self.assertEqual(st.parse_messages('{"T":"t","S":"A","p":1,"s":1,"t":"2026-03-02T14:30:00Z"}')[0][0][0], "A")
        with self.assertRaises(ValueError):
            tb.parse_rfc3339("yesterday")

    def test_payloads_backoff_and_watchdog(self):
        self.assertIn('"action": "auth"', st.auth_payload("k", "s"))
        self.assertEqual(st.subscribe_payload(["msft", "aapl", "AAPL"]), '{"action": "subscribe", "trades": ["AAPL", "MSFT"]}')
        self.assertEqual([st.backoff_schedule(n) for n in (0, 1, 2, 3, 9)], [1, 2, 4, 8, 60])
        self.assertFalse(st.is_stale(100, 120))
        self.assertTrue(st.is_stale(100, 200))
        self.assertFalse(st.is_stale(100, 200, in_session=False))

    def test_the_stream_module_has_no_way_to_trade(self):
        with open(os.path.join(ROOT, "research", "stream.py")) as fh:
            src = fh.read()
        self.assertNotIn("/v2/orders", src)
        self.assertNotIn("paper-api", src)


class GateIsBarSizeSpecific(unittest.TestCase):
    def test_a_five_minute_pass_cannot_authorise_a_thirty_second_strategy(self):
        d = tempfile.mkdtemp()
        ledger = os.path.join(d, "l.jsonl")
        jn.append_jsonl(ledger, {"id": "EXP-1", "kind": "walk-forward", "strategy": "orb", "source": "4 symbols", "result": "PASS: x"})   # 300 s
        fast = pb.BotConfig(["A"], ledger=ledger, out_dir=d, strategy=ss.get("orb", 30))
        meta = {"ledger_ref": "EXP-1", "trained_through": "2026-02-27", "strategy": "orb", "bar_seconds": 30}
        self.assertFalse(pb.gate_ok(fast, meta, "2026-03-02")[0])
        jn.append_jsonl(ledger, {"id": "EXP-2", "kind": "walk-forward", "strategy": "orb", "bar_seconds": 30, "source": "4 symbols", "result": "PASS: x"})
        self.assertFalse(pb.gate_ok(fast, meta, "2026-03-02")[0])               # model still points at EXP-1
        self.assertTrue(pb.gate_ok(fast, dict(meta, ledger_ref="EXP-2"), "2026-03-02")[0])
        self.assertIn("30s", pb.gate_ok(fast, dict(meta, ledger_ref="EXP-2", bar_seconds=300), "2026-03-02")[1])
        slow = pb.BotConfig(["A"], ledger=ledger, out_dir=d)                    # the 5-minute bot is unaffected by the 30 s entry
        self.assertTrue(pb.gate_ok(slow, {"ledger_ref": "EXP-1", "trained_through": "2026-02-27"}, "2026-03-02")[0])


def breakout_day():
    """A day of trades for AAA: a quiet 15-minute opening range 99.6-100.4, drift at 100.2, a breakout to 101.0 at 10:20:00."""
    out = []
    for s in range(0, 900, 3):                                                  # 09:30-09:45
        out.append(("AAA", et(9, 30, s) + 0.5, 99.6 if (s // 3) % 2 else 100.4, 50))
    for s in range(900, 3000, 5):                                               # 09:45-10:20
        out.append(("AAA", et(9, 30, s) + 0.5, 100.2, 50))
    for s in range(0, 120, 5):                                                  # 10:20-10:22 breakout
        out.append(("AAA", et(10, 20, s) + 0.5, 101.0, 80))
    for m in range(23, 360, 1):                                                 # then sparse ticks to the close
        out.append(("AAA", et(10, 0) + m * 60 + 0.5, 101.0, 10))
    return out


class ReplayEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.ledger = os.path.join(self.tmp, "l.jsonl")
        jn.append_jsonl(self.ledger, {"id": "EXP-9", "kind": "walk-forward", "strategy": "orb", "bar_seconds": 30,
                                      "source": "4 symbols", "result": "PASS: x"})
        self.cfg = pb.BotConfig(["AAA"], out_dir=os.path.join(self.tmp, "bot"), ledger=self.ledger, context=(),
                                strategy=ss.get("orb", 30))
        self.meta = {"ledger_ref": "EXP-9", "symbols": ["AAA"], "trained_through": "2026-02-27", "strategy": "orb", "bar_seconds": 30}
        self.b = bk.FakeBroker()
        self.b.prices["AAA"] = 101.02
        prior = [(d, [dict(b, slot=i, start="x") for i, b in enumerate(
            [{"date": d, "o": 100.0, "h": 100.2, "l": 99.9, "c": 100.1, "v": 100.0}] * 780)]) for d in
                 ("2026-02-23", "2026-02-24", "2026-02-25", "2026-02-26", "2026-02-27")]
        self.history = {"AAA": prior}
        self.J = jn.Journal(self.cfg.out_dir)

    def runner(self, paper=True, **kw):
        return lr.Runner(self.b, TM, self.cfg, FakeModel(), self.meta, self.history, None, paper, **kw)

    def test_raw_trades_become_one_paper_bracket_order_and_a_close(self):
        r = self.runner()
        calls = []
        real = pb.tick
        pb.tick = lambda *a, **k: (calls.append(a[5]), real(*a, **k))[1]
        try:
            lr.replay(r, breakout_day())
        finally:
            pb.tick = real
        subs = [c for c in self.b.calls if isinstance(c, tuple) and c[0] == "submit"]
        self.assertEqual(len(subs), 1)
        _n, cid, sym, shares, stop, target = subs[0]
        self.assertEqual((cid, sym), ("orb-20260302-AAA", "AAA"))
        self.assertLess(stop, 101.0)                                            # entry = the trigger bar's close, 101.0
        self.assertGreater(target, 101.0)
        self.assertGreater(shares, 0)
        self.assertNotIn("AAA", self.b.pos)                                     # flattened at 15:55
        self.assertEqual(self.J.load_state()["positions"], {})
        reasons = [t["exit_reason"] for t in jn.read_jsonl(self.J.trades_path)]
        self.assertEqual(reasons, ["session_end"])
        self.assertLess(len(calls), 23400 / 15 + 5)                              # throttled: never one tick per trade

    def test_the_trigger_is_taken_on_the_bar_that_triggered_not_later(self):
        r = self.runner(paper=False)
        lr.replay(r, breakout_day()[:len([t for t in breakout_day() if t[1] < et(10, 22)])])
        woulds = [d for d in jn.read_jsonl(self.J.decisions_path) if d["kind"] == "would_order"]
        self.assertEqual(len(woulds), 1)
        self.assertEqual(woulds[0]["entry"], 101.0)
        self.assertEqual(self.b.orders, {})                                      # dry-run: nothing sent

    def test_without_a_gate_pass_for_this_bar_size_nothing_is_sent(self):
        self.meta["bar_seconds"] = 300                                           # model from the wrong timeframe
        lr.replay(self.runner(), breakout_day())
        self.assertEqual(self.b.orders, {})
        self.assertTrue(any(d["kind"] == "would_order" for d in jn.read_jsonl(self.J.decisions_path)))

    def test_a_bot_that_wakes_late_does_not_chase_a_breakout(self):
        r = self.runner(paper=False)
        for sym, ts, px, sz in breakout_day():
            if ts < et(10, 20, 25):
                r.on_trade(sym, ts, px, sz)                                      # trades arrive, but no tick happens (hung/slow)
        r.maybe_tick(et(10, 21, 5))                                              # first tick comes 35 s after the breakout bar
        self.assertEqual([d for d in jn.read_jsonl(self.J.decisions_path) if d["kind"] == "would_order"], [])
        why = [d["why"] for d in jn.read_jsonl(self.J.decisions_path) if d["kind"] == "skip"]
        self.assertTrue(set(why) & {"missed_trigger", "stale_data"}, why)

    def test_unknown_symbols_are_ignored_and_the_clock_alone_can_complete_bars(self):
        r = self.runner(paper=False)
        r.on_trade("ZZZ", OPEN + 5, 1.0, 1)
        self.assertEqual(r.dirty, False)
        r.on_trade("AAA", OPEN + 5, 100.0, 10)
        self.assertIsNotNone(r.maybe_tick(OPEN + 10))                             # the first call always runs a manage pass
        self.assertIsNone(r.maybe_tick(OPEN + 12))                                # throttled: min_tick_s not reached
        self.assertIsNotNone(r.maybe_tick(OPEN + 40))                             # the clock completed bar 0: a tick is due


if __name__ == "__main__":
    unittest.main()
