import datetime
import json
import os
import sys
import tempfile
import unittest
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
warnings.filterwarnings("ignore")
import broker as bk  # noqa: E402
import journal as jn  # noqa: E402
import orb_dataset as o  # noqa: E402
import paper_bot as pb  # noqa: E402
import trade_manager as tmgr  # noqa: E402

TM = o.load_pack()
TODAY = "2026-03-02"                        # a Monday, EST (UTC-5): the session opens 14:30 UTC
OPEN = datetime.datetime(2026, 3, 2, 14, 30)


def bar(date, slot, o_, h, l, c, v=1000.0):
    mins = 9 * 60 + 30 + 5 * slot
    return {"date": date, "slot": slot, "start": "%02d:%02d" % (mins // 60, mins % 60), "o": o_, "h": h, "l": l, "c": c, "v": v}


def day_bars(date, n, trigger=None):
    out = []
    for i in range(n):
        if i < 3:
            out.append(bar(date, i, 100.0, 100.5, 99.5, 100.0))
        elif i == trigger:
            out.append(bar(date, i, 100.3, 101.2, 100.2, 101.0))
        else:
            out.append(bar(date, i, 100.1, 100.4, 100.0, 100.2))
    return out


def history():
    return [(d, day_bars(d, 78)) for d in ("2026-02-23", "2026-02-24", "2026-02-25", "2026-02-26", "2026-02-27")]


def raw(D):
    return [{"t": (OPEN + datetime.timedelta(minutes=5 * b["slot"])).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"], "v": b["v"]} for b in D]


class FakeModel:
    cols, threshold = [], 0.5

    def __init__(self, take=True):
        self._take = take

    def take(self, row):
        return self._take, 0.7


def at(h, m):                                           # New York clock on TODAY -> naive UTC
    return datetime.datetime(2026, 3, 2, h + 5, m)


class Base(unittest.TestCase):
    sym = "AAA"

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.ledger = os.path.join(self.tmp, "ledger.jsonl")
        self.cfg = pb.BotConfig([self.sym], out_dir=os.path.join(self.tmp, "bot"), ledger=self.ledger, context=())
        self.J = jn.Journal(self.cfg.out_dir)
        self.b = bk.FakeBroker()
        self.D = day_bars(TODAY, 10, trigger=9)          # trigger on the latest completed bar at 10:20
        self.b.raw_bars[self.sym] = raw(self.D)
        self.b.prices[self.sym] = 101.05
        self.meta = {"ledger_ref": "EXP-001", "symbols": [self.sym], "trained_through": "2026-02-27"}

    def ledger_write(self, result="PASS: send to human review", source="3 symbols from research/bars", ref="EXP-001"):
        jn.append_jsonl(self.ledger, {"id": ref, "kind": "walk-forward", "source": source, "result": result})

    def tick(self, now=None, model=None, paper=False, meta=None, hist=True):
        return pb.tick(self.b, TM, self.cfg, model, meta if meta is not None else self.meta, now or at(10, 20), paper,
                       {self.sym: history()} if hist else {})

    def state(self):
        return self.J.load_state()

    def decisions(self, kind=None):
        r = jn.read_jsonl(self.J.decisions_path)
        return [x for x in r if kind is None or x["kind"] == kind]


class Bars(Base):
    def test_incomplete_bar_is_dropped_and_gaps_are_filled(self):
        r = raw(day_bars(TODAY, 10))
        del r[5]                                                          # one missing interior bar
        D, last, why = pb.live_bars(r, TM, at(10, 20), TODAY)
        self.assertEqual((len(D), last, why), (10, 9, "ok"))
        self.assertEqual(D[5]["v"], 0.0)
        self.assertEqual(D[5]["c"], D[4]["c"])
        D2, last2, _ = pb.live_bars(r, TM, at(10, 17), TODAY)             # bar 9 ends 10:20: not complete at 10:17
        self.assertEqual(last2, 8)

    def test_no_opening_bar_or_too_many_gaps_gives_nothing(self):
        r = raw(day_bars(TODAY, 10))
        self.assertEqual(pb.live_bars(r[1:], TM, at(10, 20), TODAY)[2], "no_opening_bar")
        self.assertEqual(pb.live_bars([r[0]] + r[8:], TM, at(10, 20), TODAY, max_gap=3)[2], "gappy_data")


class Candidate(Base):
    def test_trigger_on_latest_bar(self):
        row, why = pb.live_candidate(TM, self.sym, TODAY, self.D, history(), None, self.cfg)
        self.assertEqual(why, "trigger")
        self.assertEqual(row["entry"], 101.0)
        self.assertLess(row["stop"], row["entry"])
        self.assertAlmostEqual(row["target"] - row["entry"], row["risk"], places=2)

    def test_earlier_trigger_is_missed_and_none_is_none(self):
        D = day_bars(TODAY, 12, trigger=9)
        self.assertEqual(pb.live_candidate(TM, self.sym, TODAY, D, history(), None, self.cfg)[1], "missed_trigger")
        self.assertEqual(pb.live_candidate(TM, self.sym, TODAY, day_bars(TODAY, 10), history(), None, self.cfg)[1], "no_trigger")
        self.assertEqual(pb.live_candidate(TM, self.sym, TODAY, day_bars(TODAY, 3), history(), None, self.cfg)[1],
                         "opening_range_not_complete")

    def test_features_use_only_bars_up_to_the_trigger(self):
        row, _ = pb.live_candidate(TM, self.sym, TODAY, self.D, history(), None, self.cfg)
        longer = self.D + day_bars(TODAY, 20)[10:]
        for b in longer[10:]:
            b["c"], b["h"] = 500.0, 500.0                                  # wild future bars
        again = o.features_at(longer, 9, *TM.opening_range(longer, 3), history())
        for k in o.FEATURES:
            if k != "risk_pct":
                self.assertEqual(row[k], again[k], k)


class Gate(Base):
    def g(self):
        return pb.gate_ok(self.cfg, self.meta, TODAY)

    def test_every_way_to_be_refused(self):
        self.assertFalse(self.g()[0])                                     # no ledger at all
        self.ledger_write(source="SYNTHETIC EDGE (the true answer is known)")
        self.assertFalse(self.g()[0])                                     # a synthetic PASS never counts
        self.ledger_write(result="FAIL", ref="EXP-002")
        self.assertIn("FAIL", self.g()[1])
        self.ledger_write(ref="EXP-003")
        self.assertIn("rebuild the model", self.g()[1])                   # PASS, but not the one the model was built on
        self.meta["ledger_ref"] = "EXP-003"
        self.assertTrue(self.g()[0])
        self.meta["trained_through"] = "2025-01-01"
        self.assertIn("days old", self.g()[1])


class Tick(Base):
    def test_dry_run_logs_and_sends_nothing(self):
        out = self.tick(model=FakeModel())
        self.assertEqual(out["entries"][0]["result"], "would_order")
        self.assertEqual(out["entries"][0]["shares"], 24)
        self.assertFalse([c for c in self.b.calls if isinstance(c, tuple) and c[0] == "submit"])
        self.assertEqual(self.state()["positions"], {})
        self.assertEqual(self.tick(model=FakeModel())["entries"], [])      # not logged twice

    def test_paper_orders_need_the_gate(self):
        out = self.tick(model=FakeModel(), paper=True)                     # no PASS in the ledger
        self.assertFalse(out["orders_enabled"])
        self.assertEqual(out["entries"][0]["result"], "would_order")
        self.assertEqual(self.b.orders, {})

    def test_order_flow_when_gated_open(self):
        self.ledger_write()
        out = self.tick(model=FakeModel(), paper=True)
        self.assertEqual(out["entries"][0]["result"], "submitted")
        sub = [c for c in self.b.calls if isinstance(c, tuple) and c[0] == "submit"][0]
        self.assertEqual(sub[1:3], ("orb-20260302-AAA", "AAA"))
        self.assertEqual(self.state()["positions"]["AAA"]["status"], "submitted")
        self.assertEqual(self.state()["trades_today"], 1)
        self.tick(now=at(10, 25), model=FakeModel(), paper=True)           # next tick reconciles the fill
        self.assertEqual(self.state()["positions"]["AAA"]["status"], "open")
        self.assertEqual(len([c for c in self.b.calls if isinstance(c, tuple) and c[0] == "submit"]), 1)

    def test_intent_is_on_disk_before_the_order_leaves(self):
        self.ledger_write()
        J, cfg = self.J, self.cfg

        class Spy(bk.FakeBroker):
            def submit_bracket(s, *a):
                intents = [r for r in jn.read_jsonl(J.orders_path) if r["kind"] == "intent"]
                s.seen = (len(intents), J.load_state()["positions"]["AAA"]["status"])
                return super().submit_bracket(*a)

        spy = Spy()
        spy.raw_bars, spy.prices = self.b.raw_bars, self.b.prices
        self.b = spy
        self.tick(model=FakeModel(), paper=True)
        self.assertEqual(spy.seen, (1, "sent"))

    def test_rejected_order_is_not_kept(self):
        self.ledger_write()
        self.b.fail_submit = "rejected"
        out = self.tick(model=FakeModel(), paper=True)
        self.assertEqual(out["entries"][0]["result"], "rejected")
        self.assertEqual((self.state()["positions"], self.state()["trades_today"]), ({}, 0))

    def test_unclear_order_that_did_arrive_is_adopted_and_closed_not_resent(self):
        self.ledger_write()
        self.b.fail_submit = "unknown_but_placed"
        out = self.tick(model=FakeModel(), paper=True)
        self.assertEqual(out["entries"][0]["result"], "unknown")
        self.assertEqual(self.state()["positions"]["AAA"]["status"], "unknown")
        self.b.fail_submit = None
        self.tick(now=at(10, 25), model=FakeModel(), paper=True)
        self.assertEqual(self.state()["positions"]["AAA"]["status"], "open")           # found by client_order_id
        self.assertEqual(len([c for c in self.b.calls if isinstance(c, tuple) and c[0] == "submit"]), 1)
        self.tick(now=at(15, 56), model=FakeModel(), paper=True)                       # and the 15:55 pass closes it
        self.assertNotIn("AAA", self.b.pos)
        self.assertEqual(self.state()["positions"], {})

    def test_unclear_order_that_never_arrived_is_released_after_grace(self):
        self.ledger_write()
        self.b.fail_submit = "unknown_not_placed"
        self.tick(model=FakeModel(), paper=True)
        self.b.fail_submit = None
        self.tick(now=at(10, 20) + datetime.timedelta(seconds=10), model=FakeModel(), paper=True)
        self.assertIn("AAA", self.state()["positions"])                                # inside the grace period: wait
        self.tick(now=at(10, 25), model=FakeModel(), paper=True)
        self.assertNotIn("AAA", self.state()["positions"])

    def test_flatten_closes_own_positions_only(self):
        self.ledger_write()
        self.tick(model=FakeModel(), paper=True)
        self.b.pos["ZZZ"] = {"qty": 5, "entry": 50.0}                                  # not opened by the bot
        out = self.tick(now=at(15, 56), model=FakeModel(), paper=True)
        self.assertTrue(out["flatten"])
        self.assertNotIn("AAA", self.b.pos)
        self.assertIn("ZZZ", self.b.pos)
        self.assertNotIn(("close", "ZZZ"), self.b.calls)

    def test_failed_close_is_retried_next_tick(self):
        self.ledger_write()
        self.tick(model=FakeModel(), paper=True)
        self.b.close_fails = 1
        self.tick(now=at(15, 56), model=FakeModel(), paper=True)
        self.assertIn("AAA", self.state()["positions"])
        self.tick(now=at(15, 57), model=FakeModel(), paper=True)
        self.assertNotIn("AAA", self.b.pos)

    def test_broker_leg_exit_is_recorded(self):
        self.ledger_write()
        self.tick(model=FakeModel(), paper=True)
        self.tick(now=at(10, 25), model=FakeModel(), paper=True)
        self.b.hit_leg("AAA", "target")
        self.tick(now=at(10, 30), model=FakeModel(), paper=True)
        self.assertEqual(self.state()["positions"], {})
        self.assertEqual(self.state()["closed"][0]["reason"], "target")                 # the filled bracket leg is visible
        self.assertEqual(self.tick(now=at(10, 35), model=FakeModel(), paper=True)["entries"], [])   # one trade per symbol per day

    def test_kill_switch_blocks_entries_but_not_management(self):
        self.ledger_write()
        self.tick(model=FakeModel(), paper=True)
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        open(self.cfg.kill_file, "w").close()
        out = self.tick(now=at(15, 56), model=FakeModel(), paper=True)
        self.assertEqual(out["no_entries"], "kill_switch")
        self.assertNotIn("AAA", self.b.pos)

    def test_other_reasons_for_no_trade(self):
        self.b.is_open = False
        self.assertEqual(self.tick(model=FakeModel())["skipped"], "market_closed")
        self.b.is_open = True
        self.b.blocked = True
        self.assertEqual(self.tick(model=FakeModel())["no_entries"], "account_blocked")
        self.b.blocked = False
        self.b.next_close = "2026-03-02T13:00:00-05:00"                    # early close day
        self.assertEqual(self.tick(model=FakeModel())["no_entries"], "early_close_day")
        self.assertEqual(self.tick(now=at(12, 56), model=FakeModel())["no_entries"], "flatten_window")   # flatten = close - 5 min

    def test_veto_paths(self):
        self.assertEqual(self.tick(model=FakeModel(take=False))["entries"][0]["result"], "ai_veto")
        self.setUp()
        self.b.prices[self.sym] = 103.0
        self.assertEqual(self.tick(model=FakeModel())["entries"][0]["result"], "price_drift")
        self.setUp()
        self.cfg.risk.kill_switch = True
        self.assertEqual(self.tick(model=FakeModel())["entries"][0]["why"], "kill_switch")
        self.setUp()
        self.cfg.risk.max_open = 0
        self.assertEqual(self.tick(model=FakeModel())["entries"][0]["why"], "max_open")

    def test_stale_data_and_stale_history_and_unknown_symbol(self):
        self.b.raw_bars[self.sym] = raw(self.D)[:9]                         # the 10:15 bar has not arrived
        self.tick(model=FakeModel())
        self.assertEqual(self.decisions("skip")[-1]["why"], "stale_data")
        self.setUp()
        self.tick(model=FakeModel(), hist=False)
        self.assertEqual(self.decisions("skip")[-1]["why"], "stale_history")
        self.setUp()
        self.meta["symbols"] = ["OTHER"]
        self.tick(model=FakeModel())
        self.assertEqual(self.decisions("skip")[-1]["why"], "symbol_not_in_model")

    def test_foreign_positions_count_against_limits_but_are_never_touched(self):
        self.b.pos["ZZZ"] = {"qty": 1, "entry": 10.0}
        self.b.prices["ZZZ"] = 10.0
        self.cfg.risk.max_open = 1
        self.assertEqual(self.tick(model=FakeModel())["entries"][0]["why"], "max_open")
        self.assertIn("ZZZ", self.b.pos)

    def test_overnight_carry_is_flattened_and_blocks_entries(self):
        self.ledger_write()
        self.J.save_state({"date": "2026-02-27", "start_equity": 10000.0, "trades_today": 1, "closed": [], "dry_seen": [],
                           "positions": {"AAA": {"symbol": "AAA", "cid": "orb-20260227-AAA", "status": "open", "shares": 5,
                                                 "entry": 100.0, "stop": 99.0, "target": 101.0, "risk": 1.0, "date": "2026-02-27"}}})
        self.b.pos["AAA"] = {"qty": 5, "entry": 100.0}
        out = self.tick(model=FakeModel(), paper=True)
        self.assertNotIn("AAA", self.b.pos)
        self.assertEqual(out["no_entries"], "flatten_window")


class Manager(unittest.TestCase):
    pos = {"stop": 99.0, "target": 101.0}

    def test_both_touched_is_the_stop(self):
        d = tmgr.evaluate(self.pos, {"h": 101.5, "l": 98.5}, bracket_mode=False)
        self.assertEqual((d["action"], d["reason"]), ("close", "stop"))

    def test_software_mode_target_and_hold(self):
        self.assertEqual(tmgr.evaluate(self.pos, {"h": 101.1, "l": 99.5}, bracket_mode=False)["reason"], "target")
        self.assertIsNone(tmgr.evaluate(self.pos, {"h": 100.5, "l": 99.5}, bracket_mode=False)["action"])

    def test_bracket_mode_leaves_exits_to_the_broker_unless_the_stop_leg_failed(self):
        self.assertIsNone(tmgr.evaluate(self.pos, {"h": 101.5, "l": 98.9}, last_price=99.5)["action"])
        self.assertEqual(tmgr.evaluate(self.pos, None, last_price=98.5)["reason"], "stop_backstop")

    def test_session_end_and_timeout_win(self):
        self.assertEqual(tmgr.evaluate(self.pos, None, flatten=True)["reason"], "session_end")
        self.assertEqual(tmgr.evaluate(self.pos, None, minutes_held=95, max_hold_minutes=90)["reason"], "timeout")


class BrokerGuard(unittest.TestCase):
    def make(self, status=200, doc=None, env=None):
        self.urls = []

        def transport(method, url, headers, body=None):
            self.urls.append((method, url, body))
            return status, doc if doc is not None else {}

        return bk.AlpacaPaperBroker(env or {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}, transport)

    def test_needs_keys_from_the_environment(self):
        with self.assertRaises(bk.MissingKeys):
            bk.AlpacaPaperBroker({}, lambda *a: (200, {}))

    def test_only_paper_and_data_hosts_and_the_base_url_variable_is_ignored(self):
        b = self.make(env={"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s", "APCA_API_BASE_URL": "https://api.alpaca.markets"},
                      doc={"trade": {"p": 1}, "bars": []})
        b.account(); b.positions(); b.clock(); b.last_price("AAPL"); b.bars_5m("AAPL", "a", "b"); b.close_position("AAPL")
        for _m, url, _b in self.urls:
            self.assertTrue(url.startswith((bk.PAPER_HOST + "/", bk.DATA_HOST + "/")), url)
        with self.assertRaises(bk.Refused):               # host smuggling: https://paper-api...@evil.example/x
            b._call("GET", "@evil.example/x")

    def test_guard_rejects_a_foreign_host(self):
        b = self.make()
        b_url = "https://api.alpaca.markets/v2/orders"
        self.assertFalse(b_url.startswith((bk.PAPER_HOST + "/", bk.DATA_HOST + "/")))
        with open(os.path.join(ROOT, "research", "broker.py")) as fh:
            src = fh.read()
        self.assertNotIn("api.alpaca.markets/", src.replace("paper-api.alpaca.markets", ""))

    def test_bracket_payload_and_error_mapping(self):
        b = self.make(200, {"id": "x"})
        b.submit_bracket("cid1", "AAPL", 24, 99.99, 102.01)
        _m, url, body = self.urls[0]
        self.assertTrue(url.endswith("/v2/orders"))
        self.assertEqual((body["order_class"], body["side"], body["time_in_force"], body["qty"], body["client_order_id"]),
                         ("bracket", "buy", "day", "24", "cid1"))
        self.assertEqual((body["take_profit"]["limit_price"], body["stop_loss"]["stop_price"]), ("102.01", "99.99"))
        with self.assertRaises(bk.OrderRejected):
            self.make(422, {"message": "bad"}).submit_bracket("c", "A", 1, 1, 2)
        with self.assertRaises(bk.OrderUnknown):
            self.make(503, {}).submit_bracket("c", "A", 1, 1, 2)

        def boom(*a):
            raise bk.TransportError("timeout")
        with self.assertRaises(bk.OrderUnknown):
            bk.AlpacaPaperBroker({"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}, boom).submit_bracket("c", "A", 1, 1, 2)

    def test_orders_are_never_retried(self):
        b = self.make(503, {})
        with self.assertRaises(bk.OrderUnknown):
            b.submit_bracket("c", "A", 1, 1, 2)
        self.assertEqual(len([u for u in self.urls if u[0] == "POST"]), 1)

    def test_lookup_returns_none_on_404(self):
        self.assertIsNone(self.make(404, {}).order_by_client_id("nope"))


class Journal(unittest.TestCase):
    def test_atomic_state_and_jsonl_survive_a_corrupt_line(self):
        d = tempfile.mkdtemp()
        J = jn.Journal(d)
        J.save_state({"a": 1})
        self.assertEqual(J.load_state(), {"a": 1})
        self.assertFalse(os.path.exists(J.state_path + ".tmp"))
        J.decision("x", n=1)
        with open(J.decisions_path, "a") as fh:
            fh.write("{not json\n")
        J.decision("y")
        self.assertEqual([r["kind"] for r in jn.read_jsonl(J.decisions_path)], ["x", "corrupt_line", "y"])


if __name__ == "__main__":
    unittest.main()
