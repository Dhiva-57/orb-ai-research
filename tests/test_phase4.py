import calendar
import datetime
import json
import os
import sys
import tempfile
import unittest
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import alpaca_bars as ab  # noqa: E402
import compare as cp  # noqa: E402
import journal as jn  # noqa: E402
import model_store as ms  # noqa: E402
import orb_dataset as o  # noqa: E402
import paper_bot as pb  # noqa: E402
import router as rt  # noqa: E402
import strategies as ss  # noqa: E402
from test_paper_bot import Base, FakeModel, at, TODAY  # noqa: E402
from test_strategies import sessions  # noqa: E402

TM = o.load_pack()


class TradeLog(Base):
    def trades(self):
        return jn.read_jsonl(self.J.trades_path)

    def open_position(self):
        self.ledger_write()
        self.tick(model=FakeModel(), paper=True)
        self.tick(now=at(10, 25), model=FakeModel(), paper=True)

    def test_target_leg_is_logged_with_its_price_and_reason(self):
        self.open_position()
        self.b.hit_leg("AAA", "target")
        self.tick(now=at(10, 30), model=FakeModel(), paper=True)
        (t,) = self.trades()
        self.assertEqual((t["symbol"], t["strategy"], t["exit_reason"], t["exit_source"]), ("AAA", "orb", "target", "bracket_leg"))
        self.assertEqual((t["fill_price"], t["exit_price"], t["shares"]), (101.05, t["target"], 24))
        self.assertGreater(cp.paper_r(t), 0)

    def test_stop_leg_and_unknown_exit(self):
        self.open_position()
        self.b.hit_leg("AAA", "stop")
        self.tick(now=at(10, 30), model=FakeModel(), paper=True)
        t = self.trades()[0]
        self.assertEqual((t["exit_reason"], t["exit_price"]), ("stop", t["stop"]))
        self.assertLess(cp.paper_r(t), 0)
        self.setUp()
        self.open_position()
        self.b.pos.pop("AAA")                                          # vanished with no leg information
        self.tick(now=at(10, 30), model=FakeModel(), paper=True)
        t = self.trades()[0]
        self.assertEqual((t["exit_reason"], t["exit_price"]), ("broker_exit", None))
        self.assertIsNone(cp.paper_r(t))

    def test_flatten_logs_the_price_before_closing(self):
        self.open_position()
        self.b.prices["AAA"] = 101.5
        self.tick(now=at(15, 56), model=FakeModel(), paper=True)
        t = self.trades()[0]
        self.assertEqual((t["exit_reason"], t["exit_price"], t["exit_source"]), ("session_end", 101.5, "last_price_before_close"))

    def test_orders_that_never_filled_are_not_trades(self):
        self.ledger_write()
        self.b.fail_submit = "rejected"
        self.tick(model=FakeModel(), paper=True)
        self.assertEqual(self.trades(), [])
        self.setUp()
        self.ledger_write()
        self.b.fail_submit = "unknown_not_placed"
        self.tick(model=FakeModel(), paper=True)
        self.b.fail_submit = None
        self.tick(now=at(10, 25), model=FakeModel(), paper=True)
        self.assertEqual(self.trades(), [])

    def test_trade_log_survives_the_daily_state_reset(self):
        self.open_position()
        self.b.hit_leg("AAA", "target")
        self.tick(now=at(10, 30), model=FakeModel(), paper=True)
        self.J.save_state(dict(self.state(), date="2026-03-01"))       # pretend a new day starts
        self.tick(now=at(10, 35), model=FakeModel(), paper=True)
        self.assertEqual(self.state()["closed"], [])
        self.assertEqual(len(self.trades()), 1)


def trade(sym="AAA", date="2026-03-02", fill=101.0, exit_=102.0, strat="orb", reason="target"):
    return {"symbol": sym, "date": date, "strategy": strat, "entry": 101.0, "stop": 100.0, "target": 102.0, "fill_price": fill,
            "exit_price": exit_, "exit_reason": reason}


def btrow(sym="AAA", date="2026-03-02", R=1.0, label=1, entry=101.0, stop=100.0, target=102.0):
    return {"symbol": sym, "date": date, "R": R, "label": label, "entry": entry, "stop": stop, "target": target}


class Compare(unittest.TestCase):
    def run_compare(self, trades, rows, decisions=(), dates=("2026-03-02",), min_trades=1):
        bt = {("orb", r["symbol"], r["date"]): r for r in rows}
        return cp.compare(trades, bt, list(decisions), set(dates), lambda d: d["date"], min_trades)

    def test_matching_trade(self):
        rep = self.run_compare([trade(fill=101.05)], [btrow()])
        self.assertEqual((rep["n_paper_trades"], rep["signal_parity_pct"], rep["paper_expectancy_R"]), (1, 100.0, 0.95))
        self.assertAlmostEqual(rep["mean_slippage_bps"], 4.95, places=1)
        self.assertEqual(rep["mean_delta_R"], -0.05)

    def test_signal_mismatch_and_no_history_are_reported_not_hidden(self):
        rep = self.run_compare([trade(), trade(sym="BBB")], [btrow(entry=101.5), btrow(sym="CCC")])
        self.assertEqual(rep["signal_parity_pct"], 0.0)
        self.assertEqual(rep["paper_trades_without_history"], 1)
        text = " ".join(rep["notes"])
        self.assertIn("Signal parity", text)
        self.assertIn("NO historical counterpart", text)

    def test_missed_candidates_get_a_reason_and_the_veto_is_scored(self):
        rows = [btrow(), btrow("BBB", R=-1.0, label=0), btrow("CCC", R=1.0), btrow("DDD", date="2026-03-03")]
        dec = [{"symbol": "BBB", "kind": "ai_veto", "date": "2026-03-02"}, {"symbol": "CCC", "kind": "skip", "why": "stale_data", "date": "2026-03-02"}]
        rep = self.run_compare([trade()], rows, dec)
        self.assertEqual(rep["missed_by_reason"], {"ai_veto": 1, "stale_data": 1})   # DDD is on a day the bot did not run
        self.assertEqual(rep["missed_ai_veto_backtest_expectancy_R"], -1.0)

    def test_candidate_with_no_decision_logged_is_flagged(self):
        rep = self.run_compare([], [btrow("ZZZ")])
        self.assertEqual(rep["missed_by_reason"], {"no_decision_logged": 1})

    def test_too_few_trades_says_so_and_underperformance_is_only_called_with_enough(self):
        rep = self.run_compare([trade(exit_=100.0, reason="stop")], [btrow()], min_trades=30)
        self.assertIn("too early to conclude", rep["notes"][0])
        self.assertFalse(any("trails the backtest" in n for n in rep["notes"]))
        rep = self.run_compare([trade(exit_=100.0, reason="stop")], [btrow()], min_trades=1)
        self.assertTrue(any("trails the backtest" in n for n in rep["notes"]))

    def test_unfilled_trades_are_skipped_and_unknown_exits_excluded_from_r(self):
        rep = self.run_compare([trade(fill=None), trade(exit_=None)], [btrow()])
        self.assertEqual(rep["n_paper_trades"], 1)
        self.assertIsNone(rep["paper_expectancy_R"])
        self.assertTrue(any("no recoverable exit price" in n for n in rep["notes"]))
        self.assertIn("PAPER vs BACKTEST", cp.render(rep))


class CompareCli(unittest.TestCase):
    def test_end_to_end_from_files_on_disk(self):
        d, out = tempfile.mkdtemp(), tempfile.mkdtemp()
        for sym, seed in (("AAA", 1), ("SPY", 3)):
            raw = []
            for dt, D in sessions(150, seed=seed):
                y, m, dd = map(int, dt.split("-"))
                t0 = calendar.timegm(TM.eastern_to_utc(datetime.datetime(y, m, dd, 9, 30)).timetuple())
                raw += [{"t": datetime.datetime.fromtimestamp(t0 + 300 * b["slot"], datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                         "o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"], "v": b["v"]} for b in D]
            doc, _ = ab.convert(raw, sym, TM)
            ab.write_chart(os.path.join(d, sym + ".json.gz"), doc)
        rows, _ = ss.get("orb").load_dir(TM, d, ("SPY",))
        r = [x for x in rows if x["symbol"] == "AAA"][10]
        y, m, dd = map(int, r["date"].split("-"))
        J = jn.Journal(out)
        J.trade(symbol="AAA", strategy="orb", date=r["date"], entry=r["entry"], stop=r["stop"], target=r["target"],
                fill_price=r["entry"], exit_price=r["target"] if r["label"] == 1 else r["stop"], exit_reason="x", shares=10)
        with open(J.decisions_path, "w") as fh:
            at_utc = TM.eastern_to_utc(datetime.datetime(y, m, dd, 10, 20)).replace(tzinfo=datetime.timezone.utc)
            fh.write(json.dumps({"kind": "skip", "symbol": "ZZZ", "at": at_utc.isoformat()}) + "\n")
        self.assertEqual(cp.main(["--out", out, "--bars-dir", d, "--context", "SPY"]), 0)
        rep = json.load(open(os.path.join(out, "compare.json")))
        self.assertEqual((rep["n_paper_trades"], rep["n_with_history"], rep["signal_parity_pct"]), (1, 1, 100.0))
        if r["label"] != 2:
            self.assertEqual(rep["trades"][0]["paper_R"], round(r["R"], 3))


class PStub(ss.Strategy):
    """A strategy that always has a candidate on the latest bar, with a chosen reward multiple."""
    bar_seconds = 300

    def __init__(self, name, reward=1.0):
        super().__init__()
        self.name, self.reward = name, reward

    def live_candidate(self, tm, symbol, date, D, prior, context=None):
        return {"date": date, "entry_slot": len(D) - 1, "entry": 101.0, "stop": 100.0, "target": 101.0 + self.reward,
                "risk": 1.0, "symbol": symbol, "strategy": self.name}, "trigger"


class PModel:
    cols, threshold = [], 0.5

    def __init__(self, p):
        self.p = p

    def take(self, row):
        return self.p >= self.threshold, self.p


class Routing(Base):
    def option(self, name, p, reward=1.0, ledger=True):
        if ledger:
            jn.append_jsonl(self.ledger, {"id": "EXP-" + name, "kind": "walk-forward", "strategy": name, "source": "4 symbols", "result": "PASS: x"})
        return rt.Option(PStub(name, reward), PModel(p) if p is not None else None,
                         {"symbols": [self.sym], "trained_through": "2026-02-27", "ledger_ref": "EXP-" + name, "strategy": name})

    def go(self, options, paper=True, now=None):
        return pb.tick(self.b, TM, self.cfg, None, {}, now or at(10, 20), paper, {self.sym: __import__("test_paper_bot").history()},
                       router=rt.Router(options))

    def test_ev_formula_and_ranking(self):
        self.assertAlmostEqual(rt.ev_r(0.6, {"entry": 100, "stop": 99, "target": 102, "risk": 1}), 0.8)
        self.assertAlmostEqual(rt.ev_r(0.7, {"entry": 100, "stop": 99, "target": 101, "risk": 1}), 0.4)
        a, b = {"ev": 0.4, "opt": rt.Option(PStub("a")), "cand": {}, "p": 0.7}, {"ev": 0.8, "opt": rt.Option(PStub("b")), "cand": {}, "p": 0.6}
        self.assertEqual([x["opt"].name for x in rt.rank([a, b])], ["b", "a"])      # higher EV beats higher probability

    def test_best_ev_wins_and_the_ranking_is_logged(self):
        out = self.go([self.option("alpha", 0.7, 1.0), self.option("beta", 0.6, 2.0)])
        self.assertEqual((out["entries"][0]["strategy"], out["entries"][0]["result"]), ("beta", "submitted"))
        self.assertIn("beta-20260302-AAA", self.b.orders)
        self.assertEqual(self.decisions("router_ranking")[0]["order"][0][0], "beta")
        self.assertEqual(self.state()["positions"]["AAA"]["strategy"], "beta")

    def test_one_position_per_symbol_per_day_even_if_both_agree(self):
        opts = [self.option("alpha", 0.7), self.option("beta", 0.6)]
        self.go(opts)
        self.go(opts, now=at(10, 25))
        self.assertEqual(len([c for c in self.b.calls if isinstance(c, tuple) and c[0] == "submit"]), 1)

    def test_a_strategy_without_its_own_pass_only_watches_when_orders_are_requested(self):
        out = self.go([self.option("alpha", 0.7, 1.0), self.option("beta", 0.9, 2.0, ledger=False)])   # beta ranks first, no PASS
        self.assertEqual((out["entries"][0]["strategy"], out["entries"][0]["result"]), ("alpha", "submitted"))
        self.assertEqual(self.decisions("shadow_candidate")[0]["strategy"], "beta")
        self.assertEqual(out["gate"]["alpha"], "ok")
        self.assertIn("walk-forward", out["gate"]["beta"])

    def test_dry_run_lets_everyone_compete_and_nothing_is_sent(self):
        out = self.go([self.option("alpha", 0.7), self.option("beta", 0.9, 2.0, ledger=False)], paper=False)
        self.assertEqual((out["entries"][0]["strategy"], out["entries"][0]["result"]), ("beta", "would_order"))
        self.assertEqual(self.b.orders, {})

    def test_risk_veto_falls_through_to_the_runner_up(self):
        out = self.go([self.option("poor", 0.9, 0.5), self.option("good", 0.6, 1.0)])     # poor: 0.35 EV but 0.5R reward < 1R minimum
        self.assertEqual((out["entries"][0]["strategy"], out["entries"][0]["result"]), ("good", "submitted"))
        self.assertEqual(self.decisions("risk_veto")[0]["why"], "reward_risk")

    def test_ai_veto_by_one_model_does_not_block_another(self):
        out = self.go([self.option("no", 0.3), self.option("yes", 0.6)])
        self.assertEqual(out["entries"][0]["strategy"], "yes")
        self.assertEqual(self.decisions("ai_veto")[0]["strategy"], "no")

    def test_everyone_vetoed_is_logged_once_and_not_retried(self):
        out = self.go([self.option("a", 0.2), self.option("b", 0.3)])
        self.assertEqual(out["entries"][0]["result"], "ai_veto")
        self.assertEqual(self.go([self.option("a", 0.2), self.option("b", 0.3)], now=at(10, 25))["entries"], [])

    def test_model_less_strategy_is_dry_run_only(self):
        out = self.go([self.option("solo", None)], paper=True)
        self.assertEqual(out["entries"][0]["result"], "would_order")
        self.assertEqual(self.b.orders, {})

    def test_construction_rules_and_loading(self):
        with self.assertRaises(ValueError):
            rt.Router([rt.Option(PStub("x")), rt.Option(PStub("x"))])
        fast = PStub("fast")
        fast.bar_seconds = 30
        with self.assertRaises(ValueError):
            rt.Router([rt.Option(PStub("slow")), rt.Option(fast)])
        root = tempfile.mkdtemp()
        import meta_models as mm
        ms.save(ms.fit_deployable(mm.synthetic(n_days=500, edge=2.0, seed=3), strategy="momentum"), os.path.join(root, "momentum"))
        opts = rt.load_options(["momentum", "vwap"], root)
        self.assertIsNotNone(opts[0].model)
        self.assertEqual(opts[0].meta["strategy"], "momentum")
        self.assertIsNone(opts[1].model)                                  # no model dir: dry-run only


if __name__ == "__main__":
    unittest.main()
