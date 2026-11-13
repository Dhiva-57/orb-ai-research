import copy
import os
import random
import sys
import tempfile
import unittest
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
warnings.filterwarnings("ignore")
import backtest as bt  # noqa: E402
import meta_models as mm  # noqa: E402


def cand(date="2026-03-02", sym="AAA", es=10, xs=20, entry=100.0, stop=99.0, target=101.0, label=0, R=-1.0):
    return {"date": date, "symbol": sym, "entry_slot": es, "exit_slot": xs, "entry": entry, "stop": stop,
            "target": target, "risk": entry - stop, "label": label, "R": R}


def acct(**kw):
    a = {"equity": 10000.0, "start_equity": 10000.0, "trades_today": 0, "open": [], "realized_today": 0.0}
    a.update(kw)
    return a


class RiskEngine(unittest.TestCase):
    cfg = bt.Config()

    def check(self, c=None, a=None, shares=10, cfg=None):
        return bt.risk_check(cfg or self.cfg, a or acct(), c or cand(), shares)

    def test_ok(self):
        self.assertEqual(self.check(), (True, "ok"))

    def test_kill_switch_beats_everything(self):
        self.assertEqual(self.check(cfg=bt.Config(kill_switch=True)), (False, "kill_switch"))

    def test_each_veto(self):
        pos = {"symbol": "X", "exit_slot": 99, "risk_usd": 1, "notional": 100}
        self.assertEqual(self.check(cand(es=61))[1], "too_late")
        self.assertEqual(self.check(cand(target=100.5))[1], "reward_risk")
        self.assertEqual(self.check(a=acct(trades_today=3))[1], "max_trades")
        self.assertEqual(self.check(a=acct(open=[pos, pos]))[1], "max_open")
        self.assertEqual(self.check(a=acct(open=[dict(pos, symbol="AAA")]))[1], "duplicate_symbol")
        self.assertEqual(self.check(shares=0)[1], "size_zero")
        self.assertEqual(self.check(a=acct(realized_today=-145.0))[1], "daily_loss_limit")      # 145 + 10 > 150
        self.assertEqual(self.check(a=acct(realized_today=-139.603))[0], True)  # exactly at the adjusted limit
        self.assertEqual(self.check(a=acct(realized_today=-140.0))[1], "daily_loss_limit")

    def test_daily_loss_uses_slippage_and_commission_adjusted_risk(self):
        cfg = bt.Config(commission_per_share=0.05)
        c = cand()
        self.assertAlmostEqual(bt.risk_per_share(cfg, c), 1.1397, places=7)
        self.assertEqual(
            self.check(c=c, a=acct(realized_today=-139.0), shares=10, cfg=cfg)[1],
            "daily_loss_limit"
        )
        self.assertEqual(self.check(shares=200)[1], "daily_loss_limit")
        self.assertEqual(self.check(a=acct(equity=500.0))[1], "buying_power")
        self.assertEqual(self.check(cand(entry=0.0, stop=-1.0))[1], "bad_prices")


class Sizing(unittest.TestCase):
    def test_loss_at_the_stop_never_exceeds_the_risk_budget_or_position_cap(self):
        rng, cfg = random.Random(4), bt.Config()
        for _ in range(500):
            entry = rng.uniform(5, 900)
            risk = entry * rng.uniform(0.002, 0.05)
            c = cand(entry=entry, stop=entry - risk, target=entry + risk)
            eq = rng.uniform(1000, 200000)
            n = bt.size(cfg, eq, c)
            ef, sf = entry * (1 + cfg.slip_bps / 1e4), c["stop"] * (1 - cfg.stop_slip_bps / 1e4)
            self.assertLessEqual(n * (ef - sf), cfg.risk_pct / 100 * eq + 1e-6)
            self.assertLessEqual(n * ef, cfg.max_position_pct / 100 * eq + 1e-6)

    def test_a_stop_too_far_away_gives_zero_shares_not_a_smaller_bet(self):
        self.assertEqual(bt.size(bt.Config(equity=1000), 1000.0, cand(entry=100, stop=20, target=180)), 0)


class Fills(unittest.TestCase):
    def test_slippage_costs_money_and_a_stop_loses_more_than_1r(self):
        cfg = bt.Config()
        win, loss = bt.pnl(cfg, cand(label=1, R=1.0), 10), bt.pnl(cfg, cand(label=0, R=-1.0), 10)
        self.assertGreater(win, 0)
        self.assertLess(loss, -10.0)
        self.assertAlmostEqual(bt.pnl(bt.Config(slip_bps=0, stop_slip_bps=0), cand(label=1, R=1.0), 10), 10.0, places=9)


class Replay(unittest.TestCase):
    def test_kill_switch_means_no_trades(self):
        res = bt.replay([cand(), cand(es=30, xs=40, sym="B")], bt.Config(kill_switch=True))
        self.assertEqual((len(res["trades"]), res["final_equity"]), (0, 10000.0))

    def test_replay_uses_adjusted_risk_for_open_positions(self):
        rows = [
            cand(es=5, xs=50, sym="A"),
            cand(es=6, xs=50, sym="B"),
        ]
        cfg = bt.Config(
            daily_loss_pct=0.547,
            commission_per_share=0.05,
            max_trades_day=9,
        )
        res = bt.replay(rows, cfg)

        self.assertEqual(len(res["trades"]), 1)
        self.assertEqual(res["refused"].get("daily_loss_limit"), 1)

    def test_daily_loss_limit_stops_the_day(self):
        rows = [cand(es=5 + 10 * i, xs=8 + 10 * i, sym="S%d" % i, entry=20.0, stop=19.0, target=21.0) for i in range(6)]
        res = bt.replay(rows, bt.Config(daily_loss_pct=1.0, max_trades_day=9))
        self.assertLessEqual(len(res["trades"]), 2)
        self.assertIn("daily_loss_limit", res["refused"])
        self.assertLess(10000.0 - res["final_equity"], 0.0125 * 10000.0)

    def test_max_open_is_honoured_and_a_finished_trade_frees_its_slot(self):
        rows = [cand(es=5, xs=50, sym="A"), cand(es=6, xs=50, sym="B"), cand(es=7, xs=50, sym="C")]
        res = bt.replay(rows, bt.Config(max_open=2))
        self.assertEqual((len(res["trades"]), res["refused"]), (2, {"max_open": 1}))
        rows = [cand(es=5, xs=10, sym="A"), cand(es=11, xs=30, sym="B"), cand(es=12, xs=30, sym="C"), cand(es=13, xs=30, sym="D")]
        self.assertEqual(len(bt.replay(rows, bt.Config(max_open=2, max_trades_day=9))["trades"]), 3)

    def test_input_order_does_not_matter(self):
        rows = mm.synthetic(edge=1.0, seed=3)[:300]
        a = bt.replay(rows, bt.Config())
        shuffled = copy.deepcopy(rows)
        random.Random(1).shuffle(shuffled)
        b = bt.replay(shuffled, bt.Config())
        self.assertAlmostEqual(a["final_equity"], b["final_equity"], places=6)

    def test_no_future_information_in_the_equity_curve(self):
        rows = sorted(mm.synthetic(edge=1.0, seed=4)[:300], key=lambda r: r["date"])
        full = bt.replay(rows, bt.Config())
        part = bt.replay([r for r in rows if r["date"] <= rows[150]["date"]], bt.Config())
        self.assertEqual(full["curve"][:len(part["curve"])], part["curve"])

    def test_filters_behave(self):
        rows = mm.synthetic(edge=1.0, seed=6)[:300]
        base = bt.replay(rows, bt.Config())
        self.assertEqual(base["final_equity"], bt.replay(rows, bt.Config(), take=[True] * len(rows))["final_equity"])
        oracle = bt.replay(rows, bt.Config(), take=[r["label"] == 1 for r in rows])        # cheating on purpose
        self.assertGreater(oracle["final_equity"], base["final_equity"])

    def test_equity_stays_positive_under_the_risk_rules(self):
        rows = [cand(es=5 + (i % 40), xs=60, sym="S%d" % i, date="2026-03-%02d" % (2 + i // 5 % 25)) for i in range(400)]
        res = bt.replay(rows, bt.Config())
        self.assertGreater(res["final_equity"], 0)
        self.assertLess(res["max_drawdown"], 1.0)


class GateAndExperiment(unittest.TestCase):
    good, base = {"trades": 60, "return_pct": 8.0, "expectancy_R": 0.3, "sharpe": 1.5}, {"trades": 200, "return_pct": 1.0}

    def test_pass_requires_everything(self):
        self.assertTrue(bt.gate(self.good, self.base, "ADDS VALUE over ORB-only")[0].startswith("PASS"))
        for bad, verdict in ((dict(self.good, trades=10), "ADDS VALUE"), (dict(self.good, expectancy_R=-0.1), "ADDS VALUE"),
                             (dict(self.good, return_pct=0.5), "ADDS VALUE"), (dict(self.good, sharpe=-1), "ADDS VALUE"),
                             (self.good, "NOT PROVEN: x")):
            self.assertEqual(bt.gate(bad, self.base, verdict)[0], "FAIL")

    def test_ledger_ids_increase(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "l.jsonl")
            self.assertEqual((bt.ledger_append(p, {"x": 1}), bt.ledger_append(p, {"x": 2})), ("EXP-001", "EXP-002"))

    def test_noise_does_not_pass_and_orb_only_runs_without_a_model(self):
        for seed in (1, 2, 3):
            self.assertEqual(bt.experiment(mm.synthetic(edge=0.0, seed=seed), bt.Config(), seed=seed)[0]["gate"][0], "FAIL")
        out, _ = bt.experiment(mm.synthetic(n_days=20, per_day=1, edge=1.0, seed=3), bt.Config())
        self.assertIsNotNone(out["orb_only"])
        self.assertIsNone(out["orb_ai"])


if __name__ == "__main__":
    unittest.main()
