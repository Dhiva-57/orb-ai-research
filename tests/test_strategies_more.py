import calendar
import datetime
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
import orb_dataset as o  # noqa: E402
import strategies as ss  # noqa: E402
import strategies_more as sm  # noqa: E402
from test_strategies import sessions  # noqa: E402

TM = o.load_pack()


def mk(path):
    D = []
    for i, (o_, h, l, c, v) in enumerate(path):
        mins = 570 + 5 * i
        D.append({"date": "2026-03-02", "slot": i, "start": "%02d:%02d" % (mins // 60, mins % 60),
                  "o": o_, "h": h, "l": l, "c": c, "v": float(v)})
    return D


def flat(n, px=100.0, v=1000):
    return [(px, px + 0.2, px - 0.2, px, v)] * n


class Momentum(unittest.TestCase):
    s = ss.get("momentum")

    def test_new_high_above_vwap_and_open_triggers(self):
        D = mk(flat(8) + [(100.0, 101.3, 100.0, 101.2, 1000)])
        self.assertEqual(self.s.find_trigger(TM, D, []), 8)
        e, stop, target = self.s.levels(TM, D, 8)
        self.assertEqual(e, 101.2)
        self.assertLess(stop, e)
        self.assertAlmostEqual(target - e, e - stop, delta=0.011)

    def test_flat_or_below_the_open_does_not(self):
        self.assertIsNone(self.s.find_trigger(TM, mk(flat(12)), []))
        down = mk([(100, 100.2, 99.8, 100, 1000)] + [(99.0, 99.2, 98.8, 99.0, 1000)] * 6 + [(99.0, 99.6, 99.0, 99.5, 1000)])
        self.assertIsNone(self.s.find_trigger(TM, down, []))                  # new high vs recent bars? no: below the day's open


class Vwap(unittest.TestCase):
    s = ss.get("vwap")

    def path(self):
        return [(100.0, 102.0, 100.0, 101.5, 10000)] + [(101.0, 101.0, 99.5, 100.0, 1000)] * 4

    def test_reclaim_after_three_closes_below(self):
        D = mk(self.path() + [(100.0, 101.5, 100.0, 101.2, 1000)])
        self.assertEqual(self.s.find_trigger(TM, D, []), 5)

    def test_red_bar_or_still_below_does_not(self):
        red = mk(self.path() + [(101.3, 101.5, 100.0, 101.0, 1000)])          # closes above vwap but red
        self.assertIsNone(self.s.find_trigger(TM, red, []))
        below = mk(self.path() + [(100.0, 100.8, 100.0, 100.6, 1000)])
        self.assertIsNone(self.s.find_trigger(TM, below, []))


class Pullback(unittest.TestCase):
    s = ss.get("pullback")

    def path(self, resume_close=100.9):
        return ([(100.0, 100.2, 99.9, 100.1, 1000)] * 3 +
                [(100.1, 100.8, 100.1, 100.6, 1000), (100.6, 101.2, 100.6, 101.0, 1000),
                 (101.0, 101.0, 100.5, 100.6, 1000), (100.6, 100.8, 100.4, 100.5, 1000),
                 (100.5, 101.0, 100.5, resume_close, 1000)])

    def test_up_leg_pullback_resumption(self):
        D = mk(self.path())
        self.assertEqual(self.s.find_trigger(TM, D, []), 7)
        e, stop, target = self.s.levels(TM, D, 7)
        self.assertEqual(stop, 100.39)                                        # swing low 100.4 minus a cent
        self.assertAlmostEqual(target - e, e - stop, delta=0.011)

    def test_no_resumption_no_leg_deep_break_no_trade(self):
        self.assertIsNone(self.s.find_trigger(TM, mk(self.path(resume_close=100.6)), []))      # not above the prior high
        self.assertIsNone(self.s.find_trigger(TM, mk(flat(12)), []))                            # no up-leg
        broke = self.path()
        broke[5] = (101.0, 101.0, 99.0, 99.6, 1000)                                             # pull-back loses VWAP
        self.assertIsNone(self.s.find_trigger(TM, mk(broke), []))


class MeanRev(unittest.TestCase):
    s = ss.get("meanrev")

    def base(self):
        return [(100.0, 100.3, 99.9, 100.0 if i % 2 == 0 else 100.2, 1000) for i in range(20)]

    def test_oversold_then_green_reversal(self):
        D = mk(self.base() + [(100.1, 100.1, 98.9, 99.0, 1000), (99.0, 99.5, 98.95, 99.4, 1000)])
        self.assertEqual(self.s.find_trigger(TM, D, []), 21)
        e, stop, target = self.s.levels(TM, D, 21)
        self.assertEqual(e, 99.4)
        self.assertEqual(stop, 98.89)                                            # lower of the two lows (98.90) minus a cent
        self.assertGreater(target, e)                                           # target = the 20-bar mean

    def test_a_reversal_that_is_already_above_the_mean_is_not_a_setup(self):
        # oversold bar, then a huge green bar that closes ABOVE the 20-bar mean: no room to the target, so no trigger
        D = mk(self.base() + [(100.1, 100.1, 98.9, 99.0, 1000), (99.0, 101.5, 98.95, 101.4, 1000)])
        self.assertIsNone(self.s.find_trigger(TM, D, []))

    def test_no_reversal_or_not_oversold_does_nothing(self):
        D = mk(self.base() + [(100.1, 100.1, 98.9, 99.0, 1000), (99.2, 99.2, 98.7, 98.8, 1000)])   # keeps falling
        self.assertIsNone(self.s.find_trigger(TM, D, []))
        self.assertIsNone(self.s.find_trigger(TM, mk(self.base() + [(100, 100.3, 99.9, 100.2, 1000)] * 3), []))


class Contract(unittest.TestCase):
    def test_every_registered_strategy_honours_the_contract_on_random_sessions(self):
        S = sessions(120, seed=21)
        for name in ss.names():
            problems, triggered = ss.check_contract(ss.get(name), TM, S)
            self.assertEqual(problems, [], name)
            self.assertGreater(triggered, 3, "%s never triggered in 120 random sessions: logic too tight to test" % name)

    def test_other_seeds_and_parameters(self):
        for name, params in (("momentum", {"stop_atr": 2.0, "target_r": 1.5}), ("vwap", {"below_bars": 2}),
                             ("pullback", {"min_leg_pct": 0.1}), ("meanrev", {"window": 12, "z": 1.5})):
            problems, _ = ss.check_contract(ss.get(name, **params), TM, sessions(80, seed=33, drift=0.0004))
            self.assertEqual(problems, [], name)

    def test_datasets_have_both_outcomes_and_valid_levels(self):
        S = sessions(150, seed=8)
        for name in ("momentum", "vwap", "pullback", "meanrev"):
            rows = ss.get(name).build(TM, S, symbol="X")
            self.assertGreater(len(rows), 5, name)
            self.assertTrue(all(r["stop"] < r["entry"] < r["target"] and r["label"] in (0, 1, 2) for r in rows), name)
            self.assertTrue(all(r["strategy"] == name for r in rows))

    def test_bot_can_run_each_strategy_and_rejects_nothing_new(self):
        import paper_bot as pb
        for name in ss.names():
            self.assertEqual(pb.BotConfig(["A"], strategy=ss.get(name)).strategy.name, name)


class DataPath(unittest.TestCase):
    """Alpaca-format bars -> converted chart files on disk -> strategy.load_dir -> rows: the route your real data takes."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        for sym, seed in (("AAA", 1), ("BBB", 2), ("SPY", 3)):
            raw = []
            for d, D in sessions(150, seed=seed):
                y, m, dd = map(int, d.split("-"))
                open_utc = TM.eastern_to_utc(datetime.datetime(y, m, dd, 9, 30))
                t0 = calendar.timegm(open_utc.timetuple())
                for b in D:
                    ts = datetime.datetime.fromtimestamp(t0 + 300 * b["slot"], datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    raw.append({"t": ts, "o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"], "v": b["v"]})
            doc, stats = ab.convert(raw, sym, TM)
            assert stats["kept"] == 150, stats
            ab.write_chart(os.path.join(cls.dir, sym + ".json.gz"), doc)

    def test_orb_through_the_plugin_equals_the_original_loader(self):
        old, _ = o.load_dir(TM, self.dir, ("SPY",))
        new, info = ss.get("orb").load_dir(TM, self.dir, ("SPY",))
        key = lambda r: (r["symbol"], r["date"], r["entry"], r["stop"], r["target"], r["label"])
        self.assertEqual(sorted(map(key, old)), sorted(map(key, new)))
        self.assertEqual(info["context"], ["SPY"])
        self.assertTrue(any(k.startswith("ctx_") for k in new[0]))             # context features flow through

    def test_every_strategy_loads_and_splits_chronologically(self):
        for name in ss.names():
            rows, info = ss.get(name).load_dir(TM, self.dir, ("SPY",))
            self.assertGreater(len(rows), 5, name)
            self.assertEqual(sorted(info["symbols"]), ["AAA", "BBB"])
            seg = o.chrono_split(rows)
            self.assertTrue(seg["train"] and seg["test"], name)
            self.assertLess(max(r["date"] for r in seg["train"]), min(r["date"] for r in seg["test"]))

    def test_a_model_can_be_fitted_for_a_non_orb_strategy(self):
        import model_store as ms
        rows, _ = ss.get("momentum").load_dir(TM, self.dir, ("SPY",))
        try:
            dep = ms.fit_deployable(rows, strategy="momentum")
        except ms.ModelError as e:                                             # noise data may give no usable threshold: acceptable
            self.assertIn("no model produced", str(e)) if "no model" in str(e) else self.assertIn("training trades", str(e))
            return
        self.assertEqual(dep.meta["strategy"], "momentum")
        self.assertTrue(0.0 <= dep.score(rows[-1]) <= 1.0)


if __name__ == "__main__":
    unittest.main()
