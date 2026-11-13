import copy
import os
import sys
import unittest
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
warnings.filterwarnings("ignore")
import backtest as bt  # noqa: E402
import meta_models as mm  # noqa: E402
import walkforward as wf  # noqa: E402

SHAPE = dict(train=100, cal=25, val=25, test=40, embargo=1)


class Windows(unittest.TestCase):
    def test_geometry_is_ordered_embargoed_and_test_periods_never_overlap(self):
        d = mm.datetime_days(500)
        plan = wf.window_plan(d, 100, 25, 25, 40, 1)
        self.assertEqual(len(plan), (500 - (100 + 25 + 25 + 40 + 3)) // 40 + 1)
        seen = set()
        for seg in plan:
            order = [seg[k] for k in ("train", "calibration", "validation", "test")]
            self.assertEqual([len(x) for x in order], [100, 25, 25, 40])
            for a, b in zip(order, order[1:]):
                self.assertGreaterEqual(d.index(b[0]) - d.index(a[-1]), 2)       # at least one embargo day
            self.assertLess(max(seg["train"] + seg["calibration"] + seg["validation"]), min(seg["test"]))
            self.assertFalse(seen & set(seg["test"]))
            seen |= set(seg["test"])

    def test_too_little_data_is_an_error_not_a_crash(self):
        res = wf.walk_forward(mm.synthetic(n_days=50, edge=1.0, seed=1), bt.Config(), **SHAPE)
        self.assertIn("not enough data", res["error"])
        self.assertEqual(wf.wf_gate(res)[0], "FAIL")


class NoLeakage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = mm.synthetic(n_days=400, edge=2.0, seed=2)
        cls.base = wf.walk_forward(cls.rows, bt.Config(), **SHAPE)

    def flipped(self, keep):
        out = copy.deepcopy(self.rows)
        for r in out:
            if not keep(r["date"]):
                r["label"], r["R"] = 1 - r["label"], -r["R"]
        return out

    def test_the_future_cannot_change_an_earlier_window(self):
        end = self.base["windows"][0]["test_to"]
        res = wf.walk_forward(self.flipped(lambda d: d <= end), bt.Config(), **SHAPE)
        self.assertEqual(self.base["windows"][0], res["windows"][0])

    def test_a_windows_own_test_labels_cannot_change_its_fit_or_threshold(self):
        w = self.base["windows"][1]
        res = wf.walk_forward(self.flipped(lambda d: not (w["test_from"] <= d <= w["test_to"])), bt.Config(), **SHAPE)
        for key in ("model", "threshold", "calibration"):
            self.assertEqual(w[key], res["windows"][1][key], key)
        self.assertNotEqual(w["orb_expectancy_R"], res["windows"][1]["orb_expectancy_R"])      # the flip did reach the test set


class Verdicts(unittest.TestCase):
    def test_noise_does_not_pass_and_a_strong_edge_does(self):
        noise = wf.walk_forward(mm.synthetic(n_days=400, edge=0.0, seed=11), bt.Config(), seed=11, **SHAPE)
        self.assertEqual(wf.wf_gate(noise)[0], "FAIL")
        edge = wf.walk_forward(mm.synthetic(n_days=400, edge=2.0, seed=3), bt.Config(), seed=3, **SHAPE)
        self.assertTrue(wf.wf_gate(edge)[0].startswith("PASS"), wf.wf_gate(edge))
        self.assertGreater(edge["gain_ci95_R"][0], 0)

    def test_a_window_without_a_usable_model_trades_nothing(self):
        res = wf.walk_forward(mm.synthetic(n_days=200, per_day=1, edge=2.0, seed=4), bt.Config(),
                              train=20, cal=10, val=10, test=40, embargo=1, seed=4)
        self.assertTrue(all(w["model"] is None and w["ai_takes"] == 0 for w in res["windows"]))
        self.assertEqual(res["orb_ai"]["trades"], 0)

    def test_one_lucky_window_is_not_enough_and_the_gate_needs_enough_windows_and_trades(self):
        good = {"trades": 90, "return_pct": 20.0, "expectancy_R": 0.4, "sharpe": 2.0}
        base = {"trades": 300, "return_pct": 1.0}
        win = lambda a, o: {"ai_expectancy_R": a, "orb_expectancy_R": o}            # noqa: E731
        lucky = {"orb_ai": good, "orb_only": base, "gain_ci95_R": [0.1, 0.5],
                 "windows": [win(5.0, 0.0)] + [win(-0.1, 0.0)] * 4}
        self.assertTrue(any("lucky" in r for r in wf.wf_gate(lucky)[1]))
        self.assertTrue(wf.wf_gate(dict(lucky, windows=[win(0.2, 0.0)] * 5))[0].startswith("PASS"))
        few = {"orb_ai": dict(good, trades=10), "orb_only": base, "gain_ci95_R": [0.1, 0.5], "windows": [win(0.2, 0.0)] * 2}
        reasons = wf.wf_gate(few)[1]
        self.assertTrue(any("windows" in r for r in reasons) and any("trades" in r for r in reasons))


if __name__ == "__main__":
    unittest.main()
