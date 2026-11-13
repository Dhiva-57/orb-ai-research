import copy
import os
import random
import sys
import unittest
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
warnings.filterwarnings("ignore")
import meta_models as m  # noqa: E402


class Harness(unittest.TestCase):
    def test_noise_is_not_called_an_edge(self):
        for seed in range(100, 106):
            rep = m.run(m.synthetic(edge=0.0, seed=seed), seed=seed)
            for name in ("logistic", "forest"):
                self.assertFalse(rep["models"][name]["verdict"].startswith("ADDS VALUE"), (seed, name))

    def test_a_planted_edge_is_learned(self):
        rep = m.run(m.synthetic(edge=1.0, seed=1), seed=1)
        self.assertGreater(rep["models"]["logistic"]["test_ml"]["auc"], 0.6)
        self.assertEqual(rep["models"]["rate"]["test_ml"]["auc"], 0.5)
        strong = m.run(m.synthetic(edge=2.0, seed=1), seed=1)
        self.assertTrue(strong["models"]["logistic"]["verdict"].startswith("ADDS VALUE"))

    def test_too_few_trades_fits_nothing_but_orb_only_still_runs(self):
        rep = m.run(m.synthetic(n_days=20, per_day=1, edge=1.0, seed=3))
        self.assertIn("no model is fitted", rep["note"])
        self.assertEqual(rep["models"], {})
        self.assertTrue(rep["_test_rows"])

    def test_test_labels_cannot_influence_the_fit_the_calibrator_or_the_threshold(self):
        rows = m.synthetic(edge=1.0, seed=5)
        a = m.run(rows, seed=5)
        td = {r["date"] for r in m.o.chrono_split(rows)["test"]}
        flipped = copy.deepcopy(rows)
        for r in flipped:
            if r["date"] in td:
                r["label"], r["R"] = 1 - r["label"], -r["R"]
        b = m.run(flipped, seed=5)
        for name in ("logistic", "forest"):
            for key in ("threshold", "calibration", "validation_ml"):
                self.assertEqual(a["models"][name][key], b["models"][name][key], (name, key))

    def test_timeouts_are_counted_in_trading_results_but_not_fitted(self):
        rows = m.synthetic(edge=1.0, seed=2)
        for r in rows[::7]:
            r["label"], r["R"] = 2, 0.25
        self.assertGreater(m.run(rows, seed=2)["orb_only_test"]["timeout"], 0)


class Calibration(unittest.TestCase):
    @staticmethod
    def overconfident(n, seed):
        rng = random.Random(seed)
        true_p = np.array([rng.uniform(0.2, 0.8) for _ in range(n)])
        y = np.array([1 if rng.random() < p else 0 for p in true_p])
        return 1 / (1 + np.exp(-np.log(true_p / (1 - true_p)) * 3.0)), y       # 3x too sure of itself

    def test_ece_basics(self):
        self.assertEqual(m.ece(np.array([1, 0, 1, 0]), np.array([1.0, 0.0, 1.0, 0.0])), 0.0)
        self.assertAlmostEqual(m.ece(np.array([1, 0] * 50), np.full(100, 0.9)), 0.4, places=3)
        self.assertIsNone(m.ece([], []))

    def test_platt_repairs_overconfidence_on_unseen_data(self):
        p_cal, y_cal = self.overconfident(600, 1)
        p_new, y_new = self.overconfident(2000, 2)
        self.assertLess(m.ece(y_new, m.Platt().fit(p_cal, y_cal)(p_new)), m.ece(y_new, p_new) / 2)

    def test_calibrator_is_never_worse_than_raw_on_validation(self):
        from sklearn.metrics import brier_score_loss
        p_cal, y_cal = self.overconfident(300, 3)
        p_val, y_val = self.overconfident(300, 4)
        cal = m.pick_calibrator(p_cal, y_cal, p_val, y_val)
        self.assertNotEqual(cal.name, "none")
        self.assertLessEqual(brier_score_loss(y_val, np.clip(cal(p_val), 1e-6, 1 - 1e-6)),
                             brier_score_loss(y_val, np.clip(p_val, 1e-6, 1 - 1e-6)))

    def test_honest_scores_are_left_alone_and_little_data_means_no_calibration(self):
        rng = random.Random(9)
        def draw(n):
            p = np.array([rng.uniform(0.2, 0.8) for _ in range(n)])
            return p, np.array([1 if rng.random() < q else 0 for q in p])
        p_cal, y_cal = draw(300)
        p_val, y_val = draw(300)
        self.assertIn(m.pick_calibrator(p_cal, y_cal, p_val, y_val).name, ("none", "platt"))
        p, y = self.overconfident(20, 5)
        self.assertEqual(m.pick_calibrator(p, y, p, y).name, "none")


class Bootstrap(unittest.TestCase):
    def test_day_clusters_widen_the_interval_versus_a_per_trade_bootstrap(self):
        rng = random.Random(0)
        rows, probs = [], []
        for d in range(40):                              # 5 trades a day sharing one outcome: ~40 independent draws
            good = rng.random() < 0.5
            for _ in range(5):
                rows.append({"date": "2026-02-%02d" % (d + 1) if d < 28 else "2026-03-%02d" % (d - 27),
                             "entry": 100.0, "risk": 1.0, "R": 1.0 if good else -1.0, "label": 1 if good else 0})
                probs.append(0.9 if d < 20 else 0.1)
        ci = m.bootstrap_gain(rows, probs, 0.5, 0.0, B=1500, seed=1)
        r2, R, take, diffs = random.Random(1), [r["R"] for r in rows], [p >= 0.5 for p in probs], []
        for _ in range(1500):
            idx = [r2.randrange(len(R)) for _ in R]
            t = [R[i] for i in idx if take[i]]
            if t:
                diffs.append(sum(t) / len(t) - sum(R[i] for i in idx) / len(R))
        diffs.sort()
        self.assertGreater(ci[1] - ci[0], 1.3 * (diffs[int(0.975 * len(diffs))] - diffs[int(0.025 * len(diffs))]))

    def test_no_taken_trades_gives_no_interval(self):
        rows = [{"date": "2026-02-%02d" % (i + 1), "entry": 100.0, "risk": 1.0, "R": 1.0, "label": 1} for i in range(10)]
        self.assertIsNone(m.bootstrap_gain(rows, [0.1] * 10, 0.9, 0.0, B=200))


    def test_replay_bootstrap_matches_baseline_when_every_candidate_is_selected(self):
        import backtest as bt

        rows = m.synthetic(n_days=40, per_day=2, edge=1.0, seed=17)
        cfg = bt.Config(slip_bps=0.0, stop_slip_bps=0.0,
                        commission_per_share=0.0)
        result = m.bootstrap_replay_gain(
            rows, [True] * len(rows), cfg, B=200, seed=1)

        self.assertEqual(result["gain_ci95_R"], [0.0, 0.0])
        self.assertIsNotNone(result["ai_ci95_R"])

    def test_replay_bootstrap_returns_no_interval_when_ai_takes_no_trades(self):
        import backtest as bt

        rows = m.synthetic(n_days=20, per_day=2, edge=1.0, seed=18)
        cfg = bt.Config(slip_bps=0.0, stop_slip_bps=0.0,
                        commission_per_share=0.0)
        result = m.bootstrap_replay_gain(
            rows, [False] * len(rows), cfg, B=200, seed=1)

        self.assertIsNone(result["gain_ci95_R"])
        self.assertIsNone(result["ai_ci95_R"])


if __name__ == "__main__":
    unittest.main()
