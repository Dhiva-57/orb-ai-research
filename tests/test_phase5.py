import os
import sys
import tempfile
import textwrap
import unittest
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import check_predictor as chk  # noqa: E402
import daily_signal as ds  # noqa: E402
import orb_dataset as o  # noqa: E402
import paper_bot as pb  # noqa: E402
import strategies as ss  # noqa: E402
from test_strategies import sessions  # noqa: E402

TM = o.load_pack()


def series(seed, n=900, ar=0.0, vol=0.012):
    """Daily bars whose next-day return = ar * today's return + noise. ar=0 is a pure random walk."""
    r = np.random.default_rng(seed)
    ret = np.zeros(n)
    for t in range(1, n):
        ret[t] = ar * ret[t - 1] + r.normal(0, vol)
    c = 100 * np.exp(np.cumsum(ret))
    o_ = np.r_[c[0], c[:-1]]
    return o_, c * 1.003, c * 0.997, c, r.integers(1_000_000, 2_000_000, n).astype(float)


class Features(unittest.TestCase):
    def test_a_row_depends_only_on_bars_up_to_it(self):
        o_, h, l, c, v = series(1, 400)
        full = ds.features(o_, h, l, c, v)
        cut = ds.features(o_[:300], h[:300], l[:300], c[:300], v[:300])
        np.testing.assert_allclose(full[:300][ds.WARMUP:], cut[ds.WARMUP:], rtol=1e-9, atol=1e-12)
        c2 = c.copy()
        c2[300:] *= 5.0                                                       # rewrite the future
        again = ds.features(o_, h, l, c2, v)
        np.testing.assert_allclose(full[ds.WARMUP:300], again[ds.WARMUP:300], rtol=1e-9, atol=1e-12)

    def test_warmed_up_rows_are_finite_and_scale_free(self):
        o_, h, l, c, v = series(2, 300)
        F = ds.features(o_, h, l, c, v)
        self.assertTrue(np.isfinite(F[ds.WARMUP:]).all())
        F10 = ds.features(o_ * 10, h * 10, l * 10, c * 10, v)                  # price level must not matter
        np.testing.assert_allclose(F[ds.WARMUP:], F10[ds.WARMUP:], rtol=1e-6, atol=1e-9)


class Evaluate(unittest.TestCase):
    def test_a_planted_pattern_is_found(self):
        r = ds.evaluate(*series(3, n=2500, ar=0.5))
        self.assertTrue(r["verdict"].startswith("EDGE"), r)
        self.assertGreater(r["accuracy_edge_ci95_pts"][0], 0)

    def test_random_walks_are_not_flagged(self):
        flagged = [s for s in range(12) if ds.evaluate(*series(100 + s, n=1200))["verdict"].startswith("EDGE")]
        self.assertLessEqual(len(flagged), 1, flagged)                         # ~2-3% false-positive rate by design

    def test_too_little_data_gives_no_verdict_and_says_why(self):
        r = ds.evaluate(*series(4, n=400))
        self.assertEqual(r["verdict"], "NO EDGE DEMONSTRATED")
        self.assertTrue(any("out-of-sample days" in w for w in r["why"]))
        self.assertEqual(ds.evaluate(*series(4, n=60))["oos_days"], 0)

    def test_training_never_uses_labels_from_the_future(self):
        # A label (close t+1) may only be used by a model trained at t0 >= t+1. Plant a series where the LAST day is wild:
        # predictions for earlier blocks must be identical with or without it.
        o_, h, l, c, v = series(5, n=700)
        a = ds.evaluate(o_, h, l, c, v, step=20)
        c2 = c.copy(); c2[-1] *= 3.0
        b = ds.evaluate(o_, h, l, c2, v, step=20)
        self.assertEqual(a["oos_days"], b["oos_days"])
        self.assertAlmostEqual(a["accuracy"], b["accuracy"], delta=100.0 / a["oos_days"] + 1e-9)   # at most the last day's own hit changes


class PUpAsof(unittest.TestCase):
    def test_none_until_enough_history_then_a_probability(self):
        o_, h, l, c, v = series(6, n=100)
        self.assertIsNone(ds.p_up_asof(o_, h, l, c, v))
        p = ds.p_up_asof(*series(6, n=400))
        self.assertTrue(0.0 <= p <= 1.0)

    def test_follows_a_planted_pattern(self):
        o_, h, l, c, v = series(7, n=800, ar=0.6)
        ups, downs = [], []
        for t in range(500, 800, 6):
            p = ds.p_up_asof(o_[:t + 1], h[:t + 1], l[:t + 1], c[:t + 1], v[:t + 1])
            (ups if c[t] > c[t - 1] else downs).append(p)
        self.assertGreater(np.mean(ups), np.mean(downs) + 0.05)

    def test_only_the_last_lookback_days_matter(self):
        a = series(8, n=600)
        b = tuple(x.copy() for x in a)
        for x in b[:4]:
            x[:100] *= 7.0                                                    # corrupt days older than the lookback window
        self.assertAlmostEqual(ds.p_up_asof(*a), ds.p_up_asof(*b), places=9)

    def test_sessions_become_correct_daily_bars(self):
        S = sessions(5, seed=3)
        o_, h, l, c, v = ds.daily_from_sessions(S)
        d, D = S[2]
        self.assertEqual((o_[2], c[2], h[2], l[2]), (D[0]["o"], D[-1]["c"], max(b["h"] for b in D), min(b["l"] for b in D)))
        self.assertEqual(v[2], sum(b["v"] for b in D))


class DailyVariants(unittest.TestCase):
    def test_registered_and_meant_to_load_long_history(self):
        for n in ("orb", "momentum", "pullback", "vwap", "meanrev"):
            self.assertIn(n + "_daily", ss.names())
            self.assertGreater(ss.get(n + "_daily").history_sessions, ds.LOOKBACK)
            self.assertEqual(ss.get(n).history_sessions, 25)

    def test_same_trades_plus_one_extra_feature(self):
        S = sessions(330, seed=12)
        base, daily = ss.get("orb").build(TM, S, symbol="X"), ss.get("orb_daily").build(TM, S, symbol="X")
        key = lambda r: (r["date"], r["entry"], r["stop"], r["target"], r["label"])
        self.assertEqual([key(r) for r in base], [key(r) for r in daily])      # the filter's candidates are unchanged
        late = [r for r in daily if r["date"] > S[ds.WARMUP + ds.MIN_TRAIN + 5][0]]
        self.assertTrue(late)
        self.assertTrue(all(r["ctx_daily_p_up"] is not None and 0 <= r["ctx_daily_p_up"] <= 1 for r in late))
        self.assertIn("ctx_daily_p_up", o.feature_columns(daily))
        self.assertNotIn("ctx_daily_p_up", o.feature_columns(base))

    def test_contract_holds(self):
        problems, triggered = ss.check_contract(ss.get("orb_daily"), TM, sessions(260, seed=5))
        self.assertEqual(problems, [])
        self.assertGreater(triggered, 10)

    def test_bot_loads_enough_history_for_a_daily_strategy(self):
        import json
        import gzip
        d = tempfile.mkdtemp()
        import alpaca_bars as ab
        import calendar
        import datetime
        raw = []
        for dt, D in sessions(340, seed=2):
            y, m, dd = map(int, dt.split("-"))
            t0 = calendar.timegm(TM.eastern_to_utc(datetime.datetime(y, m, dd, 9, 30)).timetuple())
            raw += [{"t": datetime.datetime.fromtimestamp(t0 + 300 * b["slot"], datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"], "v": b["v"]} for b in D]
        doc, _ = ab.convert(raw, "AAA", TM)
        ab.write_chart(os.path.join(d, "AAA.json.gz"), doc)
        self.assertEqual(len(pb.load_history(TM, d, ["AAA"])["AAA"]), 25)
        self.assertEqual(len(pb.load_history(TM, d, ["AAA"], ss.get("orb_daily").history_sessions)["AAA"]), 320)


FAKE = textwrap.dedent('''
    import numpy as np
    MODEL_DIR = None
    def train_ticker(ticker, hist):
        return {"direction_accuracy": %s}
''')


class CheckPredictor(unittest.TestCase):
    def repo(self, acc):
        d = tempfile.mkdtemp()
        app = os.path.join(d, "stock_system", "app")
        os.makedirs(app)
        with open(os.path.join(app, "ml_model.py"), "w") as fh:
            fh.write(FAKE % acc)
        return d

    def test_leaky_and_clean_models_are_told_apart(self):
        self.assertEqual(chk.main(["--repo", self.repo(100.0)]), 1)
        self.assertEqual(chk.main(["--repo", self.repo(51.2)]), 0)
        self.assertEqual(chk.main(["--repo", tempfile.mkdtemp()]), 2)

    def test_random_walks_are_deterministic_and_have_no_pattern(self):
        a, b = chk.random_walk(1), chk.random_walk(1)
        self.assertTrue(a.equals(b))
        r = np.log(a["close"]).diff().dropna()
        self.assertLess(abs(r.autocorr(1)), 0.12)


if __name__ == "__main__":
    unittest.main()
