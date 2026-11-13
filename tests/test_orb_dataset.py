import copy
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
import orb_dataset as o  # noqa: E402

TM = o.load_pack()
FIXTURE = None
for base in (os.environ.get("DAY_TRADER_DIR"), "/tmp/day-trader-skill", os.path.join(ROOT, "..", "day-trader-skill")):
    p = base and os.path.join(base, "tests", "fixtures", "trader-timemachine", "TSLA-60d5m.json.gz")
    if p and os.path.exists(p):
        FIXTURE = p
        break


def bar(date, slot, o_, h, l, c, v=1000):
    mins = 9 * 60 + 30 + 5 * slot
    return {"date": date, "slot": slot, "start": "%02d:%02d" % divmod(mins, 60), "o": o_, "h": h, "l": l, "c": c, "v": v}


def flat_day(date, price=100.0):
    return [bar(date, i, price, price + 0.2, price - 0.2, price) for i in range(78)]


def breakout_day(date, after):
    """Opening range 99.8..100.2; bar 3 closes at 101 (stop 99.99, risk ~1.01); `after` = (high, low, close) bars."""
    D = flat_day(date)
    D[3] = bar(date, 3, 100.2, 101.2, 100.1, 101.0, v=3000)
    last = 101.0
    for i, (h, l, c) in enumerate(after):
        D[4 + i] = bar(date, 4 + i, last, h, l, c)
        last = c
    for i in range(4 + len(after), 78):
        D[i] = bar(date, i, last, last + 0.01, last - 0.01, last)
    return D


class Labels(unittest.TestCase):
    def cand(self, D):
        return o.candidate(TM, D[0]["date"], D, [])

    def test_target_first(self):
        r = self.cand(breakout_day("2026-01-05", [(102.2, 100.9, 102.0)]))
        self.assertEqual((r["label"], r["R"], r["entry"]), (1, 1.0, 101.0))

    def test_stop_first(self):
        r = self.cand(breakout_day("2026-01-05", [(101.1, 99.0, 99.5)]))
        self.assertEqual((r["label"], r["R"]), (0, -1.0))

    def test_one_bar_touching_both_counts_as_the_stop(self):
        self.assertEqual(self.cand(breakout_day("2026-01-05", [(103.0, 99.0, 101.0)]))["label"], 0)

    def test_timeout_pays_the_close(self):
        r = self.cand(breakout_day("2026-01-05", [(101.3, 100.7, 101.1)]))
        self.assertEqual(r["label"], 2)
        self.assertAlmostEqual(r["R"], (101.1 - 101.0) / r["risk"], places=6)

    def test_highs_and_lows_not_closes_decide_it(self):
        self.assertEqual(self.cand(breakout_day("2026-01-05", [(102.5, 100.9, 101.2)]))["label"], 1)

    def test_no_breakout_no_candidate(self):
        self.assertIsNone(self.cand(flat_day("2026-01-05")))


class NoLookahead(unittest.TestCase):
    def test_features_ignore_everything_after_the_trigger_bar(self):
        D = breakout_day("2026-01-06", [(102.5, 100.9, 101.2), (99.0, 90.0, 91.0)])
        prior = [("2026-01-05", flat_day("2026-01-05"))]
        a = o.candidate(TM, "2026-01-06", D, prior)
        D2 = copy.deepcopy(D)
        for i in range(a["entry_slot"] + 1, 78):
            D2[i] = bar("2026-01-06", i, 1.0, 999.0, 0.5, 500.0, v=10 ** 9)
        b = o.candidate(TM, "2026-01-06", D2, prior)
        for f in o.FEATURES:
            self.assertEqual(a[f], b[f], f)

    def test_features_ignore_later_sessions(self):
        D = breakout_day("2026-01-06", [(101.3, 100.7, 101.1)])
        prior = [("2026-01-05", flat_day("2026-01-05"))]
        a = o.candidate(TM, "2026-01-06", D, prior)
        rows = o.build(TM, prior + [("2026-01-06", D), ("2026-01-07", breakout_day("2026-01-07", [(99, 50, 60)]))])
        b = [r for r in rows if r["date"] == "2026-01-06"][0]
        for f in o.FEATURES:
            self.assertEqual(a[f], b[f], f)

    @unittest.skipUnless(FIXTURE, "needs the day-trader pack's TSLA fixture")
    def test_every_real_feature_is_reproducible_from_truncated_bars(self):
        sessions = o.load_sessions(TM, FIXTURE, "TSLA")
        rows = o.build(TM, sessions)
        self.assertEqual(len(rows), 33)             # matches the pack's own history: 33 triggers in 59 sessions
        for i, (d, D) in enumerate(sessions):
            row = next((r for r in rows if r["date"] == d), None)
            if not row:
                continue
            k = row["entry_slot"]
            oh, ol = TM.opening_range(D, 3)
            f = o.features_at(D[:k + 1], k, oh, ol, sessions[:i])      # nothing after the trigger bar exists
            for name in o.FEATURES:
                if name != "risk_pct":
                    self.assertEqual(f[name], row[name], "%s on %s" % (name, d))


class Context(unittest.TestCase):
    def test_context_reads_only_bars_up_to_the_trigger(self):
        D = breakout_day("2026-01-06", [(101.3, 100.7, 101.1)])
        spy = flat_day("2026-01-06", 400.0)
        for i in range(4):
            spy[i] = bar("2026-01-06", i, 400.0, 401.0, 399.0, 400.0 + i)
        a = o.candidate(TM, "2026-01-06", D, [], context={"SPY": {"2026-01-06": spy}})
        spy2 = copy.deepcopy(spy)
        for i in range(a["entry_slot"] + 1, 78):
            spy2[i] = bar("2026-01-06", i, 1.0, 9.0, 0.5, 7.0)
        b = o.candidate(TM, "2026-01-06", D, [], context={"SPY": {"2026-01-06": spy2}})
        self.assertEqual(a["ctx_spy_ret_pct"], b["ctx_spy_ret_pct"])
        self.assertAlmostEqual(a["ctx_spy_ret_pct"], 0.75, places=9)

    def test_missing_context_day_is_none_not_a_guess(self):
        D = breakout_day("2026-01-06", [(101.3, 100.7, 101.1)])
        self.assertIsNone(o.candidate(TM, "2026-01-06", D, [], context={"SPY": {}})["ctx_spy_ret_pct"])


class Split(unittest.TestCase):
    def rows(self, n):
        return [{"date": "2026-01-%02d" % (i + 1), "entry": 100.0, "risk": 1.0, "R": 1.0, "label": 1} for i in range(n)]

    def test_segments_are_chronological_disjoint_and_embargoed(self):
        seg = o.chrono_split(self.rows(30), embargo_days=2)
        last_end, days = "", []
        for name in ("train", "calibration", "validation", "test"):
            ds = sorted(r["date"] for r in seg[name])
            self.assertTrue(ds, name)
            self.assertGreater(ds[0], last_end)
            last_end = ds[-1]
            days += ds
        self.assertEqual(len(days), len(set(days)))
        self.assertLess(len(days), 30)                # the embargo really drops days

    def test_the_test_set_is_the_latest_data(self):
        seg = o.chrono_split(self.rows(40))
        self.assertGreater(min(r["date"] for r in seg["test"]), max(r["date"] for r in seg["train"]))


class Stats(unittest.TestCase):
    def test_costs_reduce_expectancy(self):
        rows = [{"entry": 100.0, "risk": 1.0, "R": 1.0, "label": 1}, {"entry": 100.0, "risk": 1.0, "R": -1.0, "label": 0}]
        self.assertLess(o.stats(rows, 5)["expectancy_R"], o.stats(rows, 0)["expectancy_R"])
        self.assertEqual(o.stats(rows, 0)["expectancy_R"], 0.0)
        self.assertEqual(o.stats([]), {"n": 0})


if __name__ == "__main__":
    unittest.main()
