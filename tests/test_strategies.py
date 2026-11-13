import os
import random
import sys
import tempfile
import unittest
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
warnings.filterwarnings("ignore")
import journal as jn  # noqa: E402
import model_store as ms  # noqa: E402
import orb_dataset as o  # noqa: E402
import paper_bot as pb  # noqa: E402
import strategies as ss  # noqa: E402

TM = o.load_pack()


def sessions(n=40, seed=5, drift=0.0):
    """Deterministic random-walk sessions of 78 bars on consecutive weekdays."""
    rng, out, px = random.Random(seed), [], 100.0
    day = __import__("datetime").date(2026, 1, 5)
    while len(out) < n:
        if day.weekday() < 5:
            d, D, p = day.isoformat(), [], px
            for i in range(78):
                o_ = p
                c = o_ * (1 + rng.gauss(drift, 0.0015))
                h, l = max(o_, c) * (1 + abs(rng.gauss(0, 0.0007))), min(o_, c) * (1 - abs(rng.gauss(0, 0.0007)))
                mins = 570 + 5 * i
                D.append({"date": d, "slot": i, "start": "%02d:%02d" % (mins // 60, mins % 60), "o": o_, "h": h, "l": l,
                          "c": c, "v": 1000 + rng.randrange(500)})
                p = c
            out.append((d, D))
            px = p
        day += __import__("datetime").timedelta(days=1)
    return out


class LeakyStrategy(ss.Strategy):
    """Deliberately cheats: peeks at the last bar of the day. check_contract must catch every kind of peek."""
    name = "leaky"

    def find_trigger(self, tm, D, prior):
        return next((i for i in range(3, len(D)) if D[i]["c"] > D[-1]["c"]), None)       # depends on the future

    def levels(self, tm, D, k):
        e = round(D[k]["c"], 2)
        return e, round(e * 0.99, 2), round(e * 1.01 + (D[-1]["c"] - D[k]["c"]) * 0, 2)

    def features(self, tm, D, k, prior, context):
        return {"future_ret": D[-1]["c"] - D[k]["c"]}                                    # depends on the future


class Equivalence(unittest.TestCase):
    def test_orb_on_the_plugin_matches_the_original_dataset_code_row_for_row(self):
        S, orb = sessions(60), ss.get("orb")
        n = 0
        for i, (d, D) in enumerate(S):
            old = o.candidate(TM, d, D, S[:i])
            new = orb.label_candidate(TM, d, D, S[:i])
            self.assertEqual(old is None, new is None, d)
            if old:
                n += 1
                for k, v in old.items():
                    self.assertEqual(v, new[k], "%s %s" % (d, k)) if not isinstance(v, float) else self.assertAlmostEqual(v, new[k], places=9)
        self.assertGreater(n, 10)

    def test_build_matches_orb_dataset_build(self):
        S = sessions(30, seed=9)
        a, b = o.build(TM, S, symbol="X"), ss.get("orb").build(TM, S, symbol="X")
        self.assertEqual([(r["date"], r["label"], r["entry"], r["stop"], r["target"]) for r in a],
                         [(r["date"], r["label"], r["entry"], r["stop"], r["target"]) for r in b])

    def test_params_flow_through(self):
        S = sessions(40, seed=2)
        wide = ss.get("orb", or_bars=6, target_r=2.0)
        rows = wide.build(TM, S, symbol="X")
        self.assertTrue(rows)
        for r in rows:
            self.assertAlmostEqual(r["target"] - r["entry"], 2.0 * r["risk"], delta=0.011)
            self.assertGreaterEqual(r["entry_slot"], 6)


class Contract(unittest.TestCase):
    def test_orb_honours_the_contract(self):
        problems, triggered = ss.check_contract(ss.get("orb"), TM, sessions(60, seed=11))
        self.assertEqual(problems, [])
        self.assertGreater(triggered, 10)

    def test_orb_with_other_parameters_too(self):
        for params in ({"or_bars": 6}, {"stop_rule": "or_low", "target_r": 1.5}):
            problems, _ = ss.check_contract(ss.get("orb", **params), TM, sessions(40, seed=4))
            self.assertEqual(problems, [], params)

    def test_a_cheating_strategy_is_caught(self):
        problems, triggered = ss.check_contract(LeakyStrategy(), TM, sessions(30, seed=3))
        self.assertGreater(triggered, 3)
        text = " ".join(problems)
        self.assertIn("trigger changes when future bars are removed", text)
        self.assertIn("future_ret", text)

    def test_leaky_features_alone_are_caught(self):
        class FeatureLeak(ss.ORB):
            name = "featleak"

            def features(self, tm, D, k, prior, context):
                f = super().features(tm, D, k, prior, context)
                f["tomorrow"] = D[-1]["c"]
                return f

        problems, _ = ss.check_contract(FeatureLeak(), TM, sessions(30, seed=3))
        self.assertTrue(any("tomorrow" in p for p in problems))

    def test_bad_levels_are_caught(self):
        class Upside(ss.ORB):
            name = "upside"

            def levels(self, tm, D, k):
                e = round(D[k]["c"], 2)
                return e, e + 1.0, e + 2.0                                     # stop above entry

        problems, _ = ss.check_contract(Upside(), TM, sessions(30, seed=3))
        self.assertTrue(any("stop < entry < target" in p for p in problems))


class Labels(unittest.TestCase):
    def bars(self, path):
        D = []
        for i, (h, l, c) in enumerate(path):
            D.append({"date": "2026-03-02", "slot": i, "start": "x", "o": c, "h": h, "l": l, "c": c, "v": 1})
        return D

    def test_stop_target_timeout_and_same_bar_is_stop(self):
        D = self.bars([(100, 100, 100), (100.5, 99.5, 100), (101.2, 99.9, 101), (101.2, 98.0, 100.5)])
        self.assertEqual(ss.triple_barrier(D, 0, 100.0, 99.0, 101.0, 3)[:2], (1, 2))        # target at bar 2
        D2 = self.bars([(100, 100, 100), (101.5, 98.5, 100)])
        self.assertEqual(ss.triple_barrier(D2, 0, 100.0, 99.0, 101.0, 1)[0], 0)             # both touched -> stop
        D3 = self.bars([(100, 100, 100), (100.4, 99.6, 100.2), (100.4, 99.6, 100.5)])
        label, xs, R = ss.triple_barrier(D3, 0, 100.0, 99.0, 101.0, 2)
        self.assertEqual((label, xs), (2, 2))
        self.assertAlmostEqual(R, 0.5)


class Wiring(unittest.TestCase):
    def test_registry_and_unknown_name(self):
        self.assertIn("orb", ss.names())
        with self.assertRaises(SystemExit):
            ss.get("nope")

    def test_bot_uses_the_strategy_and_needs_a_live_source_for_other_bar_sizes(self):
        cfg = pb.BotConfig(["A"], or_bars=4)
        self.assertEqual((cfg.strategy.name, cfg.strategy.or_bars), ("orb", 4))
        fast = pb.BotConfig(["A"], strategy=ss.get("orb", bar_seconds=30), out_dir=tempfile.mkdtemp())
        self.assertEqual((fast.strategy.bar_seconds, fast.risk.latest_entry_slot), (30, 600))   # the risk engine counts 30 s slots
        from test_paper_bot import TODAY, at
        import broker as bk
        with self.assertRaises(ValueError):                                    # the REST path is 5-minute only
            pb.tick(bk.FakeBroker(), TM, fast, None, {}, at(10, 20))

    def test_ledger_and_model_are_strategy_specific(self):
        d = tempfile.mkdtemp()
        ledger = os.path.join(d, "l.jsonl")
        jn.append_jsonl(ledger, {"id": "EXP-001", "kind": "walk-forward", "source": "4 symbols", "result": "PASS: x"})   # old entry = orb
        jn.append_jsonl(ledger, {"id": "EXP-002", "kind": "walk-forward", "strategy": "vwap", "source": "4 symbols", "result": "PASS: x"})
        self.assertEqual(ms.latest_real_walkforward(ledger, "orb")["id"], "EXP-001")
        self.assertEqual(ms.latest_real_walkforward(ledger, "vwap")["id"], "EXP-002")
        self.assertEqual(ms.latest_real_walkforward(ledger)["id"], "EXP-002")
        cfg = pb.BotConfig(["A"], ledger=ledger, out_dir=d)
        ok, why = pb.gate_ok(cfg, {"ledger_ref": "EXP-002", "trained_through": "2026-02-27", "strategy": "vwap"}, "2026-03-02")
        self.assertFalse(ok)                                                      # a vwap model cannot trade under the orb bot
        self.assertIn("strategy", why)
        ok, why = pb.gate_ok(cfg, {"ledger_ref": "EXP-001", "trained_through": "2026-02-27"}, "2026-03-02")
        self.assertTrue(ok, why)


if __name__ == "__main__":
    unittest.main()
