import json
import os
import sys
import tempfile
import unittest
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
warnings.filterwarnings("ignore")
import journal as jn  # noqa: E402
import meta_models as mm  # noqa: E402
import model_store as ms  # noqa: E402


class Store(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = mm.synthetic(n_days=500, edge=2.0, seed=3)
        cls.dep = ms.fit_deployable(cls.rows)

    def test_round_trip_scores_identically_and_links_the_ledger(self):
        d = tempfile.mkdtemp()
        ledger = os.path.join(d, "ledger.jsonl")
        jn.append_jsonl(ledger, {"id": "EXP-001", "kind": "walk-forward", "source": "SYNTHETIC EDGE", "result": "PASS: x"})
        jn.append_jsonl(ledger, {"id": "EXP-002", "kind": "walk-forward", "source": "4 symbols from bars", "result": "FAIL"})
        meta = ms.save(self.dep, d, ledger)
        self.assertEqual((meta["ledger_ref"], meta["ledger_result"]), ("EXP-002", "FAIL"))   # synthetic entries are skipped
        dep2, meta2 = ms.load(d)
        row = self.rows[-1]
        self.assertAlmostEqual(self.dep.score(row), dep2.score(row), places=9)
        self.assertEqual(meta2["threshold"], self.dep.threshold)

    def test_tampered_pickle_is_refused(self):
        d = tempfile.mkdtemp()
        ms.save(self.dep, d)
        with open(os.path.join(d, "model.pkl"), "ab") as fh:
            fh.write(b"x")
        with self.assertRaises(ms.ModelError):
            ms.load(d)

    def test_no_ledger_means_no_link(self):
        d = tempfile.mkdtemp()
        self.assertIsNone(ms.save(self.dep, d, os.path.join(d, "missing.jsonl"))["ledger_ref"])

    def test_probability_is_a_probability_and_missing_features_are_tolerated(self):
        row = dict(self.rows[0])
        row["rel_volume"] = None
        p = self.dep.score(row)
        self.assertTrue(0.0 <= p <= 1.0)

    def test_test_segment_is_not_used(self):
        seg = __import__("orb_dataset").chrono_split(self.rows)
        self.assertGreater(len(seg["test"]), 0)
        self.assertLess(self.dep.meta["trained_through"], min(r["date"] for r in seg["test"]))

    def test_too_little_data_refuses(self):
        with self.assertRaises(ms.ModelError):
            ms.fit_deployable(mm.synthetic(n_days=15, edge=0.0, seed=1))


if __name__ == "__main__":
    unittest.main()
