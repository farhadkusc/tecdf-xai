import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class WulverSmokeTests(unittest.TestCase):
    def test_smoke_profile_beats_its_permutation_null(self):
        subprocess.check_call(
            [sys.executable, str(ROOT / "lab" / "njit" / "run_lab.py"), "--profile", "smoke", "--workers", "2"],
            cwd=ROOT,
        )
        report = json.loads((ROOT / "lab" / "njit" / "results" / "smoke.json").read_text())
        self.assertEqual(report["profile"], "smoke")
        self.assertFalse(report["ran_on_wulver"])
        self.assertEqual(report["n_jobs"], 2)
        null = report["permutation_null"]
        self.assertEqual(null["n_permutations"], 20)
        self.assertGreater(null["observed_graph_accuracy"], null["null_max"])
        self.assertEqual(null["fraction_null_ge_observed"], 0.0)


if __name__ == "__main__":
    unittest.main()
