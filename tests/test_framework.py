import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tecdf.framework import evaluate, explanation_distance, make_corpus, stability


class FrameworkTests(unittest.TestCase):
    def test_stability_bounds(self):
        import numpy as np
        a = np.array([0.5, 0.5])
        self.assertAlmostEqual(stability(a, a, 0.2), 1.0)
        far = np.array([1.0, 0.0])
        self.assertLess(stability(a, far, 0.01), 1.0)
        self.assertGreaterEqual(explanation_distance(a, far), 0.0)

    def test_corpus_is_balanced(self):
        graphs = make_corpus(n_graphs=20, seed=1)
        self.assertEqual(sum(g.y for g in graphs), 10)

    def test_evaluation_runs(self):
        ev = evaluate(seed=4)
        self.assertEqual(ev.n, 40)
        self.assertGreater(ev.accuracy, 0.5)
        self.assertGreater(ev.mean_stability, 0.0)
        self.assertLessEqual(ev.mean_sparsity_top3, 1.0)


if __name__ == "__main__":
    unittest.main()
