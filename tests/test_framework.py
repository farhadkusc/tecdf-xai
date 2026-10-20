import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tecdf.framework import evaluate
from tecdf.nsl import load


class NslTests(unittest.TestCase):
    def test_file_shape(self):
        x, y, services = load()
        self.assertEqual(len(x), len(y))
        self.assertGreater(len(y), 20000)
        self.assertEqual(x.shape[1], 10)
        self.assertIn("http", set(services))

    def test_graph_beats_majority_and_survives_a_partial_rewire(self):
        report = evaluate(seed=7)
        graph = report["graph_service_neighborhood"]["accuracy"]
        attacked = report["graph_after_service_rewire"]["accuracy"]
        self.assertGreater(graph, report["majority_class_accuracy"])
        self.assertGreater(graph, 0.85)
        self.assertGreater(report["numeric_logistic"]["f1"], 0.8)
        self.assertLess(attacked, graph)
        self.assertGreater(attacked, 0.7)
        self.assertGreater(report["explanation_cosine_stability"], 0.8)


if __name__ == "__main__":
    unittest.main()
