import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tecdf.dummy import load_flows, load_organizations, run_dummy

ROOT = Path(__file__).resolve().parents[1]


class DummyFlowTests(unittest.TestCase):
    def test_every_flow_is_tagged_with_a_registry_organization(self):
        orgs = {org["org_id"] for org in load_organizations()}
        counts = {}
        for org in load_organizations():
            counts[org["klass"]] = counts.get(org["klass"], 0) + 1
        self.assertEqual(counts["federal-department"], 15)
        self.assertEqual(counts["national-laboratory"], 17)
        self.assertGreaterEqual(counts["ibm"], 8)
        self.assertGreaterEqual(counts["research-center"], 40)
        _, y, services, klasses = load_flows()
        self.assertEqual(len(y), 20 * len(orgs))
        self.assertGreater(set(services), {"http", "ssh", "domain_u"})
        self.assertEqual(set(klasses), set(counts))

    def test_planted_service_signal_is_above_the_majority_class(self):
        report = run_dummy()
        self.assertEqual(report["kind"], "synthetic-fixture")
        overall = report["overall"]
        graph = overall["graph_service_neighborhood"]["accuracy"]
        self.assertGreater(graph, overall["majority_class_accuracy"])
        self.assertGreater(overall["numeric_logistic"]["average_precision"], 0.7)
        self.assertLess(
            overall["graph_after_service_rewire"]["accuracy"],
            overall["graph_service_neighborhood"]["accuracy"],
        )
        self.assertIn("federal-department", report["by_class"])
        self.assertIn("ibm", report["by_class"])
        self.assertIn("research-center", report["by_class"])
        self.assertIn("national-laboratory", report["by_class"])

    def test_grey_harbor_spoof_cuts_recall(self):
        scenario = run_dummy()["attack_scenario"]
        self.assertEqual(scenario["exercise_id"], "SYN-EXERCISE-GREY-HARBOR")
        self.assertEqual(scenario["spoofed_service"], "domain_u")
        self.assertGreater(scenario["n_spoofed"], 0)
        self.assertGreater(scenario["recall_before"], scenario["recall_after"])
        self.assertIn("FED-DOE", scenario["blue_cell"])
        self.assertNotIn("IBM-QUANTUM", scenario["blue_cell"])
        self.assertIn("IBM-QUANTUM", scenario["out_of_scope"])

    def test_public_evaluation_is_still_nsl_kdd(self):
        public = json.loads((ROOT / "benchmarks" / "results" / "evaluation.json").read_text())
        self.assertTrue(public["dataset"].startswith("NSL-KDD"))
        self.assertNotEqual(public.get("kind"), "synthetic-fixture")
        saved = json.loads((ROOT / "benchmarks" / "results" / "dummy_evaluation.json").read_text())
        self.assertEqual(saved["kind"], "synthetic-fixture")
        self.assertEqual(saved["n_flows"], 2120)
        self.assertEqual(saved["n_organizations"], 106)
        self.assertEqual(saved["attack_scenario"]["exercise_id"], "SYN-EXERCISE-GREY-HARBOR")


if __name__ == "__main__":
    unittest.main()
