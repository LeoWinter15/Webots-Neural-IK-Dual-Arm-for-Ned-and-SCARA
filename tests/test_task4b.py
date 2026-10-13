"""Mocked control and aggregation tests, not Webots evaluation evidence."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import summarize_task4b as summary
from task4b_common import plan_id, atomic_json

spec = importlib.util.spec_from_file_location("task4b_runner", ROOT / "controllers/task4b_runner/task4b_runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Robot:
    def __init__(self, values):
        self.values = values
        self.index = -1

    def getTime(self):
        return (self.index + 1) * 0.1

    def step(self, timestep):
        self.index += 1
        return 0 if self.index < len(self.values) else -1

    def getValue(self):
        return self.values[self.index]


def fixture():
    jobs = {}
    for arm in ("ned", "scara"):
        jobs[arm] = [{"model": model, "samples": [
            {"sample_index": i, "target_xyz_m": [0, 0, 0],
             "predicted_joints": [0.1, 0.2, 0.0], "fk_predicted_xyz_m": [0, 0, 0]}
            for i in range(1, 101)]} for model in ("small", "medium", "large")]
    return {"plan_id": "synthetic_test_only", "jobs": jobs,
            "settling": {"ned_tolerances": [0.002] * 3,
                         "scara_tolerances": [0.002, 0.002, 0.0005], "fixed_tolerance_rad": 0.002}}


class Task4BTests(unittest.TestCase):
    def test_requires_consecutive_stable_steps(self):
        robot = Robot([0, 0, 0.5, float("nan"), 0, 0, 0])
        status, actual, elapsed = runner.wait_for_settle(robot, [robot], [0], [0.01], 100, 3, 5)
        self.assertEqual(status, "settled")
        self.assertAlmostEqual(elapsed, 0.7)

    def test_timeout_is_not_a_measurement_success(self):
        robot = Robot([1] * 20)
        status, _, _ = runner.wait_for_settle(robot, [robot], [0], [0.01], 100, 3, 0.5)
        self.assertEqual(status, "settle_timeout")

    def test_simulation_end_is_not_a_success(self):
        robot = Robot([0])
        status, _, _ = runner.wait_for_settle(robot, [robot], [0], [0.01], 100, 3, 5)
        self.assertEqual(status, "simulation_ended")

    def test_tcp_pose_translation_and_reference(self):
        arm = object()
        class TCP:
            def getPose(self, reference):
                self.reference = reference
                return [1, 0, 0, 0.1, 0, 1, 0, 0.2, 0, 0, 1, 0.3, 0, 0, 0, 1]
        tcp = TCP()
        self.assertEqual(runner.measured_tcp(tcp, arm), [0.1, 0.2, 0.3])
        self.assertIs(tcp.reference, arm)

    def test_plan_identity_changes_with_commands(self):
        a = {"joints": [0, 0, 0]}
        self.assertEqual(plan_id(a), plan_id({**a, "plan_id": "ignored"}))
        self.assertNotEqual(plan_id(a), plan_id({"joints": [0, 0, 1]}))

    def test_summary_requires_every_settled_sample(self):
        plan = fixture()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def sample_path(plan, arm, model, index):
                return root / arm / f"{model}_{index}.json"
            with patch.object(summary, "result_dir", return_value=root), patch.object(summary, "sample_path", side_effect=sample_path):
                empty = summary.summarize(plan)
                self.assertFalse(empty["all_complete"])
                self.assertTrue(all(g["mean_m"] is None and g["missing"] == 100 for g in empty["results"]))
                for arm, jobs in plan["jobs"].items():
                    for job in jobs:
                        for sample in job["samples"]:
                            event = {**sample, "plan_id": plan["plan_id"], "arm": arm,
                                     "model": job["model"], "status": "settled",
                                     "actual_joints": sample["predicted_joints"],
                                     "actual_fixed_joints": [0] * (3 if arm == "ned" else 1),
                                     "measured_xyz_m": [0.003, 0.004, 0], "elapsed_simulation_s": 1}
                            atomic_json(sample_path(plan, arm, job["model"], sample["sample_index"]), event)
                complete = summary.summarize(plan)
                self.assertTrue(complete["all_complete"])
                for group in complete["results"]:
                    self.assertAlmostEqual(group["mean_m"], 0.005)
                    self.assertAlmostEqual(group["std_m"], 0)
                failed_path = sample_path(plan, "ned", "small", 1)
                event = json.loads(failed_path.read_text())
                event["status"] = "settle_timeout"
                atomic_json(failed_path, event)
                failed = summary.summarize(plan)
                self.assertFalse(failed["all_complete"])
                group = failed["results"][0]
                self.assertEqual(group["failed"], 1)
                self.assertEqual(group["settled"], 99)
                self.assertIsNone(group["mean_m"])
                event["status"] = "settled"
                event["actual_joints"] = [9, 9, 9]
                atomic_json(failed_path, event)
                with self.assertRaises(ValueError):
                    summary.summarize(plan)


if __name__ == "__main__":
    unittest.main()
