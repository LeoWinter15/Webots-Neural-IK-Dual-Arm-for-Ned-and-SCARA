"""Checks for the PDF's exact error formulas and the ANN -> FK pipeline."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_ann import joint_errors, statistics, evaluate, FK
from robot_config import normalise_joints


class FixedModel:
    def __init__(self, outputs):
        self.outputs = outputs

    def predict_normalised(self, arm, xyz):
        return self.outputs.copy()


class Task4ATests(unittest.TestCase):
    def test_signed_wrap_crosses_pi_without_absolute_value(self):
        true = np.array([[np.pi - 0.1, -np.pi + 0.1, 0.2]])
        predicted = np.array([[-np.pi + 0.1, np.pi - 0.1, 0.5]])
        np.testing.assert_allclose(joint_errors("ned", true, predicted), [[-0.2, 0.2, -0.3]], atol=1e-14)

    def test_prismatic_error_is_squared_metres(self):
        true = np.array([[0, 0.1, -0.1], [0, 0.3, -0.15]])
        predicted = np.array([[0.2, 0.2, -0.08], [-0.2, 0.2, -0.18]])
        np.testing.assert_allclose(joint_errors("scara", true, predicted),
                                   [[-0.2, -0.1, 0.0004], [0.2, 0.1, 0.0009]], atol=1e-14)

    def test_population_standard_deviation(self):
        self.assertEqual(statistics([1, 3]), {"mean": 2.0, "std": 1.0})
        self.assertEqual(statistics([0]), {"mean": 0.0, "std": 0.0})
        with self.assertRaises(ValueError):
            statistics([np.nan])

    def test_perfect_predictions_have_zero_error_both_arms(self):
        for arm, q in (("ned", [[0.2, 0.3, 0.4], [-0.3, -0.2, 0.8]]),
                       ("scara", [[0.2, 0.3, -0.1], [-0.3, 0.2, -0.15]])):
            joints = np.asarray(q)
            labels = np.asarray([normalise_joints(arm, row) for row in joints])
            xyz = np.asarray([FK[arm](*row) for row in joints])
            details, header, rows = evaluate(FixedModel(labels), arm, (xyz, labels, joints))
            self.assertLess(details["position_error"]["mean"], 1e-14)
            for error in details["joint_errors"]:
                self.assertLess(abs(error["mean"]), 1e-14)
            self.assertEqual([r[0] for r in rows], [1, 2])
            self.assertEqual(len(header), len(rows[0]))

    def test_fk_uses_prediction_and_retains_nonpositive_elbow(self):
        arm = "scara"
        true = np.array([[0.1, 0.4, -0.1]])
        predicted = np.array([[0.1, -0.2, -0.05]])
        labels = np.array([normalise_joints(arm, true[0])])
        outputs = np.array([normalise_joints(arm, predicted[0])])
        xyz = np.array([FK[arm](*true[0])])
        details, _, _ = evaluate(FixedModel(outputs), arm, (xyz, labels, true))
        expected = np.linalg.norm(np.array(FK[arm](*predicted[0])) - xyz[0])
        self.assertAlmostEqual(details["position_error"]["mean"], expected)
        self.assertEqual(details["nonpositive_elbow_predictions"], 1)
        self.assertEqual(details["samples"], 1)
        self.assertEqual(details["excluded_samples"], 0)


if __name__ == "__main__":
    unittest.main()
