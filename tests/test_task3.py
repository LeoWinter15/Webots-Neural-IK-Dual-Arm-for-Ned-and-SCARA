"""Numerical/contract checks; never read Task 4 test datasets."""
import json
import csv
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ann_model import Adam, Network, IKModel, features, save_model, INPUTS, OUTPUTS
from robot_config import ROBOTS, CONFIG_VERSION, normalise_joints
from train_ann import load_datasets, read_split, train_candidate


class Task3Tests(unittest.TestCase):
    def test_backprop_matches_finite_differences(self):
        rng = np.random.default_rng(8)
        model = Network((4, 3), 3)
        x, y = rng.normal(size=(5, 5)), rng.uniform(-0.8, 0.8, (5, 3))
        analytic = model.gradients(x, y)
        epsilon = 1e-6
        for parameter, gradient in zip(model.parameters(), analytic):
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                parameter[index] = original + epsilon
                plus = np.mean((model.predict(x) - y) ** 2)
                parameter[index] = original - epsilon
                minus = np.mean((model.predict(x) - y) ** 2)
                parameter[index] = original
                self.assertAlmostEqual(gradient[index], (plus - minus) / (2 * epsilon), places=8)

    def test_adam_first_update_and_learning(self):
        model = Network((8,), 4)
        before = model.snapshot()
        gradients = [np.ones_like(p) * 0.25 for p in model.parameters()]
        Adam(model, 0.01).step(gradients)
        for initial, value in zip(before, model.parameters()):
            np.testing.assert_allclose(value, initial - 0.01 * 0.25 / (0.25 + 1e-8))
        rng = np.random.default_rng(12)
        x = rng.normal(size=(32, 5))
        y = np.tanh(x[:, :3] * 0.3)
        initial_loss = np.mean((model.predict(x) - y) ** 2)
        optimiser = Adam(model, 0.01)
        for _ in range(150):
            optimiser.step(model.gradients(x, y))
        self.assertLess(np.mean((model.predict(x) - y) ** 2), initial_loss / 20)

    def test_roundtrip_units_and_configuration(self):
        metadata = {"format_version": 1, "config_version": CONFIG_VERSION, "robots": ROBOTS,
                    "inputs": INPUTS, "outputs": OUTPUTS, "hidden": [4],
                    "activation": "tanh_all_layers"}
        model = Network((4,), 2)
        mean, scale = np.array([0.1, 0.2, 0.3]), np.array([0.2, 0.3, 0.4])
        xyz = np.array([[0.2, 0.15, 0.3], [-0.2, 0.2, 0.4]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            save_model(path, model, mean, scale, metadata)
            loaded = IKModel(path)
            for arm in ROBOTS:
                expected = model.predict(features(arm, xyz, mean, scale))
                np.testing.assert_array_equal(loaded.predict_normalised(arm, xyz), expected)
                physical = loaded.predict_joints(arm, xyz)
                np.testing.assert_allclose([normalise_joints(arm, row) for row in physical], expected)
            metadata["config_version"] = "old_scaling"
            save_model(path, model, mean, scale, metadata)
            with self.assertRaises(ValueError):
                IKModel(path)

    def test_flags_bounds_and_invalid_inputs(self):
        xyz = np.array([[0.1, 0.2, 0.3]])
        for arm, flags in (("ned", [0, 1]), ("scara", [1, 0])):
            x = features(arm, xyz, np.zeros(3), np.ones(3))
            np.testing.assert_array_equal(x[0, :2], flags)
            self.assertTrue((np.abs(Network().predict(x * 1e6)) <= 1).all())
        with self.assertRaises(ValueError):
            features("unknown", xyz, np.zeros(3), np.ones(3))
        with self.assertRaises(ValueError):
            features("ned", [[np.nan, 0, 0]], np.zeros(3), np.ones(3))
        with self.assertRaises(ValueError):
            features("ned", [1, 2, 3], np.zeros(3), np.ones(3))

    def test_test_split_is_rejected_before_file_access(self):
        with self.assertRaises(ValueError):
            read_split(Path("does_not_exist"), "ned", "test", 1000, {})

    def test_training_restores_lowest_validation_checkpoint(self):
        rng = np.random.default_rng(5)
        xyz = rng.uniform(-0.3, 0.3, (16, 3))
        x = np.concatenate([features(arm, xyz, np.zeros(3), np.ones(3)) for arm in ROBOTS])
        y = np.tanh(x[:, 2:])
        args = SimpleNamespace(hidden=[8], seed=12, epochs=20, batch_size=8,
                               min_delta=100.0, patience=3)
        with tempfile.TemporaryDirectory() as directory:
            history = Path(directory) / "history.csv"
            # A deliberately conflicting validation target exercises overfitting
            # and restoring an earlier checkpoint instead of the last epoch.
            model, record = train_candidate(x, y, x, -y, args, 0.03, history)
            with history.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            best = min(rows, key=lambda row: float(row["validation_mse"]))
            self.assertEqual(record["stop_reason"], "patience")
            self.assertEqual(record["epochs_run"], 4)
            self.assertEqual(record["best_epoch"], int(best["epoch"]))
            self.assertAlmostEqual(np.mean((model.predict(x) + y) ** 2),
                                   float(best["validation_mse"]), places=14)

    def test_current_training_data_provenance_and_nesting(self):
        directory = Path(__file__).resolve().parents[1] / "datasets"
        data, _ = load_datasets(directory, [1000, 5000, 20000])
        self.assertEqual(data[20000]["train"]["ned"][0].shape, (16000, 3))
        self.assertEqual(data[20000]["validation"]["scara"][0].shape, (4000, 3))
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        metadata["files"]["ned_train_1000.csv"]["sha256"] = "incorrect"
        with self.assertRaises(ValueError):
            read_split(directory, "ned", "train", 1000, metadata)


if __name__ == "__main__":
    unittest.main()
