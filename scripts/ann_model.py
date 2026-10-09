"""Shared five-input, three-output ANN and portable NumPy model format.

Every layer uses tanh, including the bounded output layer. No analytical IK
or output clipping is used. Robot-specific units/scaling come from Task 2.
"""
import json
from pathlib import Path

import numpy as np

from robot_config import CONFIG_VERSION, ROBOTS, denormalise_joints

INPUTS = ["is_scara", "is_ned", "x", "y", "z"]
OUTPUTS = ["q1_norm", "q2_norm", "q3_norm"]


class Network:
    """Dense tanh MLP; weights have shape (input width, output width)."""

    def __init__(self, hidden=(64, 64), seed=42):
        widths = [5, *hidden, 3]
        rng = np.random.default_rng(seed)
        # Xavier uniform initialisation keeps tanh preactivations moderate.
        self.weights = [rng.uniform(-np.sqrt(6 / (a + b)),
                                    np.sqrt(6 / (a + b)), (a, b))
                        for a, b in zip(widths[:-1], widths[1:])]
        self.biases = [np.zeros(b) for b in widths[1:]]

    def forward(self, x):
        activations = [x]
        for weight, bias in zip(self.weights, self.biases):
            activations.append(np.tanh(activations[-1] @ weight + bias))
        return activations

    def predict(self, x):
        return self.forward(x)[-1]

    def gradients(self, x, y):
        """Gradient of mean squared error over all samples AND outputs."""
        acts = self.forward(x)
        delta = 2 * (acts[-1] - y) / y.size
        dw, db = [None] * len(self.weights), [None] * len(self.biases)
        for i in reversed(range(len(self.weights))):
            delta = delta * (1 - acts[i + 1] ** 2)
            dw[i] = acts[i].T @ delta
            db[i] = delta.sum(axis=0)
            delta = delta @ self.weights[i].T
        return dw + db

    def parameters(self):
        return self.weights + self.biases

    def snapshot(self):
        return [value.copy() for value in self.parameters()]

    def restore(self, values):
        for parameter, value in zip(self.parameters(), values):
            parameter[:] = value


class Adam:
    """Adam with first/second moment bias correction, beta=(0.9, 0.999)."""

    def __init__(self, network, learning_rate):
        self.parameters = network.parameters()
        self.m = [np.zeros_like(p) for p in self.parameters]
        self.v = [np.zeros_like(p) for p in self.parameters]
        self.learning_rate = learning_rate
        self.step_count = 0

    def step(self, gradients):
        self.step_count += 1
        for p, m, v, g in zip(self.parameters, self.m, self.v, gradients):
            m *= 0.9
            m += 0.1 * g
            v *= 0.999
            v += 0.001 * g * g
            p -= self.learning_rate * (m / (1 - 0.9 ** self.step_count)) / (
                np.sqrt(v / (1 - 0.999 ** self.step_count)) + 1e-8)


def features(arm, xyz, mean, scale):
    if arm not in ROBOTS:
        raise ValueError(f"Unknown arm: {arm}")
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("Expected an N x 3 array of finite ARM-frame coordinates in metres")
    flags = np.tile([float(arm == "scara"), float(arm == "ned")], (len(xyz), 1))
    return np.column_stack((flags, (xyz - mean) / scale))


def save_model(path, network, mean, scale, metadata):
    """Store numeric arrays and JSON, without pickle or sklearn dependencies."""
    payload = {"mean": mean, "scale": scale,
               "metadata": np.asarray(json.dumps(metadata, allow_nan=False))}
    for i, (w, b) in enumerate(zip(network.weights, network.biases)):
        payload[f"weight_{i}"] = w
        payload[f"bias_{i}"] = b
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    temporary.replace(path)


class IKModel:
    """The same loaded model handles either arm; predictions are never clipped."""

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.metadata = json.loads(str(data["metadata"]))
            if (self.metadata.get("format_version") != 1
                    or self.metadata.get("config_version") != CONFIG_VERSION
                    or self.metadata.get("robots") != ROBOTS
                    or self.metadata.get("inputs") != INPUTS
                    or self.metadata.get("outputs") != OUTPUTS
                    or self.metadata.get("activation") != "tanh_all_layers"):
                raise ValueError("Model format or robot configuration does not match this project")
            self.network = Network(self.metadata["hidden"])
            for i, (w, b) in enumerate(zip(self.network.weights, self.network.biases)):
                saved_w, saved_b = data[f"weight_{i}"], data[f"bias_{i}"]
                if saved_w.shape != w.shape or saved_b.shape != b.shape:
                    raise ValueError("Model weight dimensions do not match its metadata")
                w[:] = saved_w
                b[:] = saved_b
            self.mean, self.scale = data["mean"].copy(), data["scale"].copy()
        if (self.mean.shape != (3,) or self.scale.shape != (3,)
                or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all()
                or (self.scale <= 0).any()
                or any(not np.isfinite(p).all() for p in self.network.parameters())):
            raise ValueError("Model contains invalid scaler or network parameters")

    def predict_normalised(self, arm, xyz):
        return self.network.predict(features(arm, xyz, self.mean, self.scale))

    def predict_joints(self, arm, xyz):
        normalised = self.predict_normalised(arm, xyz)
        return np.asarray([denormalise_joints(arm, row) for row in normalised]).reshape(-1, 3)
