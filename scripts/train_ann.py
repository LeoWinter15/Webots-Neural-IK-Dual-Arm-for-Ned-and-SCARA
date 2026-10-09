"""Task 3: train small/medium/large shared Ned + SCARA ANNs using NumPy.

Only explicit training/validation CSVs are opened. Two learning rates are
compared using validation MSE; the test CSVs are reserved for Task 4.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from datetime import datetime, timezone

# Small dense matrices run faster without BLAS thread oversubscription.
# Respect any explicit settings provided by the caller.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np

from ann_model import Network, Adam, features, save_model, IKModel, INPUTS, OUTPUTS
from robot_config import CONFIG_VERSION, ROBOTS, denormalise_joints

ROOT = Path(__file__).resolve().parent.parent
ARMS = ("ned", "scara")
HEADER = ["arm", "x", "y", "z", "q1_norm", "q2_norm", "q3_norm"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_split(directory, arm, split, size, metadata):
    """Validate provenance before accepting Task 2 labels or positions."""
    if arm not in ARMS or split not in ("train", "validation"):
        raise ValueError("Training only accepts known arms and train/validation splits")
    path = directory / f"{arm}_{split}_{size}.csv"
    entry = metadata["files"][path.name]
    if sha256(path) != entry["sha256"]:
        raise ValueError(f"Dataset hash mismatch: {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != HEADER:
            raise ValueError(f"Unexpected header: {path}")
        rows = []
        for row in reader:
            if len(row) != 7 or row[0] != arm:
                raise ValueError(f"Malformed row or arm label: {path}")
            rows.append([float(v) for v in row[1:]])
    values = np.asarray(rows, dtype=float)
    expected = size * 4 // 5 if split == "train" else size - size * 4 // 5
    if values.shape != (expected, 6) or len(values) != entry["rows"]:
        raise ValueError(f"Incorrect sample count: {path}")
    if not np.isfinite(values).all() or (np.abs(values[:, 3:]) > 1).any():
        raise ValueError(f"Invalid values or labels outside [-1,1]: {path}")
    joints = np.asarray([denormalise_joints(arm, q) for q in values[:, 3:]])
    elbow = ROBOTS[arm]["positive_elbow_index"]
    if (joints[:, elbow] <= 0).any():
        raise ValueError(f"Dataset contains a nonpositive elbow: {path}")
    return values[:, :3], values[:, 3:]


def load_datasets(directory, sizes):
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if (metadata["config_version"] != CONFIG_VERSION or metadata["robots"] != ROBOTS
            or metadata["coordinate_frame"] != "ARM" or metadata["coordinate_units"] != "m"):
        raise ValueError("Dataset scaling/frame does not match robot_config.py")
    data = {size: {split: {arm: read_split(directory, arm, split, size, metadata)
                         for arm in ARMS} for split in ("train", "validation")}
            for size in sizes}
    for arm in ARMS:
        for split in ("train", "validation"):
            for small, large in zip(sizes[:-1], sizes[1:]):
                a, b = data[small][split][arm], data[large][split][arm]
                if any(not np.array_equal(x, y[:len(x)]) for x, y in zip(a, b)):
                    raise ValueError(f"Datasets are not nested: {arm}/{split}")
        train = data[sizes[-1]]["train"][arm][0]
        val = data[sizes[-1]]["validation"][arm][0]
        if set(map(tuple, train)).intersection(map(tuple, val)):
            raise ValueError(f"Training/validation position overlap: {arm}")
    return data, metadata


def combine(parts, mean, scale):
    return (np.concatenate([features(arm, parts[arm][0], mean, scale) for arm in ARMS]),
            np.concatenate([parts[arm][1] for arm in ARMS]))


def validation_metrics(network, x, y):
    predicted = network.predict(x)
    result = {"normalised_mse": float(np.mean((predicted - y) ** 2))}
    for index, arm in enumerate(ARMS):
        mask = x[:, 1 if arm == "ned" else 0] == 1
        error = predicted[mask] - y[mask]
        physical = np.asarray([denormalise_joints(arm, q) for q in predicted[mask]])
        elbow = ROBOTS[arm]["positive_elbow_index"]
        result[arm] = {
            "samples": int(mask.sum()),
            "normalised_mse": float(np.mean(error ** 2)),
            "normalised_mae_per_joint": np.mean(np.abs(error), axis=0).tolist(),
            "nonpositive_elbow_predictions": int((physical[:, elbow] <= 0).sum()),
        }
    return result


def train_candidate(x, y, xv, yv, args, learning_rate, history_path):
    model = Network(args.hidden, args.seed)
    optimiser = Adam(model, learning_rate)
    rng = np.random.default_rng(args.seed + 1)
    best_loss, significant_loss = float("inf"), float("inf")
    best_epoch, stale, best = 0, 0, None
    started = time.perf_counter()
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "train_mse", "validation_mse", "ned_validation_mse",
                         "scara_validation_mse"])
        for epoch in range(1, args.epochs + 1):
            order = rng.permutation(len(x))
            for start in range(0, len(x), args.batch_size):
                batch = order[start:start + args.batch_size]
                optimiser.step(model.gradients(x[batch], y[batch]))
            train_loss = float(np.mean((model.predict(x) - y) ** 2))
            prediction = model.predict(xv)
            mse = np.mean((prediction - yv) ** 2, axis=1)
            loss = float(mse.mean())
            if not np.isfinite(train_loss + loss):
                raise ValueError("Nonfinite training loss; inspect learning rate and data")
            writer.writerow([epoch, train_loss, loss, float(mse[xv[:, 1] == 1].mean()),
                             float(mse[xv[:, 0] == 1].mean())])
            # Always retain the absolute minimum, even if improvement is tiny.
            if loss < best_loss:
                best_loss, best_epoch, best = loss, epoch, model.snapshot()
            if loss < significant_loss - args.min_delta:
                significant_loss, stale = loss, 0
            else:
                stale += 1
            if epoch == 1 or epoch % 50 == 0:
                stream.flush()
                print(f"  lr={learning_rate:g} epoch={epoch} train={train_loss:.6g} "
                      f"validation={loss:.6g} best={best_loss:.6g}", flush=True)
            if stale >= args.patience:
                break
    model.restore(best)
    return model, {"learning_rate": learning_rate, "epochs_run": epoch,
                   "best_epoch": best_epoch,
                   "stop_reason": "patience" if stale >= args.patience else "epoch_limit",
                   "seconds": time.perf_counter() - started,
                   "validation": validation_metrics(model, xv, yv),
                   "history_file": history_path.name}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, default=ROOT / "datasets")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "task3")
    parser.add_argument("--sizes", type=int, nargs=3, default=[1000, 5000, 20000])
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.001, 0.003])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--min-delta", type=float, default=1e-7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (len(set(args.sizes)) != 3 or min(args.sizes) < 1000
            or min(args.hidden) < 1 or min(args.epochs, args.batch_size, args.patience) < 1
            or any(not np.isfinite(v) or v <= 0 for v in args.learning_rates)
            or not np.isfinite(args.min_delta) or args.min_delta < 0 or args.seed < 0):
        parser.error("Use three distinct sizes >=1000, positive training settings and nonnegative seed/min-delta")
    sizes = sorted(args.sizes)
    data, metadata = load_datasets(args.datasets, sizes)
    args.models.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)
    summary = {"task": 3, "created_utc": datetime.now(timezone.utc).isoformat(),
               "python": platform.python_version(), "numpy": np.__version__,
               "test_sets_used": False, "selection_metric": "combined validation normalised MSE",
               "models": []}
    for label, size in zip(("small", "medium", "large"), sizes):
        raw_train = np.concatenate([data[size]["train"][arm][0] for arm in ARMS])
        mean, scale = raw_train.mean(axis=0), raw_train.std(axis=0)
        scale[scale == 0] = 1
        x, y = combine(data[size]["train"], mean, scale)
        xv, yv = combine(data[size]["validation"], mean, scale)
        print(f"{label}: {len(x)} training + {len(xv)} validation rows (both arms)", flush=True)
        candidates, winner, best_score, winner_record = [], None, float("inf"), None
        for index, rate in enumerate(args.learning_rates):
            history = args.results / f"{label}_candidate_{index + 1}_history.csv"
            network, record = train_candidate(x, y, xv, yv, args, rate, history)
            candidates.append(record)
            if record["validation"]["normalised_mse"] < best_score:
                best_score = record["validation"]["normalised_mse"]
                winner, winner_record = network, record
        consumed = {name: entry for name, entry in metadata["files"].items()
                    if name in {f"{arm}_{split}_{size}.csv" for arm in ARMS
                                for split in ("train", "validation")}}
        model_metadata = {
            "format_version": 1, "config_version": CONFIG_VERSION, "robots": ROBOTS,
            "inputs": INPUTS, "outputs": OUTPUTS, "hidden": args.hidden,
            "activation": "tanh_all_layers", "output_policy": "tanh; no clipping or elbow projection",
            "coordinate_frame": "ARM", "coordinate_units": "m", "size_per_arm_before_split": size,
            "model_label": label, "seed": args.seed, "loss": "mean_squared_error_all_outputs",
            "optimiser": "Adam", "adam_beta1": 0.9, "adam_beta2": 0.999, "adam_epsilon": 1e-8,
            "batch_size": args.batch_size, "epochs_max": args.epochs, "patience": args.patience,
            "min_delta": args.min_delta, "train_samples": len(x), "validation_samples": len(xv),
            "scaler_fit": "combined training xyz only; flags unchanged",
            "selected": winner_record, "candidates": candidates, "datasets": consumed,
            "source_sha256": {name: sha256(Path(__file__).parent / name)
                              for name in ("ann_model.py", "train_ann.py", "robot_config.py")},
            "python": summary["python"], "numpy": np.__version__, "test_sets_used": False,
        }
        output = args.models / f"ik_shared_{size}.npz"
        save_model(output, winner, mean, scale, model_metadata)
        # Round trip the actual selected checkpoint for BOTH robot flags.
        loaded = IKModel(output)
        for arm in ARMS:
            xyz = data[size]["validation"][arm][0]
            np.testing.assert_allclose(loaded.predict_normalised(arm, xyz),
                                       winner.predict(features(arm, xyz, mean, scale)), rtol=0, atol=0)
        (args.models / f"ik_shared_{size}.json").write_text(
            json.dumps(model_metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        summary["models"].append({"label": label, "file": output.name, "sha256": sha256(output),
                                  "size": size, "selected": winner_record, "candidates": candidates})
        (args.results / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Saved {output.name}: validation MSE={best_score:.6g}", flush=True)


if __name__ == "__main__":
    main()
