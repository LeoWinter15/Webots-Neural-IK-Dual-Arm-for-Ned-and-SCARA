"""Task 4.A: frozen ANN predictions -> Task 1 FK -> held-out error statistics.

PDF page 12 specifies SIGNED wrapped angular differences (radians) and
SQUARED prismatic errors (square metres). All standard deviations use ddof=0.
No training, model selection, clipping, IK fallback or sample removal occurs.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np

from ann_model import IKModel
from robot_config import ROBOTS, CONFIG_VERSION, denormalise_joints
from fk_ned import fk as fk_ned
from fk_scara import fk as fk_scara
from train_ann import ROOT, ARMS, HEADER, sha256, load_datasets

FK = {"ned": fk_ned, "scara": fk_scara}
OUTPUT_POLICY = "tanh; no clipping or elbow projection"


def load_test(directory, arm, metadata):
    path = directory / f"{arm}_test.csv"
    entry = metadata["files"][path.name]
    if sha256(path) != entry["sha256"]:
        raise ValueError(f"Test file hash mismatch: {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != HEADER:
            raise ValueError(f"Unexpected test header: {path}")
        rows = []
        for row in reader:
            if len(row) != 7 or row[0] != arm:
                raise ValueError(f"Malformed test row: {path}")
            rows.append([float(v) for v in row[1:]])
    values = np.asarray(rows, dtype=float)
    count = metadata["test_size_per_arm"]
    if count < 500 or values.shape != (count, 6) or count != entry["rows"]:
        raise ValueError(f"Test set must match metadata and contain >=500 samples: {path}")
    if not np.isfinite(values).all() or (np.abs(values[:, 3:]) > 1).any():
        raise ValueError(f"Nonfinite data or normalised labels outside [-1,1]: {path}")
    xyz, normalised = values[:, :3], values[:, 3:]
    joints = np.asarray([denormalise_joints(arm, row) for row in normalised])
    if (joints[:, ROBOTS[arm]["positive_elbow_index"]] <= 0).any():
        raise ValueError(f"Nonpositive reference elbow in {path}")
    if len(set(map(tuple, xyz))) != count or len(set(map(tuple, joints))) != count:
        raise ValueError(f"Duplicate test positions or joint values in {path}")
    reconstructed = np.asarray([FK[arm](*row) for row in joints])
    if not np.allclose(reconstructed, xyz, rtol=0, atol=1e-10):
        raise ValueError(f"Test labels/positions disagree with current FK: {path}")
    return xyz, normalised, joints


def joint_errors(arm, true_joints, predicted_joints):
    """Exactly the two PDF formulas; do not take abs() of angular errors."""
    delta = np.asarray(true_joints) - np.asarray(predicted_joints)
    if delta.ndim != 2 or delta.shape[1] != 3 or not np.isfinite(delta).all():
        raise ValueError("Expected finite N x 3 physical joint differences")
    error = np.empty_like(delta, dtype=float)
    for i, unit in enumerate(ROBOTS[arm]["units"]):
        error[:, i] = (np.arctan2(np.sin(delta[:, i]), np.cos(delta[:, i]))
                       if unit == "rad" else delta[:, i] ** 2)
    return error


def statistics(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Cannot summarise empty or nonfinite errors")
    return {"mean": float(values.mean()), "std": float(values.std(ddof=0))}


def evaluate(model, arm, test):
    xyz, true_normalised, true_joints = test
    predicted_normalised = model.predict_normalised(arm, xyz)
    if predicted_normalised.shape != true_normalised.shape or not np.isfinite(predicted_normalised).all():
        raise ValueError("Model returned invalid predictions")
    if (np.abs(predicted_normalised) > 1).any():
        raise ValueError("Model violated its bounded-output contract")
    predicted_joints = np.asarray([denormalise_joints(arm, row) for row in predicted_normalised])
    predicted_xyz = np.asarray([FK[arm](*row) for row in predicted_joints])
    position_error = np.linalg.norm(predicted_xyz - xyz, axis=1)
    errors = joint_errors(arm, true_joints, predicted_joints)
    config = ROBOTS[arm]
    elbow_bad = predicted_joints[:, config["positive_elbow_index"]] <= 0
    bounds = np.asarray(config["normalisation_limits"])
    limit_bad = ((predicted_joints < bounds[:, 0]) | (predicted_joints > bounds[:, 1])).any(axis=1)
    details = {"samples": len(xyz), "position_error": {"unit": "m", **statistics(position_error)},
               "joint_errors": [], "nonpositive_elbow_predictions": int(elbow_bad.sum()),
               "physical_limit_violations": int(limit_bad.sum()), "excluded_samples": 0}
    for i, (name, unit) in enumerate(zip(config["joint_names"], config["units"])):
        details["joint_errors"].append({
            "output_index": i + 1, "joint": name,
            "metric": "signed_wrapped_angular_difference" if unit == "rad" else "squared_prismatic_error",
            "unit": "rad" if unit == "rad" else "m^2", **statistics(errors[:, i])})
    header = (["sample_index"] + [f"target_{a}_m" for a in "xyz"]
              + [f"predicted_{a}_m" for a in "xyz"]
              + [f"true_q{i}_norm" for i in (1, 2, 3)]
              + [f"predicted_q{i}_norm" for i in (1, 2, 3)]
              + [f"true_q{i+1}_{unit}" for i, unit in enumerate(config["units"])]
              + [f"predicted_q{i+1}_{unit}" for i, unit in enumerate(config["units"])]
              + ["position_error_m"]
              + [f"q{i+1}_error_{'rad' if unit == 'rad' else 'm2'}"
                 for i, unit in enumerate(config["units"])]
              + ["nonpositive_elbow", "physical_limit_violation"])
    rows = [[i + 1, *xyz[i], *predicted_xyz[i], *true_normalised[i],
             *predicted_normalised[i], *true_joints[i], *predicted_joints[i],
             position_error[i], *errors[i], int(elbow_bad[i]), int(limit_bad[i])]
            for i in range(len(xyz))]
    return details, header, rows


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def run(datasets, models, training_results, output):
    selection = json.loads((training_results / "summary.json").read_text(encoding="utf-8"))
    records = selection["models"]
    if len(records) != 3 or [r["label"] for r in records] != ["small", "medium", "large"]:
        raise ValueError("Expected three frozen Task 3 models")
    sizes = [r["size"] for r in records]
    data, metadata = load_datasets(datasets, sizes)
    for name in ("fk_ned.py", "fk_scara.py", "robot_config.py"):
        if sha256(Path(__file__).parent / name) != metadata["source_sha256"][name]:
            raise ValueError(f"FK/configuration changed since dataset generation: {name}")
    tests = {arm: load_test(datasets, arm, metadata) for arm in ARMS}
    for arm in ARMS:
        for split in ("train", "validation"):
            for index, test_values in ((0, tests[arm][0]), (1, tests[arm][1])):
                if set(map(tuple, test_values)).intersection(map(tuple, data[max(sizes)][split][arm][index])):
                    raise ValueError(f"Test overlap with {arm} {split} data")
    report = {"task": "4.A", "created_utc": datetime.now(timezone.utc).isoformat(),
              "python": platform.python_version(), "numpy": np.__version__,
              "coordinate_frame": "ARM", "config_version": CONFIG_VERSION,
              "standard_deviation_ddof": 0, "output_policy": OUTPUT_POLICY,
              "test_order": "original CSV order; sample_index is 1-based",
              "test_files": {f"{a}_test.csv": metadata["files"][f"{a}_test.csv"] for a in ARMS},
              "dataset_metadata_sha256": sha256(datasets / "metadata.json"),
              "training_summary_sha256": sha256(training_results / "summary.json"),
              "source_sha256": {n: sha256(Path(__file__).parent / n) for n in
                                ("evaluate_ann.py", "ann_model.py", "robot_config.py", "fk_ned.py",
                                 "fk_scara.py", "train_ann.py")}, "results": []}
    pending, position_rows, joint_rows = [], [], []
    for record in records:
        path = models / record["file"]
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Model differs from frozen Task 3 selection: {path}")
        model = IKModel(path)
        for name, digest in model.metadata["source_sha256"].items():
            if sha256(Path(__file__).parent / name) != digest:
                raise ValueError(f"Model source changed since training: {name}")
        if (model.metadata["output_policy"] != OUTPUT_POLICY or model.metadata["test_sets_used"]
                or model.metadata["size_per_arm_before_split"] != record["size"]
                or model.metadata["model_label"] != record["label"]
                or model.metadata["selected"] != record["selected"]):
            raise ValueError("Model metadata is inconsistent with Task 3 selection")
        for name, entry in model.metadata["datasets"].items():
            if metadata["files"].get(name) != entry:
                raise ValueError(f"Model training data differs from current dataset: {name}")
        for arm in ARMS:
            details, header, rows = evaluate(model, arm, tests[arm])
            filename = f"{record['label']}_{arm}_samples.csv"
            details.update(model=record["label"], arm=arm, size_per_arm=record["size"],
                           model_file=path.name, model_sha256=record["sha256"], samples_file=filename)
            report["results"].append(details)
            pending.append((filename, header, rows))
            p = details["position_error"]
            position_rows.append([record["label"], arm, details["samples"], p["mean"], p["std"],
                                  p["mean"] * 1000, p["std"] * 1000,
                                  details["nonpositive_elbow_predictions"], details["excluded_samples"]])
            for joint in details["joint_errors"]:
                joint_rows.append([record["label"], arm, details["samples"], joint["output_index"],
                                   joint["joint"], joint["metric"], joint["unit"], joint["mean"], joint["std"]])
    # Validate all inputs/models before writing any evaluation files.
    output.mkdir(parents=True, exist_ok=True)
    for filename, header, rows in pending:
        write_csv(output / filename, header, rows)
    write_csv(output / "position_summary.csv",
              ["model", "arm", "samples", "mean_m", "std_m", "mean_mm", "std_mm",
               "nonpositive_elbow_predictions", "excluded_samples"], position_rows)
    write_csv(output / "joint_summary.csv",
              ["model", "arm", "samples", "output_index", "joint", "metric", "unit", "mean", "std"], joint_rows)
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for row in position_rows:
        print(f"{row[0]:6s} {row[1]:5s} n={row[2]} position={row[5]:.6f} +/- {row[6]:.6f} mm "
              f"(mean +/- population std), nonpositive elbow={row[7]}")
    print(f"Saved 6 position summaries, 18 joint summaries and 6 sample files to {output}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, default=ROOT / "datasets")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--training-results", type=Path, default=ROOT / "results" / "task3")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "task4a")
    args = parser.parse_args()
    run(args.datasets, args.models, args.training_results, args.output)


if __name__ == "__main__":
    main()
