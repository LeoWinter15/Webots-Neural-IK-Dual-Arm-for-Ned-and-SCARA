"""Recheck saved Task 3 checkpoints and selection records, without test data."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ann_model import IKModel
from train_ann import ARMS, ROOT, load_datasets, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, default=ROOT / "datasets")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "task3")
    args = parser.parse_args()
    summary = json.loads((args.results / "summary.json").read_text(encoding="utf-8"))
    records = summary["models"]
    if len(records) != 3 or [r["label"] for r in records] != ["small", "medium", "large"]:
        raise ValueError("Expected three completed small/medium/large models")
    data, _ = load_datasets(args.datasets, [record["size"] for record in records])
    checks = []
    for record in records:
        path = args.models / record["file"]
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Model hash mismatch: {path}")
        loaded = IKModel(path)
        metadata = loaded.metadata
        if metadata != json.loads(path.with_suffix(".json").read_text(encoding="utf-8")):
            raise ValueError("Embedded and human-readable model metadata differ")
        for name, digest in metadata["source_sha256"].items():
            if sha256(Path(__file__).parent / name) != digest:
                raise ValueError(f"Training source has changed since this model was saved: {name}")
        if metadata["selected"] != record["selected"] or metadata["test_sets_used"]:
            raise ValueError("Inconsistent model selection or test-set policy")
        for trial in record["candidates"]:
            with (args.results / trial["history_file"]).open(newline="", encoding="utf-8") as stream:
                history = list(csv.DictReader(stream))
            best = min(history, key=lambda row: float(row["validation_mse"]))
            if len(history) != trial["epochs_run"] or int(best["epoch"]) != trial["best_epoch"]:
                raise ValueError("Saved checkpoint epoch does not match training history")
            np.testing.assert_allclose(float(best["validation_mse"]),
                                       trial["validation"]["normalised_mse"], rtol=1e-12)
        chosen_mse = record["selected"]["validation"]["normalised_mse"]
        if chosen_mse != min(t["validation"]["normalised_mse"] for t in record["candidates"]):
            raise ValueError("Selected candidate is not the validation minimum")
        arm_mse = {}
        for arm in ARMS:
            xyz, target = data[record["size"]]["validation"][arm]
            predicted = loaded.predict_normalised(arm, xyz)
            if not np.isfinite(predicted).all() or (np.abs(predicted) > 1).any():
                raise ValueError("Invalid normalised predictions")
            arm_mse[arm] = float(np.mean((predicted - target) ** 2))
            np.testing.assert_allclose(arm_mse[arm],
                record["selected"]["validation"][arm]["normalised_mse"], rtol=1e-12)
        np.testing.assert_allclose(np.mean(list(arm_mse.values())), chosen_mse, rtol=1e-12)
        checks.append({"model": path.name, "passed": True, "validation_mse": chosen_mse,
                       "per_arm_validation_mse": arm_mse})
        print(f"PASS {path.name}: validation MSE={chosen_mse:.9g}")
    report = {"all_passed": True, "test_sets_used": False, "models": checks}
    (args.results / "verification.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
