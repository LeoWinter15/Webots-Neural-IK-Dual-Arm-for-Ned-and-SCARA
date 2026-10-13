"""Independently recompute Task 4.A CSV errors/statistics using stdlib math.

Does not call the evaluator's error/statistics functions or rerun training.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from fk_ned import fk as fk_ned
from fk_scara import fk as fk_scara
from robot_config import ROBOTS, denormalise_joints

ROOT = Path(__file__).resolve().parent.parent


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def close(actual, expected):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-11, abs_tol=1e-14):
        raise ValueError(f"Mismatch: {actual} != {expected}")


def check_stats(values, mean, std):
    close(statistics.mean(values), mean)
    close(statistics.pstdev(values), std)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "task4a")
    parser.add_argument("--datasets", type=Path, default=ROOT / "datasets")
    args = parser.parse_args()
    report = json.loads((args.results / "summary.json").read_text(encoding="utf-8"))
    positions = read_csv(args.results / "position_summary.csv")
    joints = read_csv(args.results / "joint_summary.csv")
    if len(positions) != 6 or len(joints) != 18 or len(report["results"]) != 6:
        raise ValueError("Expected 6 position and 18 joint statistic groups")
    seen, verified = set(), []
    for entry in report["results"]:
        arm, model = entry["arm"], entry["model"]
        key = (model, arm)
        if key in seen:
            raise ValueError("Duplicate model/arm result")
        seen.add(key)
        test_path = args.datasets / f"{arm}_test.csv"
        if hashlib.sha256(test_path.read_bytes()).hexdigest() != report["test_files"][test_path.name]["sha256"]:
            raise ValueError("Test input changed since evaluation")
        test = read_csv(test_path)
        rows = read_csv(args.results / entry["samples_file"])
        if len(rows) != len(test) or len(rows) != entry["samples"]:
            raise ValueError("Incomplete per-sample output")
        pos_errors, joint_values = [], [[], [], []]
        nonpositive, violations = 0, 0
        fk = fk_ned if arm == "ned" else fk_scara
        units = ROBOTS[arm]["units"]
        for index, (row, target) in enumerate(zip(rows, test), 1):
            if int(row["sample_index"]) != index:
                raise ValueError("Changed test order or sample index")
            true = denormalise_joints(arm, [float(target[f"q{i}_norm"]) for i in (1, 2, 3)])
            pred = denormalise_joints(arm, [float(row[f"predicted_q{i}_norm"]) for i in (1, 2, 3)])
            calculated_xyz = fk(*pred)
            for axis, value in zip("xyz", calculated_xyz):
                close(row[f"predicted_{axis}_m"], value)
                close(row[f"target_{axis}_m"], target[axis])
            pos = math.sqrt(sum((value - float(target[axis])) ** 2 for axis, value in zip("xyz", calculated_xyz)))
            close(row["position_error_m"], pos)
            pos_errors.append(pos)
            for i, unit in enumerate(units, 1):
                close(row[f"true_q{i}_{unit}"], true[i - 1])
                close(row[f"predicted_q{i}_{unit}"], pred[i - 1])
                close(row[f"true_q{i}_norm"], target[f"q{i}_norm"])
                delta = true[i - 1] - pred[i - 1]
                error = math.atan2(math.sin(delta), math.cos(delta)) if unit == "rad" else delta ** 2
                close(row[f"q{i}_error_{'rad' if unit == 'rad' else 'm2'}"], error)
                joint_values[i - 1].append(error)
            elbow_bad = int(pred[ROBOTS[arm]["positive_elbow_index"]] <= 0)
            limit_bad = int(any(not low <= value <= high for value, (low, high)
                                in zip(pred, ROBOTS[arm]["normalisation_limits"])))
            close(row["nonpositive_elbow"], elbow_bad)
            close(row["physical_limit_violation"], limit_bad)
            nonpositive += elbow_bad
            violations += limit_bad
        close(nonpositive, entry["nonpositive_elbow_predictions"])
        close(violations, entry["physical_limit_violations"])
        close(entry["excluded_samples"], 0)
        check_stats(pos_errors, entry["position_error"]["mean"], entry["position_error"]["std"])
        p_rows = [r for r in positions if (r["model"], r["arm"]) == key]
        if len(p_rows) != 1:
            raise ValueError("Missing/duplicate position summary")
        p = p_rows[0]
        close(p["samples"], len(rows))
        check_stats(pos_errors, p["mean_m"], p["std_m"])
        close(p["mean_mm"], float(p["mean_m"]) * 1000)
        close(p["std_mm"], float(p["std_m"]) * 1000)
        close(p["nonpositive_elbow_predictions"], nonpositive)
        close(p["excluded_samples"], 0)
        for index, values in enumerate(joint_values, 1):
            j_rows = [r for r in joints if (r["model"], r["arm"]) == key and int(r["output_index"]) == index]
            if len(j_rows) != 1:
                raise ValueError("Missing/duplicate joint summary")
            j, detail = j_rows[0], entry["joint_errors"][index - 1]
            for field in ("joint", "metric", "unit"):
                if j[field] != detail[field]:
                    raise ValueError("Joint metric metadata mismatch")
            close(j["samples"], len(rows))
            check_stats(values, j["mean"], j["std"])
            check_stats(values, detail["mean"], detail["std"])
        verified.append({"model": model, "arm": arm, "samples": len(rows), "passed": True})
        print(f"PASS {model}/{arm}: {len(rows)} samples, all four error statistics")
    if seen != {(m, a) for m in ("small", "medium", "large") for a in ROBOTS}:
        raise ValueError("Missing model/arm combination")
    (args.results / "verification.json").write_text(
        json.dumps({"all_passed": True, "method": "stdlib math + statistics, CSV recomputation",
                    "checks": verified}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
