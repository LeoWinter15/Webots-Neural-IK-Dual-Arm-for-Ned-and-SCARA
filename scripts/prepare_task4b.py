"""Freeze Task 4.B commands from the FIRST 100 test rows per arm.

NumPy is required here, but not inside Webots. ANN prediction runs here;
Webots consumes these exact motor commands without fitting or clipping.
"""
import csv
import json
from pathlib import Path

import numpy as np

from ann_model import IKModel
from evaluate_ann import load_test, OUTPUT_POLICY, FK
from robot_config import ROBOTS, CONFIG_VERSION
from task4b_common import ROOT, PLAN_PATH, digest, plan_id, atomic_json, result_dir


def main():
    metadata = json.loads((ROOT / "datasets/metadata.json").read_text(encoding="utf-8"))
    if metadata["robots"] != ROBOTS or metadata["config_version"] != CONFIG_VERSION:
        raise ValueError("Dataset robot configuration mismatch")
    selection = json.loads((ROOT / "results/task3/summary.json").read_text(encoding="utf-8"))
    if [r["label"] for r in selection["models"]] != ["small", "medium", "large"]:
        raise ValueError("Three frozen Task 3 models are required")
    files = ["datasets/metadata.json", "results/task3/summary.json",
             "scripts/ann_model.py", "scripts/robot_config.py", "scripts/fk_ned.py",
             "scripts/fk_scara.py", "scripts/prepare_task4b.py", "scripts/task4b_common.py",
             "scripts/summarize_task4b.py", "scripts/evaluate_ann.py", "protos/TeachingNed.proto",
             "controllers/task4b_runner/task4b_runner.py",
             "worlds/task4b_ned.wbt", "worlds/task4b_scara.wbt"]
    plan = {"format_version": 1, "config_version": CONFIG_VERSION, "robots": ROBOTS,
            "output_policy": OUTPUT_POLICY, "sample_policy": "first 100 rows in original order",
            "measurement": "Webots Supervisor TCP.getPose(ARM); no analytical FK replacement",
            "settling": {"timeout_s": 60.0, "consecutive_steps": 20,
                         "ned_tolerances": [0.002, 0.002, 0.002],
                         "scara_tolerances": [0.002, 0.002, 0.0005],
                         "fixed_tolerance_rad": 0.002,
                         "angular_velocity": 0.7, "linear_velocity": 0.05},
            "home": {"ned": [0.0, 0.0, 0.4], "scara": [0.0, 0.4, -0.05]},
            "jobs": {}}
    for arm in ROBOTS:
        test = load_test(ROOT / "datasets", arm, metadata)
        files.append(f"datasets/{arm}_test.csv")
        # Actual reduced test CSVs, retaining the original headers/values/order.
        with (ROOT / f"datasets/{arm}_test.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))[:101]
        out = ROOT / "datasets" / f"{arm}_test_100_task4b.csv"
        with out.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)
        files.append(out.relative_to(ROOT).as_posix())
        jobs = []
        for record in selection["models"]:
            relative = "models/" + record["file"]
            if digest(ROOT / relative) != record["sha256"]:
                raise ValueError("Model differs from Task 3 selection")
            model = IKModel(ROOT / relative)
            if model.metadata["test_sets_used"] or model.metadata["output_policy"] != OUTPUT_POLICY:
                raise ValueError("Incompatible model policy")
            for name, expected in model.metadata["source_sha256"].items():
                if digest(ROOT / "scripts" / name) != expected:
                    raise ValueError(f"Training source changed: {name}")
            files.append(relative)
            xyz = test[0][:100]
            joints = model.predict_joints(arm, xyz)
            bounds = np.array(ROBOTS[arm]["normalisation_limits"])
            if not np.isfinite(joints).all() or ((joints < bounds[:, 0]) | (joints > bounds[:, 1])).any():
                raise ValueError("Invalid ANN targets; no clipping is permitted")
            jobs.append({"model": record["label"], "model_file": record["file"],
                         "samples": [{"sample_index": i + 1, "target_xyz_m": xyz[i].tolist(),
                                      "predicted_joints": joints[i].tolist(),
                                      "fk_predicted_xyz_m": list(FK[arm](*joints[i]))}
                                     for i in range(100)]})
        plan["jobs"][arm] = jobs
    plan["sha256"] = {name: digest(ROOT / name) for name in sorted(set(files))}
    plan["plan_id"] = plan_id(plan)
    atomic_json(PLAN_PATH, plan)
    print(f"Prepared 600 motor commands; plan {plan['plan_id']}")
    print(f"Results: {result_dir(plan)}")
    print("Open worlds/task4b_ned.wbt then worlds/task4b_scara.wbt in Webots R2023b.")


if __name__ == "__main__":
    main()
