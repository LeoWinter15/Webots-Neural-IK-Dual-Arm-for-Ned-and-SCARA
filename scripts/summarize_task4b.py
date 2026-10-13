"""Task 4.B: aggregate only actual Supervisor measurements, never substitute FK.

Final mean/std are populated only for groups with 100 settled measurements.
Incomplete groups expose explicitly labelled settled-subset statistics.
"""
import csv
import json
import math
import statistics

from task4b_common import load_plan, result_dir, sample_path, atomic_json


def distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def summarize(plan):
    groups, samples = [], []
    for arm, jobs in plan["jobs"].items():
        for job in jobs:
            errors, fk_errors, discrepancies = [], [], []
            failed, missing = 0, 0
            for sample in job["samples"]:
                path = sample_path(plan, arm, job["model"], sample["sample_index"])
                status, xyz, error, elapsed = "missing", None, None, None
                if path.exists():
                    event = json.loads(path.read_text(encoding="utf-8"))
                    for key, expected in {"plan_id": plan["plan_id"], "arm": arm,
                                           "model": job["model"], **sample}.items():
                        if event.get(key) != expected:
                            raise ValueError(f"Measurement provenance mismatch: {path} ({key})")
                    status, xyz, elapsed = event["status"], event["measured_xyz_m"], event["elapsed_simulation_s"]
                    if status not in ("settled", "settle_timeout", "simulation_ended"):
                        raise ValueError(f"Unknown status: {path}")
                    if xyz is not None and (len(xyz) != 3 or not all(math.isfinite(v) for v in xyz)):
                        raise ValueError(f"Invalid measured coordinates: {path}")
                    if status == "settled":
                        if xyz is None:
                            raise ValueError(f"Settled measurement missing TCP: {path}")
                        actual = event["actual_joints"] + event["actual_fixed_joints"]
                        targets = sample["predicted_joints"] + [0.0] * (3 if arm == "ned" else 1)
                        tolerances = plan["settling"][arm + "_tolerances"] + [plan["settling"]["fixed_tolerance_rad"]] * (len(targets) - 3)
                        if len(actual) != len(targets) or not all(
                                v is not None and math.isfinite(v) and abs(v - t) < tol
                                for v, t, tol in zip(actual, targets, tolerances)):
                            raise ValueError(f"Settled sample violates sensor tolerances: {path}")
                        error = distance(xyz, sample["target_xyz_m"])
                        errors.append(error)
                        fk_errors.append(distance(sample["fk_predicted_xyz_m"], sample["target_xyz_m"]))
                        discrepancies.append(distance(xyz, sample["fk_predicted_xyz_m"]))
                    else:
                        failed += 1
                else:
                    missing += 1
                samples.append([arm, job["model"], sample["sample_index"], status,
                                *sample["target_xyz_m"], *(xyz or [None] * 3), error, elapsed])
            complete = len(errors) == 100 and failed == 0 and missing == 0
            mean = statistics.mean(errors) if errors else None
            std = statistics.pstdev(errors) if errors else None
            groups.append({"arm": arm, "model": job["model"], "expected": 100,
                           "settled": len(errors), "failed": failed, "missing": missing,
                           "complete": complete, "mean_m": mean if complete else None,
                           "std_m": std if complete else None,
                           "settled_subset_mean_m": mean, "settled_subset_std_m": std,
                           "matched_fk_mean_m": statistics.mean(fk_errors) if errors else None,
                           "matched_fk_std_m": statistics.pstdev(fk_errors) if errors else None,
                           "webots_vs_fk_mean_m": statistics.mean(discrepancies) if errors else None})
    report = {"task": "4.B", "plan_id": plan["plan_id"], "measurement_source": "Webots Supervisor",
              "all_complete": all(g["complete"] for g in groups), "standard_deviation_ddof": 0,
              "failed_policy": "failure retained; full-group mean/std unavailable until all 100 settle",
              "results": groups}
    folder = result_dir(plan)
    atomic_json(folder / "summary.json", report)
    with (folder / "position_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(groups[0]))
        writer.writeheader()
        writer.writerows(groups)
    with (folder / "samples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["arm", "model", "sample_index", "status", "target_x_m", "target_y_m",
                         "target_z_m", "measured_x_m", "measured_y_m", "measured_z_m",
                         "settled_position_error_m", "elapsed_simulation_s"])
        writer.writerows(samples)
    for group in groups:
        print(f"{group['arm']}/{group['model']}: settled={group['settled']}/100 "
              f"failed={group['failed']} missing={group['missing']}")
    return report


if __name__ == "__main__":
    result = summarize(load_plan())
    print("TASK4B COMPLETE" if result["all_complete"] else "TASK4B INCOMPLETE: Webots measurements still required")
    raise SystemExit(0 if result["all_complete"] else 2)
