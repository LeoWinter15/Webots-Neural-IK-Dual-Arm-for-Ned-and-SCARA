"""Check reduced tests and prepared commands against Task 4.A before Webots."""
import csv
import math

from task4b_common import ROOT, load_plan


def read_rows(path, dictionaries=False):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream) if dictionaries else csv.reader(stream))


def main():
    plan = load_plan()
    count, maximum_difference = 0, 0.0
    for arm, jobs in plan["jobs"].items():
        original = read_rows(ROOT / "datasets" / f"{arm}_test.csv")[:101]
        reduced = read_rows(ROOT / "datasets" / f"{arm}_test_100_task4b.csv")
        if reduced != original:
            raise ValueError("Reduced test is not exactly the original first 100 rows")
        for job in jobs:
            reference = read_rows(ROOT / "results/task4a" / f"{job['model']}_{arm}_samples.csv", True)[:100]
            if len(reference) != 100 or len(job["samples"]) != 100:
                raise ValueError("Expected 100 samples per job")
            for sample, row in zip(job["samples"], reference):
                if sample["sample_index"] != int(row["sample_index"]):
                    raise ValueError("Changed sample order")
                for axis, value in zip("xyz", sample["target_xyz_m"]):
                    if value != float(row[f"target_{axis}_m"]):
                        raise ValueError("Target differs from Task 4.A")
                for i, unit in enumerate(plan["robots"][arm]["units"], 1):
                    old = float(row[f"predicted_q{i}_{unit}"])
                    new = sample["predicted_joints"][i - 1]
                    if not math.isclose(old, new, rel_tol=0, abs_tol=1e-12):
                        raise ValueError("Command differs from frozen Task 4.A prediction")
                    maximum_difference = max(maximum_difference, abs(old - new))
                count += 1
    print(f"PASS: exact first 100 test rows for both arms; {count} commands match Task 4.A")
    print(f"Maximum floating-point difference: {maximum_difference:g}; plan={plan['plan_id']}")


if __name__ == "__main__":
    main()
