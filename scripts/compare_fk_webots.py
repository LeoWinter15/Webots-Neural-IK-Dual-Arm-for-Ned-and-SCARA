"""compare_fk_webots.py - Task 1.C: verify FK against Webots measurements.

Reads tcp_measurements.csv (produced by the supervisor tcp_pose.py
after running the Task 1.A controllers) and compares every measured
TCP position against the analytical FK of fk_ned.py / fk_scara.py.
Reports the error per sample and the overall mean / max / std.

Usage:
    1. Run the Ned world, then the SCARA world (Task 1.A controllers),
       so that tcp_measurements.csv contains rows for both arms.
    2. python compare_fk_webots.py [path/to/tcp_measurements.csv]

Only the Python standard library is used.
"""

import math
import os
import sys

# Allow running from anywhere: make the sibling FK modules importable.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import fk_ned  # noqa: E402
import fk_scara  # noqa: E402

FK_FUNCS = {"ned": fk_ned.fk, "scara": fk_scara.fk}


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        SCRIPT_DIR, "..", "tcp_measurements.csv"
    )
    csv_path = os.path.normpath(csv_path)

    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found.")
        print("Run the Task 1.A controllers in Webots first.")
        sys.exit(1)

    errors = {}  # arm -> list of (q, measured, fk, error)

    with open(csv_path) as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            arm, q1, q2, q3, x, y, z = line.split(",")
            q = (float(q1), float(q2), float(q3))
            measured = (float(x), float(y), float(z))
            predicted = FK_FUNCS[arm](*q)
            err = math.dist(measured, predicted)
            errors.setdefault(arm, []).append((q, measured, predicted, err))

    if not errors:
        print("No measurements found in CSV.")
        sys.exit(1)

    for arm, rows in errors.items():
        errs = [r[3] for r in rows]
        mean = sum(errs) / len(errs)
        std = math.sqrt(sum((e - mean) ** 2 for e in errs) / len(errs))
        print(f"\n=== {arm.upper()} ({len(rows)} samples) ===")
        print(f"mean error = {mean * 1000:.3f} mm | "
              f"std = {std * 1000:.3f} mm | max = {max(errs) * 1000:.3f} mm")
        for q, measured, predicted, err in rows:
            print(f"  q={[round(a, 4) for a in q]}")
            print(f"    webots: {[round(v, 5) for v in measured]}")
            print(f"    FK    : {[round(v, 5) for v in predicted]}"
                  f"   err = {err * 1000:.3f} mm")


if __name__ == "__main__":
    main()
