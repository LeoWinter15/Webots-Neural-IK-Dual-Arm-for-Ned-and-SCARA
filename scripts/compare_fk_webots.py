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
import json
import hashlib
import shutil
from pathlib import Path
from datetime import datetime, timezone
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

    summaries = {}
    for arm, rows in errors.items():
        errs = [r[3] for r in rows]
        mean = sum(errs) / len(errs)
        std = math.sqrt(sum((e - mean) ** 2 for e in errs) / len(errs))
        summaries[arm] = {"count": len(rows), "mean_m": mean, "std_m": std,
                          "max_m": max(errs),
                          "joint_values_tested": [sorted(set(row[0][i] for row in rows)) for i in range(3)]}
        print(f"\n=== {arm.upper()} ({len(rows)} samples) ===")
        print(f"mean error = {mean * 1000:.3f} mm | "
              f"std = {std * 1000:.3f} mm | max = {max(errs) * 1000:.3f} mm")
        for q, measured, predicted, err in rows:
            print(f"  q={[round(a, 4) for a in q]}")
            print(f"    webots: {[round(v, 5) for v in measured]}")
            print(f"    FK    : {[round(v, 5) for v in predicted]}"
                  f"   err = {err * 1000:.3f} mm")

    # Keep the original measurement evidence with a hash and full-precision summary.
    digest = hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()
    destination = Path(SCRIPT_DIR).parent / "results" / "task1" / digest[:12]
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(csv_path, destination / "tcp_measurements.csv")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "measurements_sha256": digest,
        "coordinate_frame": "ARM", "position_unit": "m", "std_ddof": 0,
        "provenance_note": "Legacy CSV contains commanded joints; historical settled status and actual joints are not recorded.",
        "robots": summaries,
        "fk_source_sha256": {name: hashlib.sha256((Path(SCRIPT_DIR) / name).read_bytes()).hexdigest()
                             for name in ("fk_ned.py", "fk_scara.py")},
    }
    (destination / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved measurement snapshot and summary: {destination}")


if __name__ == "__main__":
    main()
