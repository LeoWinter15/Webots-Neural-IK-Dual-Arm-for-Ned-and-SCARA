"""Task 2: create nested FK datasets for the Ned and SCARA arms.

Run from any directory: python scripts/generate_datasets.py --output datasets
Only the Python standard library and the existing Task 1.B FK scripts are used.
"""

import argparse
import csv
import random
import json
import hashlib
from datetime import datetime, timezone
from robot_config import ROBOTS as CONFIG, CONFIG_VERSION, normalise, denormalise, normalise_joints
from pathlib import Path

import fk_ned
import fk_scara


ROBOTS = {
    arm: (fk, CONFIG[arm]["sampling_limits"], CONFIG[arm]["joint_names"],
          CONFIG[arm]["positive_elbow_index"])
    for arm, fk in (("ned", fk_ned.fk), ("scara", fk_scara.fk))
}
HEADER = ("arm", "x", "y", "z", "q1_norm", "q2_norm", "q3_norm")


def sample_rows(arm, count, rng, used_joints, used_positions):
    """Generate distinct joint/position pairs using the Task 1.B FK."""
    fk, limits, _names, elbow_index = ROBOTS[arm]
    rows = []
    while len(rows) < count:
        joints = tuple(rng.uniform(low, high) for low, high in limits)
        if joints[elbow_index] <= 0.0 or joints in used_joints:
            continue
        position = fk(*joints)
        if position in used_positions:
            continue
        used_joints.add(joints)
        used_positions.add(position)
        scaled = normalise_joints(arm, joints)
        rows.append((arm, *position, *scaled))
    return rows


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADER)
        writer.writerows(rows)


def generate(output, sizes, test_size, seed):
    """Create nested training/validation files and a held-out test per arm.

    Generate the largest train and validation pools once, then take prefixes
    for smaller sizes. This guarantees nesting *within each split*: no row
    moves from training to validation when the requested size changes.
    """
    largest = sizes[-1]
    max_train = largest * 4 // 5
    max_validation = largest - max_train
    for arm_index, arm in enumerate(ROBOTS):
        rng = random.Random(seed + arm_index)
        used_joints, used_positions = set(), set()
        train = sample_rows(arm, max_train, rng, used_joints, used_positions)
        validation = sample_rows(
            arm, max_validation, rng, used_joints, used_positions
        )
        test = sample_rows(arm, test_size, rng, used_joints, used_positions)
        for size in sizes:
            train_count = size * 4 // 5
            save_csv(output / f"{arm}_train_{size}.csv", train[:train_count])
            save_csv(
                output / f"{arm}_validation_{size}.csv",
                validation[: size - train_count],
            )
        save_csv(output / f"{arm}_test.csv", test)
        print(f"{arm}: {len(sizes)} train, {len(sizes)} validation, 1 test files")
    paths = [output / f"{arm}_{split}_{size}.csv"
             for arm in ROBOTS for split in ("train", "validation") for size in sizes]
    paths += [output / f"{arm}_test.csv" for arm in ROBOTS]
    metadata = {
        "config_version": CONFIG_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "coordinate_frame": "ARM", "coordinate_units": "m",
        "robots": CONFIG, "sizes_before_split": list(sizes),
        "train_fraction": 0.8, "test_size_per_arm": test_size,
        "seeds": {arm: seed + i for i, arm in enumerate(ROBOTS)},
        "sampling": "independent uniform joints; elbow strictly positive",
        "nesting": "prefixes within separate train and validation pools",
        "test_policy": "disjoint held-out; frozen order; first 100 for Task 4.B",
        "files": {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                           "rows": sum(1 for _ in p.open(encoding="utf-8")) - 1}
                  for p in paths},
        "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                          for name in ("robot_config.py", "generate_datasets.py", "fk_ned.py", "fk_scara.py")},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved to {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "datasets")
    parser.add_argument("--sizes", type=int, nargs=3, default=(1000, 5000, 20000),
                        metavar="N", help="three total sizes before the 80/20 split")
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sizes = sorted(args.sizes)
    if sizes[0] < 1000 or len(set(sizes)) != 3 or any(n % 5 for n in sizes):
        parser.error("sizes must be three distinct multiples of five, all >= 1000")
    if args.test_size < 500:
        parser.error("test-size must be at least 500")
    generate(args.output, sizes, args.test_size, args.seed)


if __name__ == "__main__":
    main()
