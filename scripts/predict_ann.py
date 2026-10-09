"""Task 3 inference: ARM-frame xyz (metres) -> named physical motor targets."""
import argparse
from pathlib import Path

from ann_model import IKModel
from robot_config import ROBOTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=tuple(ROBOTS))
    parser.add_argument("x", type=float)
    parser.add_argument("y", type=float)
    parser.add_argument("z", type=float)
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parent.parent
                        / "models" / "ik_shared_20000.npz")
    args = parser.parse_args()
    model = IKModel(args.model)
    xyz = [[args.x, args.y, args.z]]
    normalised = model.predict_normalised(args.arm, xyz)[0]
    joints = model.predict_joints(args.arm, xyz)[0]
    print("normalised:", ", ".join(f"{v:.8f}" for v in normalised))
    config = ROBOTS[args.arm]
    for name, value, unit in zip(config["joint_names"], joints, config["units"]):
        print(f"{name}: {value:.8f} {unit}")
    if joints[config["positive_elbow_index"]] <= 0:
        print("WARNING: predicted elbow is outside the positive training branch; prediction was not clipped.")


if __name__ == "__main__":
    main()
