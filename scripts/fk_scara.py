"""fk_scara.py - Task 1.B: forward kinematics of the Epson SCARA (ScaraT6).

Pure Python script (no Webots). Computes the TCP position in the ARM
(top-level ScaraT6) frame from the three active joint values:

    q1 = base_arm_motor      (base rotation,   rad)
    q2 = arm_motor           (second rotation, rad)
    q4 = shaft_linear_motor  (vertical shaft displacement, METRES;
                              q4 = 0 shaft fully up, q4 = -0.2 fully down)

The shaft_rotation_motor (tool yaw) is ignored in this coursework.

The TCP is the end of the shaft where the manipulator would attach.
Formula from the coursework specification ("Robot specifications -
SCARA"):

    p = [0.060, 0, 0.220]
        + Rz(q1) * ( [0.310, 0.005, -0.004]
                     + Rz(q2) * [0.2545, -0.005, q4 - 0.034] )

with  Rz(q) = [ cos q, -sin q, 0; sin q, cos q, 0; 0, 0, 1 ].

All three vectors are given directly in the specification, so unlike
the Ned there is no derivation to do - the script implements the
formula exactly and the Task 1.C comparison against Webots verifies it.

Usage:
    python fk_scara.py                  # demo table
    python fk_scara.py q1 q2 q4         # e.g. python fk_scara.py 0.3 0.5 -0.1

Task 2 imports fk() to generate the training dataset.
"""

import math
import sys

# ------------------------------------------------------------------
# SCARA geometry constants (from the coursework specification)
# ------------------------------------------------------------------

# Offset of the top-level ScaraT6 frame relative to joint 1.
TOP_TO_JOINT1 = (0.060, 0.0, 0.220)

# Joint 1 -> joint 2 vector.
JOINT1_TO_JOINT2 = (0.310, 0.005, -0.004)

# Joint 2 -> TCP/handSlot vector (z depends on the shaft extension q4).
JOINT2_TO_TCP_Z_OFFSET = -0.034   # TCP sits 34 mm below joint 2 at q4 = 0
JOINT2_TO_TCP_XY = (0.2545, -0.005)

# Joint limits from the coursework specification (used for warnings).
JOINT_LIMITS = {
    "base_arm_motor": (-0.73, 0.73),
    "arm_motor": (-0.83, 0.83),
    "shaft_linear_motor": (-0.2, 0.0),
}


def _rz(q, vector):
    """Rotate a 3D vector about the z axis by angle q (rad)."""
    c, s = math.cos(q), math.sin(q)
    x, y, z = vector
    return (c * x - s * y, s * x + c * y, z)


def fk(q1, q2, q4):
    """Forward kinematics: (rad, rad, m) -> TCP (x, y, z) in ARM frame."""
    # Innermost term: joint 2 -> TCP, including the shaft extension.
    tcp_local = (
        JOINT2_TO_TCP_XY[0],
        JOINT2_TO_TCP_XY[1],
        q4 + JOINT2_TO_TCP_Z_OFFSET,
    )

    # Rotate by q2 (second arm link) and translate by joint 1 -> joint 2.
    v = _rz(q2, tcp_local)
    v = tuple(a + b for a, b in zip(JOINT1_TO_JOINT2, v))

    # Rotate by q1 (base) and translate into the ScaraT6 top frame.
    v = _rz(q1, v)
    v = tuple(a + b for a, b in zip(TOP_TO_JOINT1, v))

    return v


def check_limits(q1, q2, q4):
    """Print a warning for any joint outside its Webots limits."""
    for name, value in zip(
        ("base_arm_motor", "arm_motor", "shaft_linear_motor"), (q1, q2, q4)
    ):
        low, high = JOINT_LIMITS[name]
        if not (low <= value <= high):
            print(f"WARNING: {name}={value} outside [{low}, {high}].")


def main():
    if len(sys.argv) >= 4:
        q1, q2, q4 = (float(v) for v in sys.argv[1:4])
        check_limits(q1, q2, q4)
        x, y, z = fk(q1, q2, q4)
        print(f"q1={q1:.4f} rad, q2={q2:.4f} rad, q4={q4:.4f} m")
        print(f"TCP in ARM frame: x={x:.5f}  y={y:.5f}  z={z:.5f} m")
    else:
        print("No arguments - running demo poses.\n")
        demo = [
            ("zero pose (shaft fully up)", (0.0, 0.0, 0.0)),
            ("default pose", (0.3, 0.5, -0.1)),
        ]
        for label, q in demo:
            check_limits(*q)
            x, y, z = fk(*q)
            print(f"{label:30s} q={q} -> x={x:.4f}  y={y:.4f}  z={z:.4f}")
        print("\nUsage: python fk_scara.py q1 q2 q4")


if __name__ == "__main__":
    main()
