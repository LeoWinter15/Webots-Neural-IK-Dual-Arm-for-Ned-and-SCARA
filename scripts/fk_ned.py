"""fk_ned.py - Task 1.B: forward kinematics of the TeachingNed arm.

Pure Python script (no Webots). Computes the TCP position in the ARM
frame from the three active joint values:

    q1 = joint_1  (waist yaw,      rad, q1 = 0 points along +Y, not +X!)
    q2 = joint_2  (shoulder pitch, rad)
    q3 = joint_3  (elbow pitch,    rad)

The wrist joints (joint_4..6) are fixed and do not enter the model.

Geometry (coursework specification, "Robot specifications - Ned"):
    base frame -> shoulder height   h  = 0.1065 + 0.0650 = 0.1715 m
    shoulder -> elbow vector        (A, B) = (0.2210, 0.0120) m
    elbow  -> TCP vector            (U, V) = (0.03185, 0.23000) m

In the arm's vertical plane each link vector is stored as
(v, h) = (vertical component, horizontal/radial component). A pitch
angle q rotates such a vector into the shoulder plane as:

    radial = v * sin(q) + h * cos(q)
    elev   = v * cos(q) - h * sin(q)

Derivation:
    1. Elbow-local TCP offset (U, V) is rotated by the elbow angle q3
       and added to the shoulder->elbow offset (A, B):
           wx = A + U*cos(q3) - V*sin(q3)    (vertical)
           wy = B + U*sin(q3) + V*cos(q3)    (radial)
    2. The shoulder angle q2 rotates (wx, wy) into the base frame:
           r = wx*sin(q2) + wy*cos(q2)       (horizontal radial dist)
           z = h + wx*cos(q2) - wy*sin(q2)   (world height)
    3. The waist angle q1 sweeps r around the z axis. Because the Ned
       zero pose points along +Y:
           x = -r*sin(q1),  y = r*cos(q1)

Usage:
    python fk_ned.py                 # demo table (zero pose + defaults)
    python fk_ned.py q1 q2 q3        # e.g. python fk_ned.py -0.54 0.4 0.9

Task 2 imports fk() to generate the training dataset.
"""

import math
import sys

# ------------------------------------------------------------------
# Ned geometry constants (metres / radians)
# ------------------------------------------------------------------

H = 0.1065 + 0.0650      # base frame -> shoulder joint height
A, B = 0.2210, 0.0120    # shoulder -> elbow  (A vertical, B radial)
# Exact values from the TeachingNed PROTO tcpSlot translation
# (coursework PDF rounds these to 0.03185 / 0.23000).
U, V = 0.0318503164, 0.22999994   # elbow -> teaching TCP

# Joint limits from the coursework specification (used for warnings).
JOINT_LIMITS = {
    "joint_1": (-2.8, 2.8),
    "joint_2": (-0.8, 0.8),
    "joint_3": (-1.5, 1.3),
}


def fk(q1, q2, q3):
    """Forward kinematics: joint angles (rad) -> TCP (x, y, z) in ARM frame."""
    # Step 1: TCP offset in the elbow frame, rotated by q3, plus the
    # shoulder->elbow offset -> TCP position in the shoulder plane.
    wx = A + U * math.cos(q3) - V * math.sin(q3)   # vertical component
    wy = B + U * math.sin(q3) + V * math.cos(q3)   # radial component

    # Step 2: rotate the shoulder plane by the shoulder pitch q2.
    r = wx * math.sin(q2) + wy * math.cos(q2)      # horizontal radius
    z = H + wx * math.cos(q2) - wy * math.sin(q2)  # world height

    # Step 3: waist yaw q1 about z (Ned zero pose points along +Y).
    x = -r * math.sin(q1)
    y = r * math.cos(q1)

    return x, y, z


def check_limits(q1, q2, q3):
    """Print a warning for any joint outside its Webots limits."""
    for name, value in zip(("joint_1", "joint_2", "joint_3"), (q1, q2, q3)):
        low, high = JOINT_LIMITS[name]
        if not (low <= value <= high):
            print(f"WARNING: {name}={value:.4f} rad outside [{low}, {high}].")


def main():
    if len(sys.argv) >= 4:
        q = [float(v) for v in sys.argv[1:4]]
        check_limits(*q)
        x, y, z = fk(*q)
        print(f"q = {q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f} rad")
        print(f"TCP in ARM frame: x={x:.5f}  y={y:.5f}  z={z:.5f} m")
    else:
        print("No arguments - running demo poses.\n")
        demo = [
            ("zero pose (arm points up)", (0.0, 0.0, 0.0)),
            ("lecture example pose", (-0.7854, 0.3504, 0.3554)),
        ]
        for label, q in demo:
            check_limits(*q)
            x, y, z = fk(*q)
            print(f"{label:30s} q={q} -> x={x:.4f}  y={y:.4f}  z={z:.4f}")
        print("\nUsage: python fk_ned.py q1 q2 q3")


if __name__ == "__main__":
    main()
