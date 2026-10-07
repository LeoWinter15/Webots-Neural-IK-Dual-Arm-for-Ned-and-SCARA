"""tcp_pose.py - Supervisor controller (Task 1.A/1.C).

Waits for a *_ready_*.flag file written by an arm controller once the
arm has settled, then measures the TCP position in the ARM frame and
appends it to tcp_measurements.csv in the project root. Finally it
deletes the flag, which tells the arm controller it may exit.

The CSV file has the header:
    arm,q1,q2,q3,x,y,z
and is created automatically on first use. It can be read directly by
the Task 1.C evaluation script to compare FK against Webots.
"""

import os
import math

from controller import Supervisor

# Webots runs controllers with CWD = <project>/controllers/<name>/,
# so "../../" reaches the project root. Everything is relative.
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RESULTS_CSV = os.path.join(ROOT, "tcp_measurements.csv")

FLAG_FILES = {
    "_ready_ned.flag": "ned",
    "_ready_scara.flag": "scara",
}

# Joint values are echoed into the CSV for convenience. They are read
# from the flag file content, which the arm controller is expected to
# fill as: "q1 q2 q3".


def get_tcp_in_arm_frame(robot, arm_node):
    """Return the TCP position (x, y, z) in the ARM frame."""
    tcp = robot.getFromDef("TCP")

    if tcp is None:
        tcp_slot = arm_node.getField("tcpSlot")
        if tcp_slot is None or tcp_slot.getCount() == 0:
            return None
        tcp = tcp_slot.getMFNode(0)

    pose = tcp.getPose(arm_node)
    # getPose returns a 16-element row-major 4x4 matrix;
    # translation is stored at indices 3, 7 and 11.
    return pose[3], pose[7], pose[11]


def append_result(arm, joints, x, y, z):
    """Append one measurement row, creating the CSV with header if needed."""
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a") as f:
        if write_header:
            f.write("arm,q1,q2,q3,x,y,z\n")
        f.write(f"{arm},{joints[0]},{joints[1]},{joints[2]},{x},{y},{z}\n")


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

arm_node = robot.getFromDef("ARM")
if arm_node is None:
    print("ERROR: Could not find DEF ARM")
    exit()

print("Supervisor ready. Waiting for an arm controller to settle...")

while robot.step(timestep) != -1:
    for flag_name, arm in FLAG_FILES.items():
        flag_path = os.path.join(ROOT, flag_name)

        if not os.path.exists(flag_path):
            continue

        # Read the joint values the controller was given (optional).
        try:
            with open(flag_path) as f:
                joints = [float(v) for v in f.read().split()]
            if len(joints) != 3 or not all(math.isfinite(v) for v in joints):
                raise ValueError("Expected three finite joints")
        except (ValueError, IndexError):
            print(f"ERROR: malformed request for {arm}; measurement rejected.")
            os.remove(flag_path)
            continue

        position = get_tcp_in_arm_frame(robot, arm_node)
        if position is None:
            print(f"ERROR: no TCP node found for {arm}.")
        else:
            x, y, z = position
            append_result(arm, joints, x, y, z)
            print(f"[{arm}] joints={joints} -> TCP in ARM frame: "
                  f"x={x:.5f}, y={y:.5f}, z={z:.5f}")

        # Deleting the flag tells the arm controller we are done.
        os.remove(flag_path)
