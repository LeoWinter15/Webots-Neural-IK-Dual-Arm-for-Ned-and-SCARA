"""ned_simple.py - Task 1.A controller for the TeachingNed arm.

Accepts three joint values (radians) for joint_1..joint_3, commands the
Webots motors, waits for the arm to settle, and then signals the
supervisor (tcp_pose.py) to record the measured TCP position.

Joint values are read from controller arguments (see controllerArgs in
the world file) or, if empty, from the DEFAULT_JOINTS below.

Handshake protocol (via files in the project root, no hardcoded paths):
  1. This controller writes ../../_ready_ned.flag once the arm settled.
  2. The supervisor reads the TCP pose, appends it to
     ../../tcp_measurements.csv and deletes the flag.
"""

import os
import json
import math
from datetime import datetime, timezone
import sys

from controller import Robot

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

MOTOR_NAMES = ["joint_1", "joint_2", "joint_3"]

# Wrist joints are fixed for this coursework; park them at 0.
FIXED_MOTOR_NAMES = ["joint_4", "joint_5", "joint_6"]

# Joint limits from the coursework specification (rad).
JOINT_LIMITS = {
    "joint_1": (-2.8, 2.8),
    "joint_2": (-0.8, 0.8),
    "joint_3": (-1.5, 1.3),
}

# Used when the controller is started without arguments.
DEFAULT_JOINTS = [-0.7854, 0.3504, 0.3554]

MOTOR_VELOCITY = 0.7      # rad/s, keeps the motion slow and smooth
SETTLE_TOL = 0.002        # rad, |target - current| must be below this
SETTLE_STEPS = 20         # ...for this many consecutive steps
TIMEOUT = 60.0            # s, safety limit for the settle wait

# Webots runs controllers with CWD = <project>/controllers/<name>/,
# so "../../" reaches the project root. Everything is relative.
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
READY_FLAG = os.path.join(ROOT, "_ready_ned.flag")


def parse_joint_args():
    """Read q1,q2,q3 from sys.argv, fall back to DEFAULT_JOINTS."""
    if len(sys.argv) >= 4:
        return [float(v) for v in sys.argv[1:4]]
    print("No joint arguments given, using DEFAULT_JOINTS.")
    return list(DEFAULT_JOINTS)


def clip_to_limits(values):
    """Clamp each joint into its limits and warn if clipping occurred."""
    clipped = []
    for name, value in zip(MOTOR_NAMES, values):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite target for {name}")
        low, high = JOINT_LIMITS[name]
        if not (low <= value <= high):
            print(f"WARNING: {name}={value:.4f} outside [{low}, {high}], clipped.")
        clipped.append(max(low, min(high, value)))
    return clipped


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

robot = Robot()
timestep = int(robot.getBasicTimeStep())

targets = clip_to_limits(parse_joint_args())
print(f"Target joints: q1={targets[0]:.4f}, q2={targets[1]:.4f}, q3={targets[2]:.4f} rad")

motors = [robot.getDevice(name) for name in MOTOR_NAMES]
sensors = []

for motor, target in zip(motors, targets):
    motor.setVelocity(MOTOR_VELOCITY)
    motor.setPosition(target)
    sensor = motor.getPositionSensor()
    sensor.enable(timestep)
    sensors.append(sensor)

# Park the fixed wrist joints at 0.
for name in FIXED_MOTOR_NAMES:
    motor = robot.getDevice(name)
    if motor is not None:
        motor.setVelocity(MOTOR_VELOCITY)
        motor.setPosition(0.0)

# Remove a stale flag from a previous run, if any.
if os.path.exists(READY_FLAG):
    os.remove(READY_FLAG)

# Wait until all joints are within SETTLE_TOL of their target.
stable_count = 0
start_time = robot.getTime()
settled = False
failure_reason = "simulation_ended"

while robot.step(timestep) != -1:
    errors = [abs(t - s.getValue()) for t, s in zip(targets, sensors)]

    if max(errors) < SETTLE_TOL:
        stable_count += 1
    else:
        stable_count = 0

    if stable_count >= SETTLE_STEPS:
        settled = True
        break

    if robot.getTime() - start_time > TIMEOUT:
        failure_reason = "settle_timeout"
        print("ERROR: settle wait timed out; measurement rejected.")
        break

if not settled:
    # Failed motion must never enter the valid TCP measurement CSV.
    actual = [sensor.getValue() for sensor in sensors]
    event = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "arm": "ned", "status": failure_reason,
        "target_joints": targets,
        "actual_joints": [v if math.isfinite(v) else None for v in actual],
        "elapsed_s": robot.getTime() - start_time,
    }
    with open(os.path.join(ROOT, "motion_failures.jsonl"), "a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, allow_nan=False) + "\n")
    sys.exit(1)

if settled:
    print(f"Arm settled after {robot.getTime() - start_time:.2f} s "
          f"(max residual error {max(errors):.5f} rad).")

# Signal the supervisor that the TCP can now be measured.
with open(READY_FLAG + ".tmp", "w") as f:
    f.write(f"{targets[0]} {targets[1]} {targets[2]}\n")
os.replace(READY_FLAG + ".tmp", READY_FLAG)
print("Flag written, waiting for supervisor to measure the TCP...")

# Wait until the supervisor has consumed the flag, then exit.
while robot.step(timestep) != -1:
    if not os.path.exists(READY_FLAG):
        print("TCP measured by supervisor. Done.")
        break
    if robot.getTime() - start_time > TIMEOUT + 30.0:
        print("WARNING: supervisor did not respond, exiting anyway.")
        break
