"""scara_simple.py - Task 1.A controller for the Epson SCARA (ScaraT6).

Accepts three joint values - base_arm_motor and arm_motor in radians,
shaft_linear_motor in metres - commands the Webots motors, waits for the
arm to settle, and signals the supervisor (tcp_pose.py) to record the
measured TCP position.

Joint values are read from controller arguments (see controllerArgs in
the world file) or, if empty, from the DEFAULT_JOINTS below.

Handshake protocol (via files in the project root, no hardcoded paths):
  1. This controller writes ../../_ready_scara.flag once the arm settled.
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

MOTOR_NAMES = ["base_arm_motor", "arm_motor", "shaft_linear_motor"]

# End-effector yaw is ignored in this coursework; fix it at 0.
FIXED_MOTOR_NAMES = ["shaft_rotation_motor"]

# Joint limits from the coursework specification.
# Units: rad, rad, metres.
JOINT_LIMITS = {
    "base_arm_motor": (-0.73, 0.73),
    "arm_motor": (-0.83, 0.83),
    "shaft_linear_motor": (-0.2, 0.0),
}

# (rad, rad, m) - used when started without arguments.
DEFAULT_JOINTS = [0.3, 0.5, -0.1]

MOTOR_VELOCITY = 0.5          # rad/s for revolute joints
LINEAR_VELOCITY = 0.05        # m/s for the prismatic shaft
SETTLE_TOL = [0.002, 0.002, 0.0005]   # rad, rad, m per joint
SETTLE_STEPS = 20
TIMEOUT = 60.0

# Webots runs controllers with CWD = <project>/controllers/<name>/,
# so "../../" reaches the project root. Everything is relative.
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
READY_FLAG = os.path.join(ROOT, "_ready_scara.flag")


def parse_joint_args():
    """Read q1,q2,q4 from sys.argv, fall back to DEFAULT_JOINTS."""
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
            print(f"WARNING: {name}={value} outside [{low}, {high}], clipped.")
        clipped.append(max(low, min(high, value)))
    return clipped


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

robot = Robot()
timestep = int(robot.getBasicTimeStep())

targets = clip_to_limits(parse_joint_args())
print(f"Target joints: base={targets[0]:.4f} rad, arm={targets[1]:.4f} rad, "
      f"shaft={targets[2]:.4f} m")

motors = [robot.getDevice(name) for name in MOTOR_NAMES]
sensors = []

for name, motor, target in zip(MOTOR_NAMES, motors, targets):
    velocity = LINEAR_VELOCITY if name == "shaft_linear_motor" else MOTOR_VELOCITY
    motor.setVelocity(velocity)
    motor.setPosition(target)
    sensor = motor.getPositionSensor()
    sensor.enable(timestep)
    sensors.append(sensor)

# Fix the ignored shaft-rotation motor at 0.
for name in FIXED_MOTOR_NAMES:
    motor = robot.getDevice(name)
    if motor is not None:
        motor.setVelocity(MOTOR_VELOCITY)
        motor.setPosition(0.0)

# Remove a stale flag from a previous run, if any.
if os.path.exists(READY_FLAG):
    os.remove(READY_FLAG)

# Wait until every joint is within its tolerance of the target.
stable_count = 0
start_time = robot.getTime()
settled = False
failure_reason = "simulation_ended"

while robot.step(timestep) != -1:
    errors = [abs(t - s.getValue()) for t, s in zip(targets, sensors)]
    scaled = [e / tol for e, tol in zip(errors, SETTLE_TOL)]

    if max(scaled) < 1.0:
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
        "arm": "scara", "status": failure_reason,
        "target_joints": targets,
        "actual_joints": [v if math.isfinite(v) else None for v in actual],
        "elapsed_s": robot.getTime() - start_time,
    }
    with open(os.path.join(ROOT, "motion_failures.jsonl"), "a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, allow_nan=False) + "\n")
    sys.exit(1)

if settled:
    print(f"Arm settled after {robot.getTime() - start_time:.2f} s.")

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
