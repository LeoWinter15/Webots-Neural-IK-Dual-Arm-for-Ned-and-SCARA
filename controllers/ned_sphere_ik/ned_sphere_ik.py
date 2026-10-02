import math
from controller import Supervisor


# ============================================================
# Webots DEF names
# ============================================================

ARM_DEF_NAME = "ARM"
TARGET_DEF_NAME = "TARGET"
TCP_DEF_NAME = "TCP"


# ============================================================
# Ned Webots-aware teaching TCP geometry
# ============================================================
# This model assumes the artificial TCP is attached inside elbow_link at:
#
#     translation 0.03185 0.23000 0
#
# The target position is measured in the ARM frame.

H = 0.1065 + 0.0650

A = 0.2210
B = 0.0120

U = 0.0318503164
V = 0.22999994

L1 = math.sqrt(A**2 + B**2)
L2 = math.sqrt(U**2 + V**2)

GAMMA = math.atan2(A, B)
DELTA = math.atan2(U, V)


# ============================================================
# Joint limits
# ============================================================
# These match the first three Ned joints reasonably well.
# Keep them conservative for coursework demos.

JOINT_LIMITS = {
    "joint_1": (-2.8, 2.8),
    "joint_2": (-0.8, 0.8),
    "joint_3": (-1.5, 1.3),
}


# ============================================================
# Helper functions
# ============================================================

def clip(value, low, high):
    return max(low, min(high, value))


def within_limits(q1, q2, q3):
    return (
        JOINT_LIMITS["joint_1"][0] <= q1 <= JOINT_LIMITS["joint_1"][1]
        and JOINT_LIMITS["joint_2"][0] <= q2 <= JOINT_LIMITS["joint_2"][1]
        and JOINT_LIMITS["joint_3"][0] <= q3 <= JOINT_LIMITS["joint_3"][1]
    )


def clip_to_limits(q1, q2, q3):
    q1 = clip(q1, *JOINT_LIMITS["joint_1"])
    q2 = clip(q2, *JOINT_LIMITS["joint_2"])
    q3 = clip(q3, *JOINT_LIMITS["joint_3"])

    return q1, q2, q3


def fk_ned(q1, q2, q3):
    """
    Forward kinematics for the Webots-aware teaching TCP.

    Returns:
        x, y, z in ARM frame
    """

    wx = A + U * math.cos(q3) - V * math.sin(q3)
    wy = B + U * math.sin(q3) + V * math.cos(q3)

    r = wy * math.cos(q2) + wx * math.sin(q2)
    z = H - wy * math.sin(q2) + wx * math.cos(q2)

    x = -r * math.sin(q1)
    y =  r * math.cos(q1)

    return x, y, z


def position_error(q1, q2, q3, target_x, target_y, target_z):
    x, y, z = fk_ned(q1, q2, q3)

    dx = x - target_x
    dy = y - target_y
    dz = z - target_z

    return math.sqrt(dx**2 + dy**2 + dz**2)


def get_xyz_relative_to_arm(node, arm_node):
    """
    Returns the node position in the ARM frame.

    getPose(reference_node) returns a 4x4 transformation matrix.
    The translation terms are indices 3, 7, and 11.
    """

    pose = node.getPose(arm_node)

    x = pose[3]
    y = pose[7]
    z = pose[11]

    return x, y, z


# ============================================================
# Inverse kinematics
# ============================================================

def ik_ned_all_solutions(target_x, target_y, target_z):
    """
    Calculate both mathematical IK branches.

    Returns a list of candidate solutions:
        [(q1, q2, q3, branch_name, geometric_reachable), ...]
    """

    solutions = []

    # Webots Ned convention:
    # q1 = 0 points along +Y, not +X.
    q1 = math.atan2(-target_x, target_y)

    r = math.sqrt(target_x**2 + target_y**2)
    zp = target_z - H

    d_squared = r**2 + zp**2

    D_raw = (d_squared - L1**2 - L2**2) / (2 * L1 * L2)

    geometric_reachable = True

    if D_raw > 1.0:
        D = 1.0
        geometric_reachable = False
    elif D_raw < -1.0:
        D = -1.0
        geometric_reachable = False
    else:
        D = D_raw

    sin_part = math.sqrt(max(0.0, 1.0 - D**2))

    theta = math.atan2(zp, r)

    for branch_name, sign in [
        ("branch_A", 1.0),
        ("branch_B", -1.0),
    ]:
        elbow_angle = math.atan2(sign * sin_part, D)

        beta = math.atan2(
            L2 * math.sin(elbow_angle),
            L1 + L2 * math.cos(elbow_angle)
        )

        alpha1 = theta - beta

        q2 = GAMMA - alpha1
        q3 = DELTA - GAMMA - elbow_angle

        solutions.append(
            (q1, q2, q3, branch_name, geometric_reachable)
        )

    return solutions


def choose_best_ik_solution(target_x, target_y, target_z, current_q=None):
    """
    Choose the best IK solution.

    Preference order:
      1. Solutions inside the joint limits.
      2. Small TCP position error after clipping.
      3. Small movement from the current joint position, if available.
    """

    candidates = ik_ned_all_solutions(target_x, target_y, target_z)

    scored_candidates = []

    for q1, q2, q3, branch_name, geometric_reachable in candidates:
        is_inside_limits = within_limits(q1, q2, q3)

        q1_clipped, q2_clipped, q3_clipped = clip_to_limits(q1, q2, q3)

        err = position_error(
            q1_clipped,
            q2_clipped,
            q3_clipped,
            target_x,
            target_y,
            target_z,
        )

        limit_penalty = 0.0 if is_inside_limits else 100.0
        reach_penalty = 0.0 if geometric_reachable else 1000.0

        movement_penalty = 0.0

        if current_q is not None:
            movement_penalty = (
                (q1_clipped - current_q[0])**2
                + (q2_clipped - current_q[1])**2
                + (q3_clipped - current_q[2])**2
            )

        score = err + limit_penalty + reach_penalty + 0.01 * movement_penalty

        scored_candidates.append(
            {
                "score": score,
                "error": err,
                "q": (q1_clipped, q2_clipped, q3_clipped),
                "raw_q": (q1, q2, q3),
                "branch": branch_name,
                "inside_limits": is_inside_limits,
                "geometric_reachable": geometric_reachable,
            }
        )

    scored_candidates.sort(key=lambda item: item["score"])

    return scored_candidates[0]


# ============================================================
# Main controller
# ============================================================

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())


# ------------------------------------------------------------
# Get world nodes
# ------------------------------------------------------------

arm_node = robot.getFromDef(ARM_DEF_NAME)
target_node = robot.getFromDef(TARGET_DEF_NAME)
tcp_node = robot.getFromDef(TCP_DEF_NAME)

if arm_node is None:
    print(f"ERROR: Could not find DEF {ARM_DEF_NAME}")
    exit()

if target_node is None:
    print(f"ERROR: Could not find DEF {TARGET_DEF_NAME}")
    exit()

if tcp_node is None:
    print(f"WARNING: Could not find DEF {TCP_DEF_NAME}")
    print("The controller will still run, but it cannot print TCP error.")


# ------------------------------------------------------------
# Get motors
# ------------------------------------------------------------

joint_1 = robot.getDevice("joint_1")
joint_2 = robot.getDevice("joint_2")
joint_3 = robot.getDevice("joint_3")

joint_4 = robot.getDevice("joint_4")
joint_5 = robot.getDevice("joint_5")
joint_6 = robot.getDevice("joint_6")


# ------------------------------------------------------------
# Set motor velocities
# ------------------------------------------------------------

joint_1.setVelocity(0.7)
joint_2.setVelocity(0.7)
joint_3.setVelocity(0.7)

if joint_4 is not None:
    joint_4.setVelocity(0.7)

if joint_5 is not None:
    joint_5.setVelocity(0.7)

if joint_6 is not None:
    joint_6.setVelocity(0.7)


# ------------------------------------------------------------
# Try to enable position sensors
# ------------------------------------------------------------

sensors = []

for motor in [joint_1, joint_2, joint_3]:
    sensor = motor.getPositionSensor()

    if sensor is not None:
        sensor.enable(timestep)

    sensors.append(sensor)


# ------------------------------------------------------------
# Initial position
# ------------------------------------------------------------

joint_1.setPosition(0.0)
joint_2.setPosition(0.3)
joint_3.setPosition(0.3)

if joint_4 is not None:
    joint_4.setPosition(0.0)

if joint_5 is not None:
    joint_5.setPosition(0.0)

if joint_6 is not None:
    joint_6.setPosition(0.0)


# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------

last_print_time = -999.0
print_period = 0.5

last_command_time = -999.0
command_period = 0.05

while robot.step(timestep) != -1:
    now = robot.getTime()

    # Read target sphere position in the ARM frame.
    target_x, target_y, target_z = get_xyz_relative_to_arm(
        target_node,
        arm_node
    )

    # Read current joint positions if sensors are available.
    current_q = None

    if all(sensor is not None for sensor in sensors):
        current_q = [
            sensors[0].getValue(),
            sensors[1].getValue(),
            sensors[2].getValue(),
        ]

    # Recompute and send IK command periodically.
    if now - last_command_time >= command_period:
        solution = choose_best_ik_solution(
            target_x,
            target_y,
            target_z,
            current_q=current_q,
        )

        q1, q2, q3 = solution["q"]

        joint_1.setPosition(q1)
        joint_2.setPosition(q2)
        joint_3.setPosition(q3)

        # Keep wrist joints fixed for this coursework model.
        if joint_4 is not None:
            joint_4.setPosition(0.0)

        if joint_5 is not None:
            joint_5.setPosition(0.0)

        if joint_6 is not None:
            joint_6.setPosition(0.0)

        last_command_time = now

    # Print useful information for debugging/video evidence.
    if now - last_print_time >= print_period:
        solution = choose_best_ik_solution(
            target_x,
            target_y,
            target_z,
            current_q=current_q,
        )

        q1, q2, q3 = solution["q"]

        print("\n--------------------------------")
        print("Target position in ARM frame:")
        print(f"x = {target_x: .4f}, y = {target_y: .4f}, z = {target_z: .4f}")

        print("Commanded joints:")
        print(f"joint_1 q1 = {q1: .4f} rad")
        print(f"joint_2 q2 = {q2: .4f} rad")
        print(f"joint_3 q3 = {q3: .4f} rad")

        print(f"IK branch: {solution['branch']}")

        if not solution["geometric_reachable"]:
            print("WARNING: Target is outside the geometric workspace.")

        if not solution["inside_limits"]:
            print("WARNING: Mathematical solution exceeds joint limits.")
            print("The command has been clipped to safe joint limits.")

        expected_error = position_error(
            q1,
            q2,
            q3,
            target_x,
            target_y,
            target_z,
        )

        print(f"Expected model error after clipping = {expected_error:.4f} m")

        if tcp_node is not None:
            tcp_x, tcp_y, tcp_z = get_xyz_relative_to_arm(
                tcp_node,
                arm_node
            )

            measured_error = math.sqrt(
                (tcp_x - target_x)**2
                + (tcp_y - target_y)**2
                + (tcp_z - target_z)**2
            )

            print("Measured TCP position in ARM frame:")
            print(f"x = {tcp_x: .4f}, y = {tcp_y: .4f}, z = {tcp_z: .4f}")
            print(f"Measured TCP error = {measured_error:.4f} m")

        last_print_time = now