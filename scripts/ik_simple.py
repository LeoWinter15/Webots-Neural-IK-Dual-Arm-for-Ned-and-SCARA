import math


H = 0.1715   # base frame to shoulder joint, metres
L1 = 0.2210  # shoulder to elbow, metres
L2 = 0.2320  # elbow to wrist point, metres


def inverse_kinematics(x, y, z, elbow="up"):
    """
    Inverse kinematics for simplified 3-DOF Ned arm.

    Input:
        x, y, z: target wrist position in metres, relative to robot base frame
        elbow: "up" or "down"

    Returns:
        q1, q2, q3 in radians
    """

    # Waist rotation
    q1 = math.atan2(y, x)

    # Convert target into the vertical plane of the arm
    r = math.sqrt(x**2 + y**2)
    
    d = math.sqrt(r**2 + (z - H)**2)

    # Cosine-rule term
    D = (d**2 - L1**2 - L2**2) / (2 * L1 * L2)

    # Check reachability
    EPSILON = 1e-9
    if D < -1.0 - EPSILON or D > 1.0 + EPSILON:
        raise ValueError("Target is outside the reachable workspace")

    # Avoid tiny numerical errors near the edge of the workspace
    D = max(-1.0, min(1.0, D))

    # Two possible elbow configurations
    if elbow == "up":
        q3 = math.atan2(math.sqrt(1 - D**2), D)
    elif elbow == "down":
        q3 = math.atan2(-math.sqrt(1 - D**2), D)
    else:
        raise ValueError("elbow must be 'up' or 'down'")

    # Shoulder angle
    q2 = math.atan2((z - H), r) - math.atan2(
        L2 * math.sin(q3),
        L1 + L2 * math.cos(q3)
    )

    return q1, q2, q3


def forward_kinematics(q1, q2, q3):
    """
    Forward kinematics for checking the IK result.
    """

    r = L1 * math.cos(q2) + L2 * math.cos(q2 + q3)

    x = math.cos(q1) * r
    y = math.sin(q1) * r
    z = H + L1 * math.sin(q2) + L2 * math.sin(q2 + q3)

    return x, y, z


# Example target
x_target = 0.20
y_target = 0.20
z_target = 0.25

q1, q2, q3 = inverse_kinematics(
    x_target,
    y_target,
    z_target,
    elbow="up"
)

print("Joint angles:")
print(f"q1 = {q1:.4f} rad = {math.degrees(q1):.2f} deg")
print(f"q2 = {q2:.4f} rad = {math.degrees(q2):.2f} deg")
print(f"q3 = {q3:.4f} rad = {math.degrees(q3):.2f} deg")

x_check, y_check, z_check = forward_kinematics(q1, q2, q3)

print("\nFK check:")
print(f"x = {x_check:.4f} m")
print(f"y = {y_check:.4f} m")
print(f"z = {z_check:.4f} m")
