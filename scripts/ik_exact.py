import math


# Geometry from the simplified TeachingNed TCP marker placement.
# Shoulder height above ARM frame:
H = 0.1065 + 0.0650

# Shoulder -> elbow vector in the arm_link frame
A = 0.221
B = 0.012

# Elbow -> virtual TCP vector in the elbow_link frame
U = 0.0318503164
V = 0.22999994

L1 = math.hypot(A, B)
L2 = math.hypot(U, V)

# Angle offsets caused by the actual Ned local frames
GAMMA = math.atan2(A, B)
DELTA = math.atan2(U, V)


def ik_ned_webots_tcp(x, y, z, elbow="up"):
    """
    IK for the Webots Ned first-three-joint teaching TCP.

    x, y, z:
        target TCP position in the ARM frame, metres

    elbow:
        "up" or "down"

    returns:
        q1, q2, q3 motor targets in radians
    """

    # Webots Ned q1=0 points along +Y, not +X.
    q1 = math.atan2(-x, y)

    r = math.sqrt(x**2 + y**2)
    
    d = math.sqrt(r**2 + (z - H)**2)

    D = (d**2 - L1**2 - L2**2) / (2 * L1 * L2)

    # Check reachability
    EPSILON = 1e-9
    if D < -1.0 - EPSILON or D > 1.0 + EPSILON:
        raise ValueError(f"Target is outside geometric workspace: D={D:.3f}")

    # Avoid tiny numerical errors near the edge of the workspace
    D = max(-1.0, min(1.0, D))

    # Two possible elbow configurations
    if elbow == "up":
        q3 = math.atan2(math.sqrt(1 - D**2), D)
    elif elbow == "down":
        q3 = math.atan2(-math.sqrt(1 - D**2), D)
    else:
        raise ValueError("elbow must be 'up' or 'down'")

    q2 = math.atan2((z - H), r) - math.atan2(
        L2 * math.sin(q3),
        L1 + L2 * math.cos(q3)
    )

    q2 = GAMMA - q2
    q3 = DELTA - GAMMA - q3

    return q1, q2, q3


def fk_check(q1, q2, q3):
    """
    Webots-aware FK for the same virtual TCP point.
    """

    wx = A + U * math.cos(q3) - V * math.sin(q3)
    wy = B + U * math.sin(q3) + V * math.cos(q3)

    r = wy * math.cos(q2) + wx * math.sin(q2)
    z = H - wy * math.sin(q2) + wx * math.cos(q2)

    x = -r * math.sin(q1)
    y =  r * math.cos(q1)

    return x, y, z


x_target = 0.270
y_target = 0.320
z_target = 0.160

for elbow in ["up", "down"]:
    q1, q2, q3 = ik_ned_webots_tcp(
        x_target,
        y_target,
        z_target,
        elbow=elbow
    )

    print(f"\n{elbow} solution:")
    print(f"q1 = {q1:.4f} rad = {math.degrees(q1):.2f} deg")
    print(f"q2 = {q2:.4f} rad = {math.degrees(q2):.2f} deg")
    print(f"q3 = {q3:.4f} rad = {math.degrees(q3):.2f} deg")

    x, y, z = fk_check(q1, q2, q3)

    print("FK check:")
    print(f"x = {x:.4f}")
    print(f"y = {y:.4f}")
    print(f"z = {z:.4f}")