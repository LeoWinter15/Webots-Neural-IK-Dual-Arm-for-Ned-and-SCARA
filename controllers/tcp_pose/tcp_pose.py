from controller import Supervisor

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

arm = robot.getFromDef("ARM")

if arm is None:
    print("ERROR: Could not find DEF ARM")
    exit()

# First try global DEF lookup
tcp = robot.getFromDef("TCP")

# If that fails, retrieve the first node from the tcpSlot field
if tcp is None:
    print("Could not find global DEF TCP. Trying ARM.tcpSlot instead...")

    tcp_slot = arm.getField("tcpSlot")

    if tcp_slot is None:
        print("ERROR: ARM has no field called tcpSlot.")
        print("This usually means the world is still using Ned, not TeachingNed,")
        print("or the tcpSlot field was not added to the PROTO interface.")
        exit()

    if tcp_slot.getCount() == 0:
        print("ERROR: ARM.tcpSlot exists but is empty.")
        print("Add a TCP marker inside tcpSlot [ ... ] in the world file.")
        exit()

    tcp = tcp_slot.getMFNode(0)

print("TCP node found.")

while robot.step(timestep) != -1:
    x, y, z = tcp.getPosition()
    print(f"TCP in world frame: x={x:.3f}, y={y:.3f}, z={z:.3f}")

    # 4x4 homogeneous transform of TCP relative to ARM
    pose = tcp.getPose(arm)

    # Webots returns it as:
    # [ R00 R01 R02 Tx
    #   R10 R11 R12 Ty
    #   R20 R21 R22 Tz
    #   0   0   0   1  ]
    x = pose[3]
    y = pose[7]
    z = pose[11]

    print(f"TCP in ARM frame: x={x:.4f}, y={y:.4f}, z={z:.4f}")