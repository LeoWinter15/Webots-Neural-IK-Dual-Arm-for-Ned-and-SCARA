from controller import Robot, Keyboard
import sys


robot = Robot()
time_step = int(robot.getBasicTimeStep())

keyboard = Keyboard()
keyboard.enable(time_step)


def step():
    if robot.step(time_step) == -1:
        sys.exit(0)


def passive_wait(seconds):
    start_time = robot.getTime()
    while start_time + seconds > robot.getTime():
        step()


def show_commands():
    print("------------COMMANDS---------------")
    print("Move joint_1 --> A and Z")
    print("Move joint_2 --> S and X")
    print("Move joint_3 --> D and C")
    print("Move joint_4 --> F and V")
    print("Move joint_5 --> G and B")
    print("Move joint_6 --> H and N")
    print("-----------------------------------")


show_commands()

motors = {}

motors[1] = robot.getDevice("joint_1")
motors[2] = robot.getDevice("joint_2")
motors[3] = robot.getDevice("joint_3")
motors[4] = robot.getDevice("joint_4")
motors[5] = robot.getDevice("joint_5")
motors[6] = robot.getDevice("joint_6")

sensors = {}

# Move every joint to its initial position and set the speed.
# Initialise the position sensor to allow incremental movement.
for i in range(1, 7):
    motors[i].setPosition(0.0)
    motors[i].setVelocity(1.0)
    sensors[i] = motors[i].getPositionSensor()
    sensors[i].enable(time_step)

step_size = 0.1

commands = {
    "A": (1, -step_size),
    "Z": (1, step_size),
    "S": (2, -step_size),
    "X": (2, step_size),
    "D": (3, -step_size),
    "C": (3, step_size),
    "F": (4, -step_size),
    "V": (4, step_size),
    "G": (5, -step_size),
    "B": (5, step_size),
    "H": (6, -step_size),
    "N": (6, step_size),
}


while robot.step(time_step) != -1:
    key = keyboard.getKey()

    if key == -1:
        continue

    try:
        key = chr(key).upper()
    except ValueError:
        continue
        
    if key in commands:
        idx, step = commands[key]
        motors[idx].setPosition(sensors[idx].getValue() + step)
        print(f"Move --> joint_{idx}")
