from controller import Robot

SHOULDER_POSE = 10.5
ELBOW_POSE = -1.3
SHAFT_TRANS = -0.1

robot = Robot()
timestep = int(robot.getBasicTimeStep())

shoulder = robot.getDevice("base_arm_motor")
elbow = robot.getDevice("arm_motor")
shaft_rotation = robot.getDevice("shaft_rotation_motor")
shaft_linear = robot.getDevice("shaft_linear_motor")

shoulder.setPosition(SHOULDER_POSE)
elbow.setPosition(ELBOW_POSE)
shaft_rotation.setPosition(0.0)
shaft_linear.setPosition(SHAFT_TRANS)

while robot.step(timestep) != -1:
    pass
