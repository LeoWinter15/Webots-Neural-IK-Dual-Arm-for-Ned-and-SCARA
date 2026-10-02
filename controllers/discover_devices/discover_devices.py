from controller import Robot, Node

robot = Robot()

# Build a lookup table from Webots node-type numbers to names.
node_type_names = {
    value: name
    for name, value in vars(Node).items()
    if name.isupper() and isinstance(value, int)
}

n = robot.getNumberOfDevices()
print(f"\nFound {n} devices:\n")

for i in range(n):
    device = robot.getDeviceByIndex(i)

    name = device.getName()
    node_type = device.getNodeType()
    type_name = node_type_names.get(node_type, str(node_type))
    model = device.getModel()

    print(f"{i:02d}: {name:30s}  type={type_name:20s}  model={model}")