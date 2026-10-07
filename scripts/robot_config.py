"""Shared joint order, units, sampling bounds and label scaling for Tasks 2-4."""
CONFIG_VERSION = "physical_limits_v1"
ROBOTS = {
    "ned": {
        "joint_names": ["joint_1", "joint_2", "joint_3"],
        "units": ["rad", "rad", "rad"],
        "sampling_limits": [[-2.8, 2.8], [-0.8, 0.8], [0.0, 1.3]],
        "normalisation_limits": [[-2.8, 2.8], [-0.8, 0.8], [-1.5, 1.3]],
        "positive_elbow_index": 2,
    },
    "scara": {
        "joint_names": ["base_arm_motor", "arm_motor", "shaft_linear_motor"],
        "units": ["rad", "rad", "m"],
        "sampling_limits": [[-0.73, 0.73], [0.0, 0.83], [-0.2, 0.0]],
        "normalisation_limits": [[-0.73, 0.73], [-0.83, 0.83], [-0.2, 0.0]],
        "positive_elbow_index": 1,
    },
}


def normalise(q, lower, upper):
    return 2.0 * (q - lower) / (upper - lower) - 1.0


def denormalise(value, lower, upper):
    return lower + (value + 1.0) * (upper - lower) / 2.0


def normalise_joints(arm, joints):
    if len(joints) != 3:
        raise ValueError("Expected three joints")
    return tuple(normalise(v, *b) for v, b in zip(joints, ROBOTS[arm]["normalisation_limits"]))


def denormalise_joints(arm, values):
    if len(values) != 3:
        raise ValueError("Expected three outputs")
    return tuple(denormalise(v, *b) for v, b in zip(values, ROBOTS[arm]["normalisation_limits"]))
