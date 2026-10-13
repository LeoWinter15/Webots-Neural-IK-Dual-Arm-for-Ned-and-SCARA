"""Webots-only batch measurement; standard library, no NumPy installation needed.

The arm is itself a Supervisor: its motors move physically and getPose reads
the simulated TCP relative to its root. No FK is used to produce measurements.
"""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from task4b_common import load_plan, result_dir, sample_path, atomic_json


def wait_for_settle(robot, sensors, targets, tolerances, timestep, steps, timeout):
    start = robot.getTime()
    stable = 0
    actual = [None] * len(sensors)
    while robot.step(timestep) != -1:
        actual = [s.getValue() for s in sensors]
        inside = all(math.isfinite(a) and abs(a - t) < tol
                     for a, t, tol in zip(actual, targets, tolerances))
        stable = stable + 1 if inside else 0
        if stable >= steps:
            return "settled", actual, robot.getTime() - start
        if robot.getTime() - start >= timeout:
            return "settle_timeout", actual, robot.getTime() - start
    return "simulation_ended", actual, robot.getTime() - start


def measured_tcp(tcp, arm_node):
    pose = tcp.getPose(arm_node)
    values = [pose[3], pose[7], pose[11]]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Nonfinite Supervisor TCP measurement")
    return values


def main():
    from controller import Supervisor
    robot = Supervisor()
    try:
        if len(sys.argv) != 2 or sys.argv[1] not in ("ned", "scara"):
            raise ValueError('controllerArgs must be ["ned"] or ["scara"]')
        arm = sys.argv[1]
        plan = load_plan()
        settings = plan["settling"]
        timestep = int(robot.getBasicTimeStep())
        arm_node = robot.getSelf()
        tcp = robot.getFromDef("TCP")
        if tcp is None:
            slot = arm_node.getField("tcpSlot" if arm == "ned" else "handSlot")
            tcp = slot.getMFNode(0) if slot and slot.getCount() else None
        if tcp is None:
            raise ValueError("TCP marker is missing; use the task4b world")
        active_names = plan["robots"][arm]["joint_names"]
        fixed_names = ["joint_4", "joint_5", "joint_6"] if arm == "ned" else ["shaft_rotation_motor"]
        motors, sensors = [], []
        for name in active_names + fixed_names:
            motor = robot.getDevice(name)
            if motor is None:
                raise ValueError(f"Missing motor {name}")
            sensor = motor.getPositionSensor()
            if sensor is None:
                raise ValueError(f"Missing position sensor for {name}")
            sensor.enable(timestep)
            velocity = settings["linear_velocity"] if name == "shaft_linear_motor" else settings["angular_velocity"]
            motor.setVelocity(min(velocity, motor.getMaxVelocity()))
            motors.append(motor)
            sensors.append(sensor)
        tolerances = settings[arm + "_tolerances"] + [settings["fixed_tolerance_rad"]] * len(fixed_names)
        arm_folder = result_dir(plan) / arm
        session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
        atomic_json(arm_folder / "sessions" / f"{session}.json", {
            "plan_id": plan["plan_id"], "arm": arm, "session": session,
            "basic_timestep_ms": timestep, "world_path": robot.getWorldPath(),
            "robot_name": robot.getName(), "python": sys.version,
            "robot_pose_world": list(arm_node.getPose()), "settings": settings})

        def move(joints):
            target = list(joints) + [0.0] * len(fixed_names)
            for motor, value in zip(motors, target):
                motor.setPosition(value)
            return wait_for_settle(robot, sensors, target, tolerances, timestep,
                                   settings["consecutive_steps"], settings["timeout_s"])

        print(f"TASK4B START {arm}: plan={plan['plan_id']}, 3 models x 100 targets", flush=True)
        failures = 0
        for job in plan["jobs"][arm]:
            # Repeat a fixed starting pose for each model/run. No teleportation.
            status, actual, elapsed = move(plan["home"][arm])
            if status != "settled":
                atomic_json(arm_folder / "sessions" / f"{session}_home_failure.json", {
                    "status": status, "model": job["model"], "elapsed_s": elapsed,
                    "actual": [v if v is not None and math.isfinite(v) else None for v in actual]})
                raise RuntimeError(f"Home pose failed ({status}); inspect Console and scene")
            for sample in job["samples"]:
                path = sample_path(plan, arm, job["model"], sample["sample_index"])
                if path.exists():
                    previous = json.loads(path.read_text(encoding="utf-8"))
                    if previous["plan_id"] != plan["plan_id"]:
                        raise ValueError("Result/plan mismatch")
                    if previous["status"] == "settled":
                        continue
                status, actual, elapsed = move(sample["predicted_joints"])
                measured = None if status == "simulation_ended" else measured_tcp(tcp, arm_node)
                event = {"plan_id": plan["plan_id"], "session": session, "arm": arm,
                         "model": job["model"], **sample, "status": status,
                         "elapsed_simulation_s": elapsed, "simulation_time_s": robot.getTime(),
                         "utc": datetime.now(timezone.utc).isoformat(),
                         "actual_joints": [v if v is not None and math.isfinite(v) else None for v in actual[:3]],
                         "actual_fixed_joints": [v if v is not None and math.isfinite(v) else None for v in actual[3:]],
                         "measured_xyz_m": measured}
                # Preserve every attempt, even when retrying a previous failure.
                attempt = arm_folder / "attempts" / f"{session}_{job['model']}_{sample['sample_index']:03d}.json"
                atomic_json(attempt, event)
                atomic_json(path, event)
                print(f"TASK4B {arm} {job['model']} {sample['sample_index']:03d}/100 {status}", flush=True)
                if status == "simulation_ended":
                    return
                failures += status != "settled"
        from summarize_task4b import summarize
        summarize(plan)
        print(f"TASK4B DONE {arm}: new failures={failures}. Results: {arm_folder}", flush=True)
        robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
    except Exception as error:
        print(f"TASK4B ERROR: {type(error).__name__}: {error}", flush=True)
        robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
        raise


if __name__ == "__main__":
    main()
