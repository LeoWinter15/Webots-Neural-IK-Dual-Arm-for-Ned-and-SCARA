"""Task 4.B shared I/O and provenance; standard library only for Webots."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLAN_PATH = ROOT / "results" / "task4b" / "plan.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan_id(plan):
    body = {k: v for k, v in plan.items() if k != "plan_id"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_plan(path=PLAN_PATH, check_sources=True):
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if plan["plan_id"] != plan_id(plan):
        raise ValueError("Task 4.B plan was edited; rerun prepare_task4b.py")
    if check_sources:
        for name, expected in plan["sha256"].items():
            if digest(ROOT / name) != expected:
                raise ValueError(f"Changed input/source: {name}; rerun prepare_task4b.py")
    return plan


def result_dir(plan):
    return ROOT / "results" / "task4b" / plan["plan_id"]


def sample_path(plan, arm, model, index):
    return result_dir(plan) / arm / f"{model}_{index:03d}.json"
