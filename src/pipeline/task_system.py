from __future__ import annotations

import json
import os
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

try:
    from .presets import available_presets
except ImportError:
    from presets import available_presets


TaskCallback = Callable[[Dict[str, Any]], Any]

DEFAULT_DATASET = "test_laptop_2014"
DEFAULT_LIMIT = 25

COMMAND_SETS: Dict[str, List[Any]] = {
    "full_stack_smoke": [
        {
            "task": "benchmark",
            "dataset": DEFAULT_DATASET,
            "preset": "full_stack",
            "limit": DEFAULT_LIMIT,
        }
    ],
    "compare_transformer": [
        {
            "task": "benchmark",
            "dataset": DEFAULT_DATASET,
            "preset": "full_stack",
            "limit": DEFAULT_LIMIT,
        },
        {
            "task": "benchmark",
            "dataset": DEFAULT_DATASET,
            "preset": "transformer_baseline",
            "limit": DEFAULT_LIMIT,
        },
    ],
}


@dataclass
class TaskDefinition:
    name: str
    callback: TaskCallback
    description: str = ""
    defaults: Dict[str, Any] = field(default_factory=dict)


class TaskSystem:
    def __init__(self) -> None:
        self._tasks: Dict[str, TaskDefinition] = {}

    def register(self, definition: TaskDefinition, *, override: bool = False) -> None:
        if definition.name in self._tasks and not override:
            raise ValueError(f"Task '{definition.name}' already registered")
        self._tasks[definition.name] = definition

    def run(self, name: str, params: Optional[Dict[str, Any]] = None) -> Any:
        if name not in self._tasks:
            raise KeyError(f"Unknown task '{name}'")
        definition = self._tasks[name]
        effective: Dict[str, Any] = dict(definition.defaults)
        if params:
            effective.update(params)
        return definition.callback(effective)

    def run_command(self, command: str) -> Any:
        name, params = parse_command(command)
        return self.run(name, params)

    def run_sequence(self, commands: Iterable[Any]) -> List[Any]:
        results: List[Any] = []
        for entry in commands:
            if isinstance(entry, str):
                command = entry.strip()
                if not command or command.startswith("#"):
                    continue
                results.append(self.run_command(command))
                continue
            if isinstance(entry, dict):
                task_name = entry.get("task")
                if not task_name:
                    raise ValueError(f"Missing 'task' key in {entry}")
                params = {k: v for k, v in entry.items() if k != "task"}
                results.append(self.run(task_name, params))
                continue
            raise TypeError(f"Unsupported command type: {type(entry)!r}")
        return results

    def run_file(self, path: str | Path) -> List[Any]:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        if file_path.suffix.lower() == ".json":
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return self.run_sequence(payload)
            raise TypeError("JSON task files must contain a list of commands or task objects")
        commands = file_path.read_text(encoding="utf-8").splitlines()
        return self.run_sequence(commands)

    def tasks(self) -> Dict[str, TaskDefinition]:
        return dict(self._tasks)


def parse_command(command: str) -> tuple[str, Dict[str, Any]]:
    tokens = shlex.split(command)
    if not tokens:
        raise ValueError("Empty command")
    name, *pairs = tokens
    params: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Malformed token '{pair}' in command '{command}'")
        key, value = pair.split("=", 1)
        params[key] = value
    return name, params


def _ensure_run_name(params: Dict[str, Any], default_prefix: str) -> None:
    if "run_name" not in params:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        params["run_name"] = f"{default_prefix}_{timestamp}"


def benchmark_task(params: Dict[str, Any]) -> Dict[str, Any]:
    from benchmark import run_benchmark

    params = dict(params)
    _ensure_run_name(params, params.get("preset", "run"))
    dataset = params.pop("dataset", None) or params.pop("dataset_name", None) or DEFAULT_DATASET
    preset = params.pop("preset", None)
    run_mode = params.pop("run_mode", None) or preset or "full_stack"

    limit = params.pop("limit", None)
    limit = int(limit) if limit not in (None, "None", "") else None

    pos = float(params.pop("pos", params.pop("pos_thresh", 0.1)))
    neg = float(params.pop("neg", params.pop("neg_thresh", -0.1)))

    return run_benchmark(
        run_name=params.pop("run_name"),
        dataset_name=dataset,
        run_mode=run_mode,
        limit=limit,
        pos_thresh=pos,
        neg_thresh=neg,
    )


def initialize_default_task_system() -> TaskSystem:
    system = TaskSystem()
    presets = ", ".join(available_presets())
    system.register(
        TaskDefinition(
            name="benchmark",
            callback=benchmark_task,
            description=f"Run ABSA benchmark. Available presets: {presets}",
            defaults={"preset": "full_stack", "dataset": DEFAULT_DATASET, "limit": DEFAULT_LIMIT},
        ),
        override=True,
    )
    return system


def available_command_sets() -> Dict[str, List[Any]]:
    return {name: list(commands) for name, commands in COMMAND_SETS.items()}


def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Task runner for ETSA benchmarks")
    parser.add_argument("commands", nargs="*", help="Inline task commands like 'benchmark dataset=test_laptop_2014'")
    parser.add_argument("--from-file", dest="task_file", help="Path to a task list (txt or json)")
    parser.add_argument("--sequence", choices=sorted(COMMAND_SETS.keys()), help="Name of a pre-defined command sequence")
    parser.add_argument("--list", action="store_true", help="List registered tasks and sequences")
    args = parser.parse_args(argv)

    system = initialize_default_task_system()

    if args.list:
        info = {
            "tasks": {name: definition.description for name, definition in system.tasks().items()},
            "sequences": available_command_sets(),
        }
        print(json.dumps(info, indent=2))
        return

    results: List[Any] = []
    if args.sequence:
        results = system.run_sequence(COMMAND_SETS[args.sequence])
    elif args.task_file:
        results = system.run_file(args.task_file)
    elif args.commands:
        results = system.run_sequence(args.commands)
    else:
        parser.error("Provide commands, --sequence, or --from-file")

    for result in results:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
