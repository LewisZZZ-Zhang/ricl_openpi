"""Merge metrics produced by task-sharded LIBERO RICL evaluation clients."""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
from typing import Any


def _timestamp() -> str:
    # The LIBERO client environment uses Python 3.8, which predates datetime.UTC.
    return datetime.datetime.now(datetime.timezone.utc).isoformat()  # noqa: UP017


def _write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value["updated_at"] = _timestamp()
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def per_task_metrics(merged: dict[str, Any]) -> dict[str, Any]:
    tasks = []
    for suite_name, suite in merged["suites"].items():
        for task in suite["tasks"]:
            tasks.append(
                {
                    "suite": suite_name,
                    "task_global_index": int(task["task_global_index"]),
                    "task_index": int(task["task_index"]),
                    "task_id": int(task["task_id"]),
                    "task_name": task["task_name"],
                    "episodes": int(task["episodes"]),
                    "successes": int(task["successes"]),
                    "success_rate": float(task["success_rate"]),
                }
            )

    tasks.sort(key=lambda task: int(task["task_global_index"]))
    return {
        "status": merged["status"],
        "started_at": merged["started_at"],
        "completed_at": merged["completed_at"],
        "updated_at": _timestamp(),
        "config": merged["config"],
        "overall": merged["overall"],
        "tasks": tasks,
    }


def _load_shards(shards_dir: pathlib.Path, num_shards: int) -> list[tuple[pathlib.Path, dict[str, Any]]]:
    shards = []
    for shard_index in range(num_shards):
        path = shards_dir / f"shard_{shard_index:02d}_of_{num_shards:02d}" / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing shard metrics: {path}")
        metrics = json.loads(path.read_text(encoding="utf-8"))
        if metrics.get("status") != "completed":
            raise ValueError(f"Shard {shard_index} is not completed: status={metrics.get('status')!r}")
        config = metrics.get("config", {})
        if config.get("num_task_shards") != num_shards:
            raise ValueError(
                f"Shard {shard_index} reports num_task_shards={config.get('num_task_shards')!r}, "
                f"expected {num_shards}"
            )
        if config.get("task_shard_index") != shard_index:
            raise ValueError(
                f"Metrics at {path} report task_shard_index={config.get('task_shard_index')!r}"
            )
        shards.append((path, metrics))
    return shards


def merge_metrics(shards_dir: pathlib.Path, num_shards: int) -> dict[str, Any]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    shards = _load_shards(shards_dir, num_shards)

    common_config = dict(shards[0][1]["config"])
    common_config.pop("task_shard_index")
    for path, metrics in shards[1:]:
        shard_config = dict(metrics["config"])
        shard_config.pop("task_shard_index")
        if shard_config != common_config:
            raise ValueError(f"Evaluation config mismatch in {path}")
    common_config["parallel_clients"] = num_shards

    suite_names = list(shards[0][1]["suites"])
    if any(list(metrics["suites"]) != suite_names for _, metrics in shards[1:]):
        raise ValueError("Shard suite lists do not match")

    merged_suites: dict[str, Any] = {}
    seen_tasks: set[tuple[str, int]] = set()
    seen_global_indices: list[int] = []
    for suite_name in suite_names:
        tasks = []
        for _, metrics in shards:
            for task in metrics["suites"][suite_name]["tasks"]:
                task_key = (suite_name, int(task["task_index"]))
                if task_key in seen_tasks:
                    raise ValueError(f"Duplicate task across shards: {suite_name}/{task['task_index']}")
                seen_tasks.add(task_key)

                global_index = int(task["task_global_index"])
                shard_index = int(metrics["config"]["task_shard_index"])
                if global_index % num_shards != shard_index:
                    raise ValueError(
                        f"Task global index {global_index} is in shard {shard_index}, "
                        f"expected shard {global_index % num_shards}"
                    )
                seen_global_indices.append(global_index)
                tasks.append(task)

        tasks.sort(key=lambda task: int(task["task_index"]))
        episodes = sum(int(task["episodes"]) for task in tasks)
        successes = sum(int(task["successes"]) for task in tasks)
        merged_suites[suite_name] = {
            "tasks": tasks,
            "num_tasks": len(tasks),
            "episodes": episodes,
            "successes": successes,
            "success_rate": successes / episodes if episodes else 0.0,
        }

    if sorted(seen_global_indices) != list(range(len(seen_global_indices))):
        raise ValueError(
            "Merged task_global_index values are not contiguous from zero: "
            f"{sorted(seen_global_indices)}"
        )

    episodes = sum(suite["episodes"] for suite in merged_suites.values())
    successes = sum(suite["successes"] for suite in merged_suites.values())
    tasks = sum(suite["num_tasks"] for suite in merged_suites.values())
    return {
        "status": "completed",
        "started_at": min(metrics["started_at"] for _, metrics in shards),
        "completed_at": max(metrics["completed_at"] for _, metrics in shards),
        "updated_at": _timestamp(),
        "config": common_config,
        "shards": [
            {
                "shard_index": int(metrics["config"]["task_shard_index"]),
                "metrics_path": str(path.resolve()),
                "tasks": int(metrics["overall"]["tasks"]),
                "episodes": int(metrics["overall"]["episodes"]),
                "successes": int(metrics["overall"]["successes"]),
                "success_rate": float(metrics["overall"]["success_rate"]),
            }
            for path, metrics in shards
        ],
        "suites": merged_suites,
        "overall": {
            "tasks": tasks,
            "episodes": episodes,
            "successes": successes,
            "success_rate": successes / episodes if episodes else 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-dir", type=pathlib.Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output-path", type=pathlib.Path, required=True)
    args = parser.parse_args()
    merged = merge_metrics(args.shards_dir.expanduser(), args.num_shards)
    output_path = args.output_path.expanduser()
    _write_json(output_path, merged)
    per_task_output_path = output_path.with_name("per_task_metrics.json")
    _write_json(per_task_output_path, per_task_metrics(merged))
    print(
        f"Merged {merged['overall']['tasks']} tasks and {merged['overall']['episodes']} episodes "
        f"from {args.num_shards} shards into {output_path.resolve()}"
    )
    print(f"Wrote per-task metrics to {per_task_output_path.resolve()}")


if __name__ == "__main__":
    main()
