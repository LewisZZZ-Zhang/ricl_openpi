from __future__ import annotations

import json

import merge_ricl_metrics


def _write_shard(tmp_path, shard_index: int, task_indices: list[int]) -> None:
    num_shards = 2
    shard_dir = tmp_path / f"shard_{shard_index:02d}_of_{num_shards:02d}"
    shard_dir.mkdir()
    tasks = [
        {
            "task_global_index": task_index,
            "task_index": task_index,
            "task_id": task_index,
            "task_name": f"task_{task_index}",
            "episodes": 2,
            "successes": 1,
            "success_rate": 0.5,
            "episode_results": [],
        }
        for task_index in task_indices
    ]
    episodes = 2 * len(tasks)
    successes = len(tasks)
    metrics = {
        "status": "completed",
        "started_at": f"2026-01-01T00:00:0{shard_index}+00:00",
        "completed_at": f"2026-01-01T01:00:0{shard_index}+00:00",
        "config": {
            "checkpoint_dir": "/checkpoint",
            "num_task_shards": num_shards,
            "task_shard_index": shard_index,
            "task_shard_strategy": "global_round_robin",
        },
        "suites": {
            "libero_90": {
                "tasks": tasks,
                "num_tasks": len(tasks),
                "episodes": episodes,
                "successes": successes,
                "success_rate": 0.5,
            }
        },
        "overall": {
            "tasks": len(tasks),
            "episodes": episodes,
            "successes": successes,
            "success_rate": 0.5,
        },
    }
    (shard_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


def test_merge_round_robin_shards(tmp_path) -> None:
    _write_shard(tmp_path, 0, [0, 2])
    _write_shard(tmp_path, 1, [1, 3])

    merged = merge_ricl_metrics.merge_metrics(tmp_path, num_shards=2)

    assert merged["status"] == "completed"
    assert merged["config"]["parallel_clients"] == 2
    assert "task_shard_index" not in merged["config"]
    assert merged["overall"] == {
        "tasks": 4,
        "episodes": 8,
        "successes": 4,
        "success_rate": 0.5,
    }
    assert [task["task_global_index"] for task in merged["suites"]["libero_90"]["tasks"]] == [0, 1, 2, 3]

    per_task = merge_ricl_metrics.per_task_metrics(merged)
    assert per_task["overall"] == merged["overall"]
    assert [task["task_global_index"] for task in per_task["tasks"]] == [0, 1, 2, 3]
    assert per_task["tasks"][0] == {
        "suite": "libero_90",
        "task_global_index": 0,
        "task_index": 0,
        "task_id": 0,
        "task_name": "task_0",
        "episodes": 2,
        "successes": 1,
        "success_rate": 0.5,
    }
