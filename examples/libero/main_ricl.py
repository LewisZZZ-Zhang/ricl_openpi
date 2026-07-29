"""Evaluate a task-scoped RICL policy on LIBERO-90, LIBERO-10, or both."""

from __future__ import annotations

import collections
import dataclasses
import datetime
import json
import logging
import math
import pathlib
from typing import Literal

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as websocket_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
MAX_STEPS = {"libero_90": 400, "libero_10": 520}
TaskSplit = Literal["unseen", "train", "all"]


@dataclasses.dataclass(frozen=True)
class CorpusTask:
    task_id: int
    is_train: bool


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    corpus_dir: str = tyro.MISSING
    resize_size: int = 224
    replan_steps: int = 5
    task_suite_name: str = "libero_100"  # libero_90, libero_10, or libero_100
    task_split: TaskSplit = "unseen"
    num_task_shards: int = 1
    task_shard_index: int = 0
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    metrics_path: str = "data/libero/metrics_ricl.json"
    checkpoint_dir: str = ""
    video_out_path: str = "data/libero/videos_ricl"
    save_videos: bool = True
    seed: int = 7


def _tasks_from_corpus(corpus_dir: str) -> dict[tuple[str, str], CorpusTask]:
    metadata_path = pathlib.Path(corpus_dir).expanduser() / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    tasks: dict[tuple[str, str], CorpusTask] = {}
    for task in metadata["tasks"]:
        key = (str(task["suite"]), str(task["task_name"]))
        if key in tasks:
            raise ValueError(f"Duplicate corpus task: {key[0]}/{key[1]}")
        if not isinstance(task.get("is_train"), bool):
            raise ValueError(f"Corpus task {key[0]}/{key[1]} has no boolean is_train field")
        tasks[key] = CorpusTask(task_id=int(task["task_id"]), is_train=bool(task["is_train"]))
    return tasks


def _matches_split(task: CorpusTask, task_split: TaskSplit) -> bool:
    if task_split == "all":
        return True
    return task.is_train if task_split == "train" else not task.is_train


def _timestamp() -> str:
    # The LIBERO client environment uses Python 3.8, which predates datetime.UTC.
    return datetime.datetime.now(datetime.timezone.utc).isoformat()  # noqa: UP017


def _write_metrics(metrics_path: str, metrics: dict) -> None:
    path = pathlib.Path(metrics_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics["updated_at"] = _timestamp()
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _get_libero_env(task, resolution: int, seed: int):
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task.language


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.array(quat, copy=True)
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(denominator, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / denominator


def _suite_names(task_suite_name: str) -> tuple[str, ...]:
    if task_suite_name == "libero_100":
        return ("libero_90", "libero_10")
    if task_suite_name in MAX_STEPS:
        return (task_suite_name,)
    raise ValueError("task_suite_name must be libero_90, libero_10, or libero_100")


def eval_libero(args: Args) -> None:
    if args.num_trials_per_task <= 0:
        raise ValueError("num_trials_per_task must be positive")
    if args.num_task_shards <= 0:
        raise ValueError("num_task_shards must be positive")
    if not 0 <= args.task_shard_index < args.num_task_shards:
        raise ValueError(
            f"task_shard_index must be in [0, {args.num_task_shards}), got {args.task_shard_index}"
        )
    np.random.seed(args.seed)
    if args.save_videos:
        pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    client = websocket_policy.WebsocketClientPolicy(args.host, args.port)
    corpus_tasks = _tasks_from_corpus(args.corpus_dir)
    benchmark_dict = benchmark.get_benchmark_dict()

    metrics = {
        "status": "running",
        "started_at": _timestamp(),
        "updated_at": _timestamp(),
        "config": {
            "checkpoint_dir": args.checkpoint_dir or None,
            "corpus_dir": str(pathlib.Path(args.corpus_dir).expanduser().resolve()),
            "task_suite_name": args.task_suite_name,
            "task_split": args.task_split,
            "num_task_shards": args.num_task_shards,
            "task_shard_index": args.task_shard_index,
            "task_shard_strategy": "global_round_robin",
            "num_trials_per_task": args.num_trials_per_task,
            "num_steps_wait": args.num_steps_wait,
            "replan_steps": args.replan_steps,
            "seed": args.seed,
        },
        "suites": {},
        "overall": {
            "tasks": 0,
            "episodes": 0,
            "successes": 0,
            "success_rate": 0.0,
        },
    }
    _write_metrics(args.metrics_path, metrics)

    total_episodes = 0
    total_successes = 0
    total_tasks = 0
    matched_task_count = 0
    for suite_name in _suite_names(args.task_suite_name):
        task_suite = benchmark_dict[suite_name]()
        selected_tasks = []
        for task_index in range(task_suite.n_tasks):
            task = task_suite.get_task(task_index)
            corpus_task = corpus_tasks.get((suite_name, task.name))
            if corpus_task is None:
                raise KeyError(f"No context retrieval bank for {suite_name}/{task.name}")
            if _matches_split(corpus_task, args.task_split):
                task_global_index = matched_task_count
                matched_task_count += 1
                if task_global_index % args.num_task_shards == args.task_shard_index:
                    selected_tasks.append((task_global_index, task_index, task, corpus_task))

        suite_episodes = 0
        suite_successes = 0
        suite_metrics = {
            "tasks": [],
            "num_tasks": len(selected_tasks),
            "episodes": 0,
            "successes": 0,
            "success_rate": 0.0,
        }
        metrics["suites"][suite_name] = suite_metrics
        _write_metrics(args.metrics_path, metrics)
        logging.info(
            "Evaluating %s split=%s shard=%d/%d (%d tasks, %d trials/task)",
            suite_name,
            args.task_split,
            args.task_shard_index,
            args.num_task_shards,
            len(selected_tasks),
            args.num_trials_per_task,
        )
        for task_global_index, task_index, task, corpus_task in tqdm.tqdm(selected_tasks, desc=suite_name):
            initial_states = task_suite.get_task_init_states(task_index)
            if len(initial_states) < args.num_trials_per_task:
                raise ValueError(
                    f"{suite_name}/{task.name} has {len(initial_states)} initial states, "
                    f"cannot run {args.num_trials_per_task} trials"
                )
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
            task_successes = 0
            total_tasks += 1
            episode_metrics = []

            try:
                for episode_index in range(args.num_trials_per_task):
                    env.reset()
                    obs = env.set_init_state(initial_states[episode_index])
                    action_plan: collections.deque[np.ndarray] = collections.deque()
                    replay_images: list[np.ndarray] = []
                    done = False
                    timestep = 0
                    rollout_error = None
                    try:
                        while timestep < MAX_STEPS[suite_name] + args.num_steps_wait:
                            if timestep < args.num_steps_wait:
                                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                                timestep += 1
                                continue

                            top_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                            wrist_image = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                            top_image = image_tools.convert_to_uint8(
                                image_tools.resize_with_pad(top_image, args.resize_size, args.resize_size)
                            )
                            wrist_image = image_tools.convert_to_uint8(
                                image_tools.resize_with_pad(wrist_image, args.resize_size, args.resize_size)
                            )
                            if args.save_videos:
                                replay_images.append(top_image)

                            if not action_plan:
                                request = {
                                    "query_top_image": top_image,
                                    "query_wrist_image": wrist_image,
                                    "query_state": np.concatenate(
                                        (
                                            obs["robot0_eef_pos"],
                                            _quat2axisangle(obs["robot0_eef_quat"]),
                                            obs["robot0_gripper_qpos"],
                                        )
                                    ).astype(np.float32),
                                    "query_prompt": str(task_description),
                                    "retrieval_task_id": np.asarray(corpus_task.task_id, dtype=np.int32),
                                }
                                action_chunk = client.infer(request)["actions"]
                                if len(action_chunk) < args.replan_steps:
                                    raise ValueError(
                                        f"RICL returned {len(action_chunk)} actions, need at least {args.replan_steps}"
                                    )
                                action_plan.extend(action_chunk[: args.replan_steps])

                            obs, _, done, _ = env.step(action_plan.popleft().tolist())
                            timestep += 1
                            if done:
                                break
                    except Exception as exc:
                        rollout_error = f"{type(exc).__name__}: {exc}"
                        logging.exception("Failed rollout for %s episode %d", task.name, episode_index)

                    total_episodes += 1
                    suite_episodes += 1
                    total_successes += int(done)
                    suite_successes += int(done)
                    if done:
                        task_successes += 1
                    episode_metrics.append(
                        {
                            "episode_index": episode_index,
                            "success": bool(done),
                            "steps": timestep,
                            "error": rollout_error,
                        }
                    )
                    if args.save_videos and replay_images:
                        video_name = (
                            f"{suite_name}_{task_index:03d}_{episode_index:03d}_"
                            f"{'success' if done else 'failure'}.mp4"
                        )
                        imageio.mimwrite(pathlib.Path(args.video_out_path) / video_name, replay_images, fps=10)
            finally:
                env.close()

            task_metrics = {
                "task_global_index": task_global_index,
                "task_index": task_index,
                "task_id": corpus_task.task_id,
                "task_name": str(task.name),
                "episodes": args.num_trials_per_task,
                "successes": task_successes,
                "success_rate": task_successes / args.num_trials_per_task,
                "episode_results": episode_metrics,
            }
            suite_metrics["tasks"].append(task_metrics)
            suite_metrics["episodes"] = suite_episodes
            suite_metrics["successes"] = suite_successes
            suite_metrics["success_rate"] = suite_successes / suite_episodes
            metrics["overall"] = {
                "tasks": total_tasks,
                "episodes": total_episodes,
                "successes": total_successes,
                "success_rate": total_successes / total_episodes,
            }
            _write_metrics(args.metrics_path, metrics)
            logging.info(
                "%s task %d (%s) success: %.3f (%d/%d)",
                suite_name,
                task_index,
                task.name,
                task_metrics["success_rate"],
                task_successes,
                args.num_trials_per_task,
            )
        if suite_episodes:
            logging.info("%s success: %.3f", suite_name, suite_successes / suite_episodes)
    if not total_episodes:
        raise ValueError(f"No tasks matched task_split={args.task_split!r} in task_suite_name={args.task_suite_name!r}")
    logging.info(
        "Overall LIBERO success: %.3f (%d/%d episodes across %d tasks)",
        total_successes / total_episodes,
        total_successes,
        total_episodes,
        total_tasks,
    )
    metrics["status"] = "completed"
    metrics["completed_at"] = _timestamp()
    _write_metrics(args.metrics_path, metrics)
    logging.info("Saved evaluation metrics to %s", pathlib.Path(args.metrics_path).expanduser().resolve())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_libero(tyro.cli(Args))
