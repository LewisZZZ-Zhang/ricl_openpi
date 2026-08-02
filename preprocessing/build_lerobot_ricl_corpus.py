"""Build a RICL corpus for LeRobot v2.1 split-action datasets.

The source dataset stores proprioception/actions in parquet and camera streams
in MP4 files. This script materializes resized frame arrays, 16-D action chunks,
an episode-level train/context split, and exact DINO nearest neighbours from
train query frames into the context bank.
"""

from __future__ import annotations

import argparse
from concurrent import futures
import json
from pathlib import Path
import time
from typing import Any

import av
import h5py
import numpy as np
from openpi_client.image_tools import resize_with_pad
import pyarrow.parquet as pq
import torch

from openpi.policies.utils import embed_with_batches
from openpi.policies.utils import embedding_dim
from openpi.policies.utils import load_dinov2
from openpi.shared import normalize

DEFAULT_DATASET_ID = "lerobot_split_action"
DEFAULT_TASK_PROMPT = "lerobot task"
STATE_LAYOUT = [f"lj{i}" for i in range(7)] + [f"rj{i}" for i in range(7)]
ACTION_LAYOUT = (
    [f"left_ee_{name}" for name in ("qw", "qx", "qy", "qz", "x", "y", "z")]
    + [f"right_ee_{name}" for name in ("qw", "qx", "qy", "qz", "x", "y", "z")]
    + ["left_gripper", "right_gripper"]
)
IMAGE_COLUMN_TO_OUTPUT = {
    "observation.images.zed": "top_image",
    "observation.images.fish0": "wrist_image",
    "observation.images.fish1": "right_image",
}


def _log(message: str) -> None:
    print(message, flush=True)


def _should_log(index: int, total: int, log_every: int) -> bool:
    return index == 1 or index == total or (log_every > 0 and index % log_every == 0)


def _log_progress(
    stage: str,
    *,
    index: int,
    total: int,
    episode_index: int,
    started_at: float,
    extra: str = "",
) -> None:
    elapsed = time.monotonic() - started_at
    suffix = f", {extra}" if extra else ""
    _log(f"[{stage}] {index}/{total} episode={episode_index} elapsed={elapsed:.1f}s{suffix}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_tasks(dataset_root: Path) -> dict[int, str]:
    tasks = {}
    for row in _read_jsonl(dataset_root / "meta" / "tasks.jsonl"):
        tasks[int(row["task_index"])] = str(row["task"])
    return tasks


def _episode_path(dataset_root: Path, episode_index: int) -> Path:
    chunk = episode_index // 100
    return dataset_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def _video_path(dataset_root: Path, image_column: str, episode_index: int) -> Path:
    chunk = episode_index // 100
    return dataset_root / "videos" / f"chunk-{chunk:03d}" / image_column / f"episode_{episode_index:06d}.mp4"


def _decode_video(path: Path) -> np.ndarray:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return np.stack(frames, axis=0).astype(np.uint8, copy=False)


def _load_episode_table(path: Path) -> dict[str, Any]:
    table = pq.read_table(path)
    return table.to_pydict()


def _stack_column(table: dict[str, Any], key: str, dtype: np.dtype = np.float32) -> np.ndarray:
    return np.asarray(table[key], dtype=dtype)


def _stack_actions(table: dict[str, Any]) -> np.ndarray:
    """Return the 16-D action used for this LeRobot RICL corpus.

    The source action also contains action.base_vel[3] and action.lift_cmd[1].
    Those four dimensions are intentionally dropped.
    """

    return np.concatenate(
        [
            _stack_column(table, "action.left_ee"),
            _stack_column(table, "action.right_ee"),
            _stack_column(table, "action.gripper"),
        ],
        axis=-1,
    ).astype(np.float32, copy=False)


def _stack_state(table: dict[str, Any]) -> np.ndarray:
    """Return the 14-D proprioceptive state used for this LeRobot RICL corpus.

    The source observation.state contains left joints, right joints, and lift.
    The lift dimension is intentionally dropped.
    """

    state = _stack_column(table, "observation.state")
    return state[:, : len(STATE_LAYOUT)].astype(np.float32, copy=False)


def _valid_length(length: int, action_horizon: int) -> int:
    valid = length - action_horizon + 1
    if valid <= 0:
        raise ValueError(f"Episode length {length} is shorter than action horizon {action_horizon}")
    return valid


def _processed_episode_rel_path(episode_index: int) -> Path:
    return Path("episodes") / f"episode_{episode_index:06d}.h5"


def _load_existing_processed_episode(
    *,
    output_dir: Path,
    episode_index: int,
    action_horizon: int,
    image_size: int,
    success: str,
) -> dict[str, Any] | None:
    rel_path = _processed_episode_rel_path(episode_index)
    path = output_dir / rel_path
    if not path.exists():
        return None

    try:
        with h5py.File(path, "r") as episode:
            length = int(episode["state"].shape[0])
            if episode["state"].shape != (length, len(STATE_LAYOUT)):
                return None
            if episode["actions"].shape != (length, len(ACTION_LAYOUT)):
                return None
            for image_key in IMAGE_COLUMN_TO_OUTPUT.values():
                if episode[image_key].shape != (length, image_size, image_size, 3):
                    return None
            valid_query_length = _valid_length(length, action_horizon)
    except (OSError, KeyError, ValueError):
        return None

    return {
        "episode_index": episode_index,
        "path": str(rel_path),
        "length": length,
        "valid_query_length": valid_query_length,
        "success": success,
    }


def _split_episodes(
    episodes: list[dict[str, Any]],
    *,
    success_only: bool,
    train_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("--train-fraction must be between 0 and 1")

    selected = [
        int(row["episode_index"])
        for row in episodes
        if not success_only or str(row.get("success", "success")) == "success"
    ]
    dropped = sorted({int(row["episode_index"]) for row in episodes}.difference(selected))
    if len(selected) < 2:
        raise ValueError("Need at least two selected episodes for a train/context split")

    rng = np.random.default_rng(seed)
    shuffled = np.asarray(sorted(selected), dtype=np.int32)
    rng.shuffle(shuffled)
    context_count = int(np.ceil(len(shuffled) * (1.0 - train_fraction)))
    context_count = min(max(context_count, 1), len(shuffled) - 1)
    context = sorted(int(x) for x in shuffled[:context_count])
    train = sorted(int(x) for x in shuffled[context_count:])
    return train, context, dropped


def _save_processed_episode(
    *,
    dataset_root: Path,
    output_dir: Path,
    episode_index: int,
    action_horizon: int,
    image_size: int,
    tasks: dict[int, str],
    fallback_prompt: str,
    success: str,
) -> dict[str, Any]:
    existing = _load_existing_processed_episode(
        output_dir=output_dir,
        episode_index=episode_index,
        action_horizon=action_horizon,
        image_size=image_size,
        success=success,
    )
    if existing is not None:
        return existing

    table = _load_episode_table(_episode_path(dataset_root, episode_index))
    task_indices = table.get("task_index") or []
    prompt = tasks.get(int(task_indices[0]), fallback_prompt) if task_indices else fallback_prompt
    states = _stack_state(table)
    actions = _stack_actions(table)
    length = int(states.shape[0])
    if actions.shape != (length, len(ACTION_LAYOUT)):
        raise ValueError(f"Unexpected action shape for episode {episode_index}: {actions.shape}")
    if states.shape != (length, len(STATE_LAYOUT)):
        raise ValueError(f"Unexpected state shape for episode {episode_index}: {states.shape}")

    data: dict[str, Any] = {
        "state": states,
        "actions": actions,
    }
    for image_column, output_key in IMAGE_COLUMN_TO_OUTPUT.items():
        frames = _decode_video(_video_path(dataset_root, image_column, episode_index))
        if len(frames) != length:
            raise ValueError(
                f"{image_column} episode {episode_index} has {len(frames)} frames, expected {length}"
            )
        data[output_key] = resize_with_pad(frames, image_size, image_size).astype(np.uint8, copy=False)

    episode_dir = output_dir / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)
    rel_path = _processed_episode_rel_path(episode_index)
    with h5py.File(output_dir / rel_path, "w") as episode:
        episode.attrs["prompt"] = prompt
        episode.attrs["success"] = success
        episode.create_dataset("state", data=data["state"])
        episode.create_dataset("actions", data=data["actions"])
        for image_key in IMAGE_COLUMN_TO_OUTPUT.values():
            frames = data[image_key]
            episode.create_dataset(
                image_key,
                data=frames,
                chunks=(1, *frames.shape[1:]),
                compression="lzf",
            )
    return {
        "episode_index": episode_index,
        "path": str(rel_path),
        "length": length,
        "valid_query_length": _valid_length(length, action_horizon),
        "success": success,
    }


def _load_episode(path: Path, *keys: str) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as episode:
        return {key: np.asarray(episode[key]) for key in keys}


def _materialize_records(
    *,
    split_name: str,
    dataset_root: Path,
    output_dir: Path,
    episode_indices: list[int],
    action_horizon: int,
    image_size: int,
    tasks: dict[int, str],
    fallback_prompt: str,
    success_by_episode: dict[int, str],
    log_every: int,
    episode_workers: int,
    skip_invalid_episodes: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _log(f"[episodes:{split_name}] start episodes={len(episode_indices)} workers={episode_workers}")
    started_at = time.monotonic()
    records: list[dict[str, Any] | None] = [None] * len(episode_indices)
    skipped: list[dict[str, Any]] = []
    frame_count = 0

    def _skip_episode(episode_index: int, exc: BaseException) -> None:
        if not skip_invalid_episodes:
            raise exc
        error = f"{type(exc).__name__}: {exc}"
        skipped.append({"split": split_name, "episode_index": int(episode_index), "error": error})
        _log(f"[episodes:{split_name}] skip episode={episode_index}: {error}")

    if episode_workers <= 1:
        for index, episode_index in enumerate(episode_indices, start=1):
            try:
                record = _save_processed_episode(
                    dataset_root=dataset_root,
                    output_dir=output_dir,
                    episode_index=episode_index,
                    action_horizon=action_horizon,
                    image_size=image_size,
                    tasks=tasks,
                    fallback_prompt=fallback_prompt,
                    success=success_by_episode[episode_index],
                )
            except Exception as exc:
                _skip_episode(episode_index, exc)
            else:
                records[index - 1] = record
                frame_count += int(record["length"])
            if _should_log(index, len(episode_indices), log_every):
                _log_progress(
                    f"episodes:{split_name}",
                    index=index,
                    total=len(episode_indices),
                    episode_index=episode_index,
                    started_at=started_at,
                    extra=f"frames={frame_count}, skipped={len(skipped)}",
                )
    else:
        processed = 0
        with futures.ProcessPoolExecutor(max_workers=episode_workers) as executor:
            future_to_index = {
                executor.submit(
                    _save_processed_episode,
                    dataset_root=dataset_root,
                    output_dir=output_dir,
                    episode_index=episode_index,
                    action_horizon=action_horizon,
                    image_size=image_size,
                    tasks=tasks,
                    fallback_prompt=fallback_prompt,
                    success=success_by_episode[episode_index],
                ): (index, int(episode_index))
                for index, episode_index in enumerate(episode_indices)
            }
            for future in futures.as_completed(future_to_index):
                index, episode_index = future_to_index[future]
                processed += 1
                try:
                    record = future.result()
                except Exception as exc:
                    _skip_episode(episode_index, exc)
                    logged_episode_index = episode_index
                else:
                    records[index] = record
                    frame_count += int(record["length"])
                    logged_episode_index = int(record["episode_index"])
                if _should_log(processed, len(episode_indices), log_every):
                    _log_progress(
                        f"episodes:{split_name}",
                        index=processed,
                        total=len(episode_indices),
                        episode_index=logged_episode_index,
                        started_at=started_at,
                        extra=f"frames={frame_count}, skipped={len(skipped)}",
                    )

    materialized = [record for record in records if record is not None]
    if not materialized:
        raise RuntimeError(f"Materialized 0 {split_name} episodes from {len(episode_indices)} requested")
    if len(materialized) != len(episode_indices) and not skip_invalid_episodes:
        raise RuntimeError(f"Materialized {len(materialized)} episodes, expected {len(episode_indices)}")
    _log(f"[episodes:{split_name}] done episodes={len(materialized)} skipped={len(skipped)} frames={frame_count}")
    return materialized, skipped


def _collect_context_bank(
    output_dir: Path,
    episode_records: list[dict[str, Any]],
    *,
    retrieval_image: str,
    action_horizon: int,
    dinov2: Any,
    embedding_type: str,
    embedding_batch_size: int,
    log_every: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _log(f"[context-embeddings] start episodes={len(episode_records)}")
    started_at = time.monotonic()
    embeddings = []
    episode_indices = []
    step_indices = []
    frame_count = 0
    for index, record in enumerate(episode_records, start=1):
        episode = _load_episode(output_dir / record["path"], "state", retrieval_image)
        valid = _valid_length(int(episode["state"].shape[0]), action_horizon)
        images = np.asarray(episode[retrieval_image][:valid], dtype=np.uint8)
        episode_embeddings = embed_with_batches(
            images,
            dinov2,
            batch_size=embedding_batch_size,
            embedding_type=embedding_type,
        )
        embeddings.append(np.asarray(episode_embeddings, dtype=np.float32))
        episode_indices.append(np.full(valid, int(record["episode_index"]), dtype=np.int32))
        step_indices.append(np.arange(valid, dtype=np.int32))
        frame_count += valid
        if _should_log(index, len(episode_records), log_every):
            _log_progress(
                "context-embeddings",
                index=index,
                total=len(episode_records),
                episode_index=int(record["episode_index"]),
                started_at=started_at,
                extra=f"frames={frame_count}",
            )

    result = (
        np.concatenate(embeddings, axis=0),
        np.concatenate(episode_indices, axis=0),
        np.concatenate(step_indices, axis=0),
    )
    _log(f"[context-embeddings] done frames={frame_count} embedding_shape={result[0].shape}")
    return result


def _neighbors(query_embeddings: np.ndarray, context_embeddings: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if len(context_embeddings) < k:
        raise ValueError(f"Need at least {k} context frames, found {len(context_embeddings)}")
    query_norm = np.sum(query_embeddings**2, axis=1, keepdims=True)
    context_norm = np.sum(context_embeddings**2, axis=1)[None, :]
    squared_distances = np.maximum(query_norm + context_norm - 2 * query_embeddings @ context_embeddings.T, 0.0)
    nearest = np.argpartition(squared_distances, k - 1, axis=1)[:, :k]
    nearest_distances = np.take_along_axis(squared_distances, nearest, axis=1)
    order = np.argsort(nearest_distances, axis=1, kind="stable")
    nearest = np.take_along_axis(nearest, order, axis=1).astype(np.int32)

    retrieved_embeddings = context_embeddings[nearest]
    first_embeddings = retrieved_embeddings[:, :1]
    relative = np.zeros((len(query_embeddings), k + 1), dtype=np.float32)
    if k > 1:
        relative[:, 1:k] = np.linalg.norm(retrieved_embeddings[:, 1:] - first_embeddings, axis=2)
    relative[:, -1] = np.linalg.norm(query_embeddings - first_embeddings[:, 0], axis=1)
    return nearest, relative


def _neighbors_torch(
    query_embeddings: np.ndarray,
    context_embeddings: torch.Tensor,
    context_norm: torch.Tensor,
    *,
    k: int,
    query_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(context_embeddings) < k:
        raise ValueError(f"Need at least {k} context frames, found {len(context_embeddings)}")
    if query_batch_size <= 0:
        raise ValueError("--neighbor-query-batch-size must be positive")

    nearest_chunks = []
    relative_chunks = []
    context_t = context_embeddings.t()
    device = context_embeddings.device
    for start in range(0, len(query_embeddings), query_batch_size):
        query = torch.as_tensor(query_embeddings[start : start + query_batch_size], device=device, dtype=torch.float32)
        query_norm = torch.sum(query * query, dim=1, keepdim=True)
        squared_distances = torch.clamp(query_norm + context_norm - 2 * query @ context_t, min=0.0)
        _, nearest = torch.topk(squared_distances, k=k, dim=1, largest=False, sorted=True)

        retrieved_embeddings = context_embeddings[nearest]
        first_embeddings = retrieved_embeddings[:, :1]
        relative = torch.zeros((len(query), k + 1), device=device, dtype=torch.float32)
        if k > 1:
            relative[:, 1:k] = torch.linalg.norm(retrieved_embeddings[:, 1:] - first_embeddings, dim=2)
        relative[:, -1] = torch.linalg.norm(query - first_embeddings[:, 0], dim=1)

        nearest_chunks.append(nearest.cpu().numpy().astype(np.int32, copy=False))
        relative_chunks.append(relative.cpu().numpy().astype(np.float32, copy=False))

    return np.concatenate(nearest_chunks, axis=0), np.concatenate(relative_chunks, axis=0)


def _build_neighbors(
    output_dir: Path,
    train_records: list[dict[str, Any]],
    *,
    retrieval_image: str,
    action_horizon: int,
    num_retrieved: int,
    dinov2: Any,
    embedding_type: str,
    embedding_batch_size: int,
    context_embeddings: np.ndarray,
    log_every: int,
    neighbor_backend: str,
    neighbor_query_batch_size: int,
) -> float:
    if neighbor_backend == "auto":
        neighbor_backend = "torch" if torch.cuda.is_available() else "numpy"
    if neighbor_backend not in {"numpy", "torch"}:
        raise ValueError("--neighbor-backend must be auto, numpy, or torch")

    context_embeddings_t = None
    context_norm_t = None
    if neighbor_backend == "torch":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        context_embeddings_t = torch.as_tensor(context_embeddings, device=device, dtype=torch.float32)
        context_norm_t = torch.sum(context_embeddings_t * context_embeddings_t, dim=1)[None, :]
        _log(
            f"[neighbors] torch backend device={device} "
            f"query_batch_size={neighbor_query_batch_size} context_embeddings={tuple(context_embeddings_t.shape)}"
        )

    _log(
        f"[neighbors] start train_episodes={len(train_records)} "
        f"context_embeddings={context_embeddings.shape} backend={neighbor_backend}"
    )
    started_at = time.monotonic()
    neighbors_dir = output_dir / "neighbors"
    neighbors_dir.mkdir(parents=True, exist_ok=True)
    max_distance = 0.0
    query_episode_indices = []
    query_step_indices = []
    frame_count = 0
    for index, record in enumerate(train_records, start=1):
        episode = _load_episode(output_dir / record["path"], "state", retrieval_image)
        valid = _valid_length(int(episode["state"].shape[0]), action_horizon)
        query_embeddings = embed_with_batches(
            np.asarray(episode[retrieval_image][:valid], dtype=np.uint8),
            dinov2,
            batch_size=embedding_batch_size,
            embedding_type=embedding_type,
        )
        query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
        if neighbor_backend == "torch":
            assert context_embeddings_t is not None and context_norm_t is not None
            retrieved_bank_indices, relative_distances = _neighbors_torch(
                query_embeddings,
                context_embeddings_t,
                context_norm_t,
                k=num_retrieved,
                query_batch_size=neighbor_query_batch_size,
            )
        else:
            retrieved_bank_indices, relative_distances = _neighbors(
                query_embeddings,
                context_embeddings,
                num_retrieved,
            )
        max_distance = max(max_distance, float(relative_distances.max()))
        np.savez_compressed(
            neighbors_dir / f"episode_{int(record['episode_index']):06d}.npz",
            query_step_indices=np.arange(valid, dtype=np.int32),
            retrieved_bank_indices=retrieved_bank_indices,
            relative_distances=relative_distances,
        )
        query_episode_indices.append(np.full(valid, int(record["episode_index"]), dtype=np.int32))
        query_step_indices.append(np.arange(valid, dtype=np.int32))
        frame_count += valid
        if _should_log(index, len(train_records), log_every):
            _log_progress(
                "neighbors",
                index=index,
                total=len(train_records),
                episode_index=int(record["episode_index"]),
                started_at=started_at,
                extra=f"query_frames={frame_count}",
            )

    np.savez_compressed(
        output_dir / "query_refs.npz",
        episode_indices=np.concatenate(query_episode_indices, axis=0),
        step_indices=np.concatenate(query_step_indices, axis=0),
    )
    _log(f"[neighbors] done query_frames={frame_count} max_distance={max_distance}")
    return max(max_distance, float(np.finfo(np.float32).eps))


def _write_norm_stats(
    output_dir: Path,
    records: list[dict[str, Any]],
    *,
    num_retrieved: int,
    log_every: int,
) -> None:
    _log(f"[norm-stats] start episodes={len(records)}")
    started_at = time.monotonic()
    state_stats = normalize.RunningStats()
    action_stats = normalize.RunningStats()
    frame_count = 0
    for index, record in enumerate(records, start=1):
        episode = _load_episode(output_dir / record["path"], "state", "actions")
        states = np.asarray(episode["state"], dtype=np.float32)
        actions = np.asarray(episode["actions"], dtype=np.float32)
        state_stats.update(states)
        action_stats.update(actions)
        frame_count += int(states.shape[0])
        if _should_log(index, len(records), log_every):
            _log_progress(
                "norm-stats",
                index=index,
                total=len(records),
                episode_index=int(record["episode_index"]),
                started_at=started_at,
                extra=f"frames={frame_count}",
            )

    base_stats = {"state": state_stats.get_statistics(), "actions": action_stats.get_statistics()}
    ricl_stats = {
        **{f"retrieved_{index}_{name}": stats for index in range(num_retrieved) for name, stats in base_stats.items()},
        **{f"query_{name}": stats for name, stats in base_stats.items()},
    }
    normalize.save(output_dir / "norm_stats", ricl_stats)
    _log(f"[norm-stats] done frames={frame_count}")


def build_corpus(
    *,
    dataset_root: Path,
    output_dir: Path,
    train_fraction: float,
    success_only: bool,
    seed: int,
    action_horizon: int,
    num_retrieved: int,
    image_size: int,
    retrieval_image: str,
    embedding_type: str,
    embedding_batch_size: int,
    dataset_id: str | None,
    task_prompt: str | None,
    log_every: int,
    episode_workers: int,
    skip_invalid_episodes: bool,
    neighbor_backend: str,
    neighbor_query_batch_size: int,
    dry_run: bool,
) -> dict[str, Any]:
    if action_horizon <= 0:
        raise ValueError("--action-horizon must be positive")
    if num_retrieved <= 0:
        raise ValueError("--num-retrieved must be positive")
    if retrieval_image not in IMAGE_COLUMN_TO_OUTPUT.values():
        raise ValueError(f"--retrieval-image must be one of {sorted(IMAGE_COLUMN_TO_OUTPUT.values())}")
    embedding_dim(embedding_type)

    dataset_root = dataset_root.resolve()
    output_dir = output_dir.resolve()
    resolved_dataset_id = dataset_id or dataset_root.name or DEFAULT_DATASET_ID
    episodes = _read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    tasks = _load_tasks(dataset_root)
    fallback_prompt = task_prompt or next(iter(tasks.values()), DEFAULT_TASK_PROMPT)
    train_indices, context_indices, dropped_indices = _split_episodes(
        episodes,
        success_only=success_only,
        train_fraction=train_fraction,
        seed=seed,
    )
    _log(
        "[split] "
        f"dataset={resolved_dataset_id} train={len(train_indices)} context={len(context_indices)} "
        f"dropped={len(dropped_indices)} success_only={success_only} seed={seed}"
    )

    if dry_run:
        metadata = {
            "dataset": resolved_dataset_id,
            "dataset_root": str(dataset_root),
            "train_episode_indices": train_indices,
            "context_episode_indices": context_indices,
            "dropped_episode_indices": dropped_indices,
            "action_dim": len(ACTION_LAYOUT),
            "state_dim": len(STATE_LAYOUT),
            "action_horizon": action_horizon,
            "num_retrieved": num_retrieved,
            "task_prompt": fallback_prompt,
            "skip_invalid_episodes": skip_invalid_episodes,
        }
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return metadata

    output_dir.mkdir(parents=True, exist_ok=True)
    success_by_episode = {int(row["episode_index"]): str(row.get("success", "success")) for row in episodes}
    train_records, skipped_train_episodes = _materialize_records(
        split_name="train",
        dataset_root=dataset_root,
        output_dir=output_dir,
        episode_indices=train_indices,
        action_horizon=action_horizon,
        image_size=image_size,
        tasks=tasks,
        fallback_prompt=fallback_prompt,
        success_by_episode=success_by_episode,
        log_every=log_every,
        episode_workers=episode_workers,
        skip_invalid_episodes=skip_invalid_episodes,
    )
    context_records, skipped_context_episodes = _materialize_records(
        split_name="context",
        dataset_root=dataset_root,
        output_dir=output_dir,
        episode_indices=context_indices,
        action_horizon=action_horizon,
        image_size=image_size,
        tasks=tasks,
        fallback_prompt=fallback_prompt,
        success_by_episode=success_by_episode,
        log_every=log_every,
        episode_workers=episode_workers,
        skip_invalid_episodes=skip_invalid_episodes,
    )
    skipped_episodes = skipped_train_episodes + skipped_context_episodes
    skipped_episode_indices = sorted(int(row["episode_index"]) for row in skipped_episodes)
    all_dropped_episode_indices = sorted(set(dropped_indices).union(skipped_episode_indices))
    actual_train_indices = [int(row["episode_index"]) for row in train_records]
    actual_context_indices = [int(row["episode_index"]) for row in context_records]

    _log("[dinov2] loading model")
    dinov2 = load_dinov2()
    _log("[dinov2] model loaded")
    context_embeddings, context_episode_indices, context_step_indices = _collect_context_bank(
        output_dir,
        context_records,
        retrieval_image=retrieval_image,
        action_horizon=action_horizon,
        dinov2=dinov2,
        embedding_type=embedding_type,
        embedding_batch_size=embedding_batch_size,
        log_every=log_every,
    )
    _log("[context-embeddings] writing context_embeddings.npy and context_refs.npz")
    np.save(output_dir / "context_embeddings.npy", context_embeddings)
    np.savez_compressed(
        output_dir / "context_refs.npz",
        episode_indices=context_episode_indices,
        step_indices=context_step_indices,
    )
    max_distance = _build_neighbors(
        output_dir,
        train_records,
        retrieval_image=retrieval_image,
        action_horizon=action_horizon,
        num_retrieved=num_retrieved,
        dinov2=dinov2,
        embedding_type=embedding_type,
        embedding_batch_size=embedding_batch_size,
        context_embeddings=context_embeddings,
        log_every=log_every,
        neighbor_backend=neighbor_backend,
        neighbor_query_batch_size=neighbor_query_batch_size,
    )
    _write_norm_stats(output_dir, train_records + context_records, num_retrieved=num_retrieved, log_every=log_every)

    metadata = {
        "format_version": 2,
        "dataset": resolved_dataset_id,
        "dataset_root": str(dataset_root),
        "source_format": "LeRobot v2.1 parquet+mp4",
        "task_prompt": fallback_prompt,
        "train_fraction": train_fraction,
        "success_only": success_only,
        "seed": seed,
        "skip_invalid_episodes": skip_invalid_episodes,
        "action_horizon": action_horizon,
        "num_retrieved": num_retrieved,
        "image_size": image_size,
        "retrieval_image": retrieval_image,
        "embedding_type": embedding_type,
        "embedding_dim": embedding_dim(embedding_type),
        "max_distance": max_distance,
        "state_dim": len(STATE_LAYOUT),
        "state_layout": STATE_LAYOUT,
        "dropped_state_fields": {"observation.state": ["lift"]},
        "action_dim": len(ACTION_LAYOUT),
        "action_layout": ACTION_LAYOUT,
        "dropped_action_fields": {
            "action.base_vel": ["vx", "vy", "omega"],
            "action.lift_cmd": ["lift_cmd"],
        },
        "image_layout": {
            "top_image": "observation.images.zed",
            "wrist_image": "observation.images.fish0",
            "right_image": "observation.images.fish1",
        },
        "train_episodes": train_records,
        "context_episodes": context_records,
        "requested_train_episode_indices": train_indices,
        "requested_context_episode_indices": context_indices,
        "train_episode_indices": actual_train_indices,
        "context_episode_indices": actual_context_indices,
        "success_filter_dropped_episode_indices": dropped_indices,
        "skipped_episode_indices": skipped_episode_indices,
        "skipped_episodes": skipped_episodes,
        "dropped_episode_indices": all_dropped_episode_indices,
        "context_embeddings_path": "context_embeddings.npy",
        "context_refs_path": "context_refs.npz",
        "query_refs_path": "query_refs.npz",
        "neighbors_dir": "neighbors",
        "norm_stats_dir": "norm_stats",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "split.json").write_text(
        json.dumps(
            {
                "requested_train_episode_indices": train_indices,
                "requested_context_episode_indices": context_indices,
                "train_episode_indices": actual_train_indices,
                "context_episode_indices": actual_context_indices,
                "success_filter_dropped_episode_indices": dropped_indices,
                "skipped_episode_indices": skipped_episode_indices,
                "skipped_episodes": skipped_episodes,
                "dropped_episode_indices": all_dropped_episode_indices,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _log(f"[metadata] wrote metadata.json and split.json to {output_dir}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a RICL corpus for a LeRobot v2.1 split-action dataset.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--dataset-id", default=None, help="Dataset id recorded in metadata; defaults to dataset root name.")
    parser.add_argument("--task-prompt", default=None, help="Fallback prompt when an episode has no task_index.")
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--include-failures", action="store_true", help="Use fail episodes as well as success episodes.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--num-retrieved", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--retrieval-image", choices=("top_image", "wrist_image", "right_image"), default="wrist_image")
    parser.add_argument("--embedding-type", choices=("CLS", "AVG", "16PATCHES", "64PATCHES"), default="CLS")
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=10, help="Print progress every N episodes; use 1 for every episode.")
    parser.add_argument(
        "--episode-workers",
        type=int,
        default=1,
        help="Parallel workers for MP4 decode/resize/HDF5 episode materialization.",
    )
    parser.add_argument(
        "--fail-on-invalid-episode",
        action="store_true",
        help="Abort when an episode cannot be materialized instead of skipping it and recording it in metadata.",
    )
    parser.add_argument(
        "--neighbor-backend",
        choices=("auto", "numpy", "torch"),
        default="auto",
        help="Backend for exact nearest-neighbor search over DINO embeddings.",
    )
    parser.add_argument(
        "--neighbor-query-batch-size",
        type=int,
        default=2048,
        help="Query frames per exact-neighbor GPU/torch distance batch.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print split/shape metadata without decoding videos or DINO.")
    args = parser.parse_args()

    metadata = build_corpus(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        train_fraction=args.train_fraction,
        success_only=not args.include_failures,
        seed=args.seed,
        action_horizon=args.action_horizon,
        num_retrieved=args.num_retrieved,
        image_size=args.image_size,
        retrieval_image=args.retrieval_image,
        embedding_type=args.embedding_type,
        embedding_batch_size=args.embedding_batch_size,
        dataset_id=args.dataset_id,
        task_prompt=args.task_prompt,
        log_every=args.log_every,
        episode_workers=args.episode_workers,
        skip_invalid_episodes=not args.fail_on_invalid_episode,
        neighbor_backend=args.neighbor_backend,
        neighbor_query_batch_size=args.neighbor_query_batch_size,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        print(
            f"Wrote LeRobot RICL corpus to {args.output_dir} "
            f"({len(metadata['train_episodes'])} train episodes, {len(metadata['context_episodes'])} context episodes)."
        )


if __name__ == "__main__":
    main()
