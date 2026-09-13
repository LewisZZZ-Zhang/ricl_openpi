"""Build a task-scoped DINO or progress retrieval corpus for RICL on LIBERO-100.

Images, states, and actions continue to live in the original LIBERO HDF5 files,
so the data is not duplicated. Training neighbours are selected exclusively
from a task's context demos; query and context episode sets must therefore be
disjoint. Progress corpora use the same versioned labels as VFE instead of
reconstructing a second definition of ground truth.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from openpi.policies.libero_retrieval import progress_neighbors


PROGRESS_SEMANTICS = "per_demo_relative_progress_v1"


@dataclass(frozen=True)
class TaskSplit:
    task_id: int
    suite: str
    task_name: str
    prompt: str
    context_demo_ids: list[str]
    query_demo_ids: list[str]
    is_train: bool


def _numeric_demo_key(demo_id: str) -> tuple[int, str]:
    suffix = demo_id.removeprefix("demo_")
    return (int(suffix), demo_id) if suffix.isdigit() else (10**9, demo_id)


def _task_map_path(libero_root: Path) -> Path:
    candidates = (
        libero_root / "libero" / "libero" / "benchmark" / "libero_suite_task_map.py",
        libero_root / "libero" / "benchmark" / "libero_suite_task_map.py",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find libero_suite_task_map.py below {libero_root}")


def _load_task_map(libero_root: Path) -> dict[str, list[str]]:
    tree = ast.parse(_task_map_path(libero_root).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "libero_task_map" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("Could not find libero_task_map")


def _prompt_from_task_name(task_name: str) -> str:
    if task_name.startswith("LIBERO_"):
        return task_name.replace("_", " ")
    if task_name.startswith("KITCHEN_") or task_name.startswith("LIVING_") or task_name.startswith("STUDY_"):
        scene_start = task_name.find("SCENE")
        if scene_start >= 0:
            offset = 8 if "SCENE10" in task_name else 7
            return task_name[scene_start + offset :].replace("_", " ").strip()
    return task_name.replace("_", " ").strip()


def _dataset_path(dataset_root: Path, suite: str, task_name: str) -> Path:
    path = dataset_root / suite / f"{task_name}_demo.hdf5"
    if path.exists():
        return path.resolve()
    matches = sorted(dataset_root.rglob(f"{task_name}_demo.hdf5"))
    if matches:
        return matches[0].resolve()
    raise FileNotFoundError(f"Could not find demonstrations for {suite}/{task_name}")


def _read_prompt(h5_file: h5py.File, fallback: str) -> str:
    try:
        problem_info = json.loads(h5_file["data"].attrs["problem_info"])
        return str(problem_info.get("language_instruction", fallback)).strip().strip('"')
    except Exception:
        return fallback


def _default_splits(dataset_root: Path, libero_root: Path, context_demos: int) -> list[TaskSplit]:
    if context_demos <= 0 or context_demos >= 50:
        raise ValueError("--context-demos must be between 1 and 49")
    task_splits: list[TaskSplit] = []
    task_map = _load_task_map(libero_root)
    for suite in ("libero_90", "libero_10"):
        for task_name in task_map[suite]:
            task_id = len(task_splits)
            path = _dataset_path(dataset_root, suite, task_name)
            with h5py.File(path, "r") as h5_file:
                demo_ids = sorted(h5_file["data"].keys(), key=_numeric_demo_key)
                prompt = _read_prompt(h5_file, _prompt_from_task_name(task_name))
            if len(demo_ids) < context_demos + 1:
                raise ValueError(f"{path} has too few demos for a context/query split")
            task_splits.append(
                TaskSplit(
                    task_id=task_id,
                    suite=suite,
                    task_name=task_name,
                    prompt=prompt,
                    context_demo_ids=demo_ids[:context_demos],
                    query_demo_ids=demo_ids[context_demos:],
                    is_train=True,
                )
            )
    if len(task_splits) != 100:
        raise ValueError(f"Expected 100 LIBERO tasks, found {len(task_splits)}")
    return task_splits


def _manifest_splits(manifest_path: Path) -> list[TaskSplit]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits: list[TaskSplit] = []
    for item in payload.get("train_tasks", []):
        splits.append(
            TaskSplit(
                task_id=int(item["task_id"]),
                suite=str(item["suite"]),
                task_name=str(item["task_name"]),
                prompt=str(item["language_instruction"]),
                context_demo_ids=[str(x) for x in item["train_context_demo_ids"]],
                query_demo_ids=[str(x) for x in item["train_query_demo_ids"]],
                is_train=True,
            )
        )
    for item in payload.get("unseen_eval_tasks", []):
        splits.append(
            TaskSplit(
                task_id=int(item["task_id"]),
                suite=str(item["suite"]),
                task_name=str(item["task_name"]),
                prompt=str(item["language_instruction"]),
                context_demo_ids=[str(x) for x in item["eval_context_demo_ids"]],
                query_demo_ids=[str(x) for x in item["eval_query_demo_ids"]],
                is_train=False,
            )
        )
    if not splits:
        raise ValueError(f"No train_tasks or unseen_eval_tasks found in {manifest_path}")
    if len({split.task_id for split in splits}) != len(splits):
        raise ValueError("Task IDs are not unique in the split manifest")
    return splits


def _embed_images(images: np.ndarray, dinov2: Any, embedding_type: str, batch_size: int) -> np.ndarray:
    from openpi.policies.utils import embed_with_batches

    images = np.ascontiguousarray(images[:, ::-1, ::-1])
    embeddings = embed_with_batches(images, dinov2, batch_size=batch_size, embedding_type=embedding_type)
    return np.asarray(embeddings, dtype=np.float32)


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


def _label_path(progress_labels_dir: Path, task_id: int, demo_id: str) -> Path:
    return progress_labels_dir / f"task_{task_id:03d}" / f"{demo_id}.npz"


def _load_gt_progress(
    progress_labels_dir: Path,
    task_id: int,
    demo_id: str,
    expected_length: int,
) -> tuple[np.ndarray, str]:
    path = _label_path(progress_labels_dir, task_id, demo_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing VFE progress labels: {path}")
    with np.load(path, allow_pickle=False) as payload:
        values = np.asarray(payload["value_continuous"], dtype=np.float32)
        value_min = float(payload["value_min"])
        value_max = float(payload["value_max"])
        recorded_task_id = int(payload["task_id"])
        recorded_demo_id = str(payload["demo_id"])
        recorded_preset = str(payload["data_split_preset"])
    if recorded_task_id != task_id or recorded_demo_id != demo_id:
        raise ValueError(f"Progress-label identity mismatch in {path}")
    if values.shape != (expected_length,):
        raise ValueError(f"Progress labels in {path} have shape {values.shape}; expected ({expected_length},)")
    if not np.isfinite(values).all() or value_max <= value_min:
        raise ValueError(f"Invalid progress labels in {path}")
    progress = (values - value_min) / (value_max - value_min)
    if np.any((progress < -1e-6) | (progress > 1.0 + 1e-6)):
        raise ValueError(f"Progress labels in {path} fall outside their recorded range")
    # Exact endpoints are part of the per-demo relative-progress contract.
    if expected_length == 1:
        endpoints_ok = np.isclose(progress[0], 1.0)
    else:
        endpoints_ok = np.isclose(progress[0], 0.0) and np.isclose(progress[-1], 1.0)
    if not endpoints_ok:
        raise ValueError(
            f"{path} is not {PROGRESS_SEMANTICS}: expected per-demo progress endpoints, "
            f"got first={progress[0]} last={progress[-1]}"
        )
    return np.clip(progress, 0.0, 1.0).astype(np.float32), recorded_preset


def _load_cached_query_progress(
    cache_dir: Path,
    task_id: int,
    demo_id: str,
    expected_length: int,
) -> np.ndarray:
    candidates = (
        cache_dir / f"task_{task_id:03d}" / f"{demo_id}.npz",
        cache_dir / f"task_{task_id:03d}_{demo_id}.npz",
    )
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(f"Missing cached query progress; tried: {', '.join(map(str, candidates))}")
    with np.load(path, allow_pickle=False) as payload:
        if "progress" in payload:
            progress = np.asarray(payload["progress"], dtype=np.float32)
        elif "prediction_progress" in payload:
            progress = np.asarray(payload["prediction_progress"], dtype=np.float32)
        elif "prediction" in payload:
            # Native VFE scalar values use [-1, 0].
            progress = np.asarray(payload["prediction"], dtype=np.float32) + 1.0
        else:
            raise KeyError(f"{path} must contain progress, prediction_progress, or prediction")
        timesteps = np.asarray(payload["timesteps"], dtype=np.int64) if "timesteps" in payload else None
    if progress.shape != (expected_length,):
        raise ValueError(f"Cached progress in {path} has shape {progress.shape}; expected ({expected_length},)")
    if timesteps is not None and not np.array_equal(timesteps, np.arange(expected_length, dtype=np.int64)):
        raise ValueError(f"Cached progress in {path} does not contain one prediction for every frame")
    if not np.isfinite(progress).all():
        raise ValueError(f"Cached progress in {path} contains non-finite values")
    return np.clip(progress, 0.0, 1.0).astype(np.float32)


def build_corpus(
    *,
    dataset_root: Path,
    libero_root: Path,
    output_dir: Path,
    split_manifest: Path | None,
    num_retrieved: int,
    action_horizon: int,
    embedding_type: str,
    embedding_batch_size: int,
    context_demos: int,
    retrieval_backend: str = "dino",
    progress_labels_dir: Path | None = None,
    query_progress_source: str = "gt",
    query_progress_cache: Path | None = None,
    retrieval_seed: int = 0,
) -> dict[str, Any]:
    if num_retrieved <= 0:
        raise ValueError("num_retrieved must be positive")
    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive")
    if retrieval_backend not in {"dino", "progress"}:
        raise ValueError("retrieval_backend must be 'dino' or 'progress'")
    if query_progress_source not in {"gt", "cache"}:
        raise ValueError("query_progress_source must be 'gt' or 'cache'")
    if retrieval_backend == "progress":
        if progress_labels_dir is None:
            raise ValueError("--progress-labels-dir is required for progress retrieval")
        progress_labels_dir = progress_labels_dir.expanduser().resolve()
        if not progress_labels_dir.is_dir():
            raise FileNotFoundError(f"Progress-label directory not found: {progress_labels_dir}")
        if query_progress_source == "cache":
            if query_progress_cache is None:
                raise ValueError("--query-progress-cache is required when --query-progress-source=cache")
            query_progress_cache = query_progress_cache.expanduser().resolve()
            if not query_progress_cache.is_dir():
                raise FileNotFoundError(f"Query-progress cache not found: {query_progress_cache}")
    elif query_progress_source != "gt" or query_progress_cache is not None:
        raise ValueError("Query-progress source/cache options are only valid for progress retrieval")

    embedding_size: int | None = None
    dinov2: Any | None = None
    progress_rng: np.random.Generator | None = None
    if retrieval_backend == "dino":
        from openpi.policies.utils import embedding_dim, load_dinov2

        embedding_size = embedding_dim(embedding_type)  # validate before beginning a long DINO run
        dinov2 = load_dinov2()
    else:
        progress_rng = np.random.default_rng(retrieval_seed)

    task_splits = (
        _manifest_splits(split_manifest.resolve())
        if split_manifest is not None
        else _default_splits(dataset_root, libero_root, context_demos)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    banks_dir = output_dir / ("embeddings" if retrieval_backend == "dino" else "progress")
    neighbors_dir = output_dir / "neighbors"
    banks_dir.mkdir(exist_ok=True)
    neighbors_dir.mkdir(exist_ok=True)

    metadata_tasks: list[dict[str, Any]] = []
    max_distance = 0.0
    progress_data_split_preset: str | None = None
    for task_position, split in enumerate(task_splits, start=1):
        if set(split.context_demo_ids).intersection(split.query_demo_ids):
            raise ValueError(f"Task {split.task_id} has overlapping context and query demos")
        source_hdf5 = _dataset_path(dataset_root, split.suite, split.task_name)
        with h5py.File(source_hdf5, "r") as h5_file:
            available_demos = set(h5_file["data"].keys())
            requested_demos = set(split.context_demo_ids).union(split.query_demo_ids)
            missing = sorted(requested_demos.difference(available_demos), key=_numeric_demo_key)
            if missing:
                raise KeyError(f"Task {split.task_id} references absent demos: {missing}")

            context_features: list[np.ndarray] = []
            context_demo_indices: list[np.ndarray] = []
            context_step_indices: list[np.ndarray] = []
            for demo_index, demo_id in enumerate(split.context_demo_ids):
                demo_length = len(h5_file[f"data/{demo_id}/actions"])
                if retrieval_backend == "dino":
                    images = h5_file[f"data/{demo_id}/obs/agentview_rgb"][:]
                    features = _embed_images(images, dinov2, embedding_type, embedding_batch_size)
                else:
                    features, label_preset = _load_gt_progress(
                        progress_labels_dir, split.task_id, demo_id, demo_length
                    )
                    if progress_data_split_preset is None:
                        progress_data_split_preset = label_preset
                    elif label_preset != progress_data_split_preset:
                        raise ValueError(
                            f"Mixed VFE data-split presets in progress labels: "
                            f"{progress_data_split_preset!r} and {label_preset!r}"
                        )
                context_features.append(features)
                context_demo_indices.append(np.full(demo_length, demo_index, dtype=np.int32))
                context_step_indices.append(np.arange(demo_length, dtype=np.int32))
            all_context_features = np.concatenate(context_features, axis=0)
            all_context_demo_indices = np.concatenate(context_demo_indices, axis=0)
            all_context_step_indices = np.concatenate(context_step_indices, axis=0)

            task_dir = banks_dir / f"task_{split.task_id:03d}"
            task_dir.mkdir(exist_ok=True)
            bank_path = task_dir / (
                "context_embeddings.npy" if retrieval_backend == "dino" else "context_progress.npy"
            )
            refs_path = task_dir / "context_refs.npz"
            np.save(bank_path, all_context_features.astype(np.float32))
            np.savez_compressed(
                refs_path,
                demo_indices=all_context_demo_indices,
                step_indices=all_context_step_indices,
            )

            neighbors_rel_path: str | None = None
            if split.is_train:
                task_neighbors_dir = neighbors_dir / f"task_{split.task_id:03d}"
                task_neighbors_dir.mkdir(exist_ok=True)
                for demo_id in split.query_demo_ids:
                    demo_length = len(h5_file[f"data/{demo_id}/actions"])
                    if retrieval_backend == "dino":
                        query_images = h5_file[f"data/{demo_id}/obs/agentview_rgb"][:]
                        query_features = _embed_images(query_images, dinov2, embedding_type, embedding_batch_size)
                        neighbor_indices, relative_distances = _neighbors(
                            query_features, all_context_features, num_retrieved
                        )
                    else:
                        if query_progress_source == "gt":
                            query_features, label_preset = _load_gt_progress(
                                progress_labels_dir, split.task_id, demo_id, demo_length
                            )
                            if label_preset != progress_data_split_preset:
                                raise ValueError(
                                    f"Query label preset {label_preset!r} does not match context-label preset "
                                    f"{progress_data_split_preset!r}"
                                )
                        else:
                            query_features = _load_cached_query_progress(
                                query_progress_cache, split.task_id, demo_id, demo_length
                            )
                        neighbor_indices, relative_distances = progress_neighbors(
                            query_features,
                            all_context_features,
                            all_context_demo_indices,
                            num_retrieved,
                            rng=progress_rng,
                        )
                    max_distance = max(max_distance, float(relative_distances.max()))
                    neighbor_payload: dict[str, np.ndarray] = {
                        "retrieved_bank_indices": neighbor_indices,
                        "relative_distances": relative_distances,
                    }
                    if retrieval_backend == "progress":
                        neighbor_payload["query_progress"] = np.asarray(query_features, dtype=np.float32)
                        neighbor_payload["retrieved_progress_distances"] = np.abs(
                            all_context_features[neighbor_indices] - np.asarray(query_features)[:, None]
                        ).astype(np.float32)
                    np.savez_compressed(task_neighbors_dir / f"{demo_id}.npz", **neighbor_payload)
                neighbors_rel_path = str(task_neighbors_dir.relative_to(output_dir))

        task_metadata = {
            "task_id": split.task_id,
            "suite": split.suite,
            "task_name": split.task_name,
            "prompt": split.prompt,
            "source_hdf5": str(source_hdf5),
            "context_demo_ids": split.context_demo_ids,
            "query_demo_ids": split.query_demo_ids,
            "is_train": split.is_train,
            "context_refs_path": str(refs_path.relative_to(output_dir)),
            "neighbors_dir": neighbors_rel_path,
        }
        task_metadata[
            "context_embeddings_path" if retrieval_backend == "dino" else "context_progress_path"
        ] = str(bank_path.relative_to(output_dir))
        metadata_tasks.append(task_metadata)
        print(
            f"[{task_position}/{len(task_splits)}] task={split.task_id} "
            f"context_frames={len(all_context_features)} train={split.is_train} backend={retrieval_backend}"
        )

    metadata = {
        "format_version": 2,
        "dataset": "LIBERO-100",
        "dataset_root": str(dataset_root.resolve()),
        "split_manifest": str(split_manifest.resolve()) if split_manifest is not None else None,
        "retrieval_backend": retrieval_backend,
        "num_retrieved": num_retrieved,
        "action_horizon": action_horizon,
        "max_distance": max(max_distance, np.finfo(np.float32).eps),
        "tasks": metadata_tasks,
    }
    if retrieval_backend == "dino":
        metadata.update(
            {
                "embedding_type": embedding_type,
                "embedding_dim": embedding_size,
                "embedding_storage_dtype": "float32",
            }
        )
    else:
        metadata.update(
            {
                "progress_semantics": PROGRESS_SEMANTICS,
                "progress_range": [0.0, 1.0],
                "progress_storage_dtype": "float32",
                "progress_tie_break": "uniform_random_v1",
                "retrieval_seed": retrieval_seed,
                "progress_labels_dir": str(progress_labels_dir),
                "progress_data_split_preset": progress_data_split_preset,
                "query_progress_source": query_progress_source,
                "query_progress_cache": str(query_progress_cache) if query_progress_cache is not None else None,
            }
        )
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a task-scoped RICL retrieval corpus from LIBERO-100 HDF5 demos.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Directory containing libero_90/ and libero_10/ demos.")
    parser.add_argument("--libero-root", type=Path, default=Path("third_party/LIBERO"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--num-retrieved", type=int, default=4)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--embedding-type", choices=("CLS", "AVG", "16PATCHES", "64PATCHES"), default="CLS")
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--context-demos", type=int, default=10, help="Used only when --split-manifest is omitted.")
    parser.add_argument("--retrieval-backend", choices=("dino", "progress"), default="dino")
    parser.add_argument(
        "--progress-labels-dir",
        type=Path,
        default=None,
        help="VFE per_demo_relative_progress_v1 labels/task_XXX directory parent; required for progress retrieval.",
    )
    parser.add_argument(
        "--query-progress-source",
        choices=("gt", "cache"),
        default="gt",
        help="Use GT query progress now, or full-frame VFE predictions from --query-progress-cache later.",
    )
    parser.add_argument("--query-progress-cache", type=Path, default=None)
    parser.add_argument("--retrieval-seed", type=int, default=0)
    args = parser.parse_args()
    metadata = build_corpus(
        dataset_root=args.dataset_root,
        libero_root=args.libero_root,
        output_dir=args.output_dir,
        split_manifest=args.split_manifest,
        num_retrieved=args.num_retrieved,
        action_horizon=args.action_horizon,
        embedding_type=args.embedding_type,
        embedding_batch_size=args.embedding_batch_size,
        context_demos=args.context_demos,
        retrieval_backend=args.retrieval_backend,
        progress_labels_dir=args.progress_labels_dir,
        query_progress_source=args.query_progress_source,
        query_progress_cache=args.query_progress_cache,
        retrieval_seed=args.retrieval_seed,
    )
    print(f"Wrote {len(metadata['tasks'])} tasks to {args.output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
