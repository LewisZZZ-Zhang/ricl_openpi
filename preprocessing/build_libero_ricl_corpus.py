"""Build a task-scoped DINO retrieval corpus for RICL on LIBERO-100.

The corpus stores only DINO embeddings and frame references. Images, states, and
actions continue to live in the original LIBERO HDF5 files, so the data is not
duplicated. Training neighbours are selected exclusively from a task's context
demos; query and context episode sets must therefore be disjoint.
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

from openpi.policies.utils import embed_with_batches, embedding_dim, load_dinov2


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
) -> dict[str, Any]:
    if num_retrieved <= 0:
        raise ValueError("num_retrieved must be positive")
    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive")
    embedding_dim(embedding_type)  # validate before beginning a long DINO run

    task_splits = (
        _manifest_splits(split_manifest.resolve()) if split_manifest is not None else _default_splits(dataset_root, libero_root, context_demos)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir = output_dir / "embeddings"
    neighbors_dir = output_dir / "neighbors"
    embeddings_dir.mkdir(exist_ok=True)
    neighbors_dir.mkdir(exist_ok=True)

    dinov2 = load_dinov2()
    metadata_tasks: list[dict[str, Any]] = []
    max_distance = 0.0
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

            context_embeddings: list[np.ndarray] = []
            context_demo_indices: list[np.ndarray] = []
            context_step_indices: list[np.ndarray] = []
            for demo_index, demo_id in enumerate(split.context_demo_ids):
                images = h5_file[f"data/{demo_id}/obs/agentview_rgb"][:]
                embeddings = _embed_images(images, dinov2, embedding_type, embedding_batch_size)
                context_embeddings.append(embeddings)
                context_demo_indices.append(np.full(len(embeddings), demo_index, dtype=np.int32))
                context_step_indices.append(np.arange(len(embeddings), dtype=np.int32))
            all_context_embeddings = np.concatenate(context_embeddings, axis=0)
            all_context_demo_indices = np.concatenate(context_demo_indices, axis=0)
            all_context_step_indices = np.concatenate(context_step_indices, axis=0)

            task_dir = embeddings_dir / f"task_{split.task_id:03d}"
            task_dir.mkdir(exist_ok=True)
            embeddings_path = task_dir / "context_embeddings.npy"
            refs_path = task_dir / "context_refs.npz"
            np.save(embeddings_path, all_context_embeddings.astype(np.float32))
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
                    query_images = h5_file[f"data/{demo_id}/obs/agentview_rgb"][:]
                    query_embeddings = _embed_images(query_images, dinov2, embedding_type, embedding_batch_size)
                    neighbor_indices, relative_distances = _neighbors(query_embeddings, all_context_embeddings, num_retrieved)
                    max_distance = max(max_distance, float(relative_distances.max()))
                    np.savez_compressed(
                        task_neighbors_dir / f"{demo_id}.npz",
                        retrieved_bank_indices=neighbor_indices,
                        relative_distances=relative_distances,
                    )
                neighbors_rel_path = str(task_neighbors_dir.relative_to(output_dir))

        metadata_tasks.append(
            {
                "task_id": split.task_id,
                "suite": split.suite,
                "task_name": split.task_name,
                "prompt": split.prompt,
                "source_hdf5": str(source_hdf5),
                "context_demo_ids": split.context_demo_ids,
                "query_demo_ids": split.query_demo_ids,
                "is_train": split.is_train,
                "context_embeddings_path": str(embeddings_path.relative_to(output_dir)),
                "context_refs_path": str(refs_path.relative_to(output_dir)),
                "neighbors_dir": neighbors_rel_path,
            }
        )
        print(
            f"[{task_position}/{len(task_splits)}] task={split.task_id} "
            f"context_frames={len(all_context_embeddings)} train={split.is_train}"
        )

    metadata = {
        "format_version": 1,
        "dataset": "LIBERO-100",
        "dataset_root": str(dataset_root.resolve()),
        "split_manifest": str(split_manifest.resolve()) if split_manifest is not None else None,
        "embedding_type": embedding_type,
        "embedding_dim": embedding_dim(embedding_type),
        "embedding_storage_dtype": "float32",
        "num_retrieved": num_retrieved,
        "action_horizon": action_horizon,
        "max_distance": max(max_distance, np.finfo(np.float32).eps),
        "tasks": metadata_tasks,
    }
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
    )
    print(f"Wrote {len(metadata['tasks'])} tasks to {args.output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
