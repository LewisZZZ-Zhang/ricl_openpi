"""Validate every task, bank, and training-neighbor file in a LIBERO RICL corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from openpi.policies.libero_retrieval import LiberoRiclCorpus
from openpi.shared import normalize
from openpi.training.libero_ricl_dataset import RiclLiberoDataset


def validate_corpus(corpus_dir: Path, expected_backend: str | None = None) -> dict[str, int | str]:
    corpus = LiberoRiclCorpus(corpus_dir)
    backend = corpus.retrieval_backend
    if expected_backend is not None and backend != expected_backend:
        raise ValueError(f"Expected retrieval backend {expected_backend!r}, found {backend!r}")
    task_ids = [int(task["task_id"]) for task in corpus.metadata["tasks"]]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Corpus contains duplicate task IDs")

    context_frames = 0
    query_frames = 0
    neighbor_files = 0
    try:
        for task in corpus.metadata["tasks"]:
            task_id = int(task["task_id"])
            context_demo_ids = [str(value) for value in task["context_demo_ids"]]
            query_demo_ids = [str(value) for value in task["query_demo_ids"]]
            overlap = set(context_demo_ids).intersection(query_demo_ids)
            if overlap:
                raise ValueError(f"Task {task_id} has query/context overlap: {sorted(overlap)}")
            bank = corpus.context_bank(task_id) if backend == "dino" else corpus.progress_bank(task_id)
            context_frames += len(bank.demo_indices)
            if np.any(bank.demo_indices < 0) or np.any(bank.demo_indices >= len(context_demo_ids)):
                raise ValueError(f"Task {task_id} contains invalid context demo indices")
            if backend == "progress":
                if not np.isfinite(bank.progress).all() or np.any((bank.progress < 0) | (bank.progress > 1)):
                    raise ValueError(f"Task {task_id} contains invalid context progress")

            if not task["is_train"]:
                if task.get("neighbors_dir") is not None:
                    raise ValueError(f"Eval-only task {task_id} unexpectedly contains training neighbors")
                continue
            neighbors_dir = corpus.root / task["neighbors_dir"]
            with h5py.File(task["source_hdf5"], "r") as h5_file:
                for demo_id in query_demo_ids:
                    path = neighbors_dir / f"{demo_id}.npz"
                    if not path.exists():
                        raise FileNotFoundError(f"Missing neighbor file: {path}")
                    expected_length = len(h5_file[f"data/{demo_id}/actions"])
                    with np.load(path, allow_pickle=False) as payload:
                        indices = np.asarray(payload["retrieved_bank_indices"], dtype=np.int32)
                        relative = np.asarray(payload["relative_distances"], dtype=np.float32)
                        if indices.shape != (expected_length, int(corpus.metadata["num_retrieved"])):
                            raise ValueError(f"Unexpected neighbor shape in {path}: {indices.shape}")
                        if relative.shape != (expected_length, indices.shape[1] + 1):
                            raise ValueError(f"Unexpected relative-distance shape in {path}: {relative.shape}")
                        if np.any(indices < 0) or np.any(indices >= len(bank.demo_indices)):
                            raise ValueError(f"Out-of-range neighbor index in {path}")
                        if backend == "progress":
                            selected_demos = bank.demo_indices[indices]
                            if any(len(np.unique(row)) != indices.shape[1] for row in selected_demos):
                                raise ValueError(f"Progress neighbors are not demo-diverse in {path}")
                            query_progress = np.asarray(payload["query_progress"], dtype=np.float32)
                            distances = np.asarray(payload["retrieved_progress_distances"], dtype=np.float32)
                            if query_progress.shape != (expected_length,) or distances.shape != indices.shape:
                                raise ValueError(f"Progress audit arrays do not align in {path}")
                            expected_distances = np.abs(bank.progress[indices] - query_progress[:, None])
                            np.testing.assert_allclose(distances, expected_distances, atol=1e-6)
                    query_frames += expected_length
                    neighbor_files += 1
    finally:
        corpus.close()
    return {
        "retrieval_backend": backend,
        "tasks": len(task_ids),
        "context_frames": context_frames,
        "query_frames": query_frames,
        "neighbor_files": neighbor_files,
    }


def validate_training_inputs(
    corpus_dir: Path,
    *,
    assets_dir: Path,
    init_checkpoint_dir: Path,
) -> dict[str, int | str]:
    metadata_path = init_checkpoint_dir / "_CHECKPOINT_METADATA"
    params_dir = init_checkpoint_dir / "params"
    if not metadata_path.is_file() or not params_dir.is_dir():
        raise FileNotFoundError(f"Incomplete initialization checkpoint: {init_checkpoint_dir}")

    corpus = LiberoRiclCorpus(corpus_dir)
    try:
        num_retrieved = int(corpus.metadata["num_retrieved"])
        action_horizon = int(corpus.metadata["action_horizon"])
    finally:
        corpus.close()
    stats = normalize.load(assets_dir / "libero_ricl")
    expected_stats = {"query_state", "query_actions"}
    expected_stats.update(
        f"retrieved_{index}_{name}"
        for index in range(num_retrieved)
        for name in ("state", "actions")
    )
    missing_stats = expected_stats.difference(stats)
    if missing_stats:
        raise ValueError(f"Normalization statistics are missing keys: {sorted(missing_stats)}")

    dataset = RiclLiberoDataset(
        corpus_dir,
        num_retrieved_observations=num_retrieved,
        action_horizon=action_horizon,
        use_action_interpolation=False,
        lamda=10.0,
    )
    try:
        if not len(dataset):
            raise ValueError("Training dataset is empty")
        for index in sorted({0, len(dataset) // 2, len(dataset) - 1}):
            sample = dataset[index]
            if np.asarray(sample["query_actions"]).shape != (action_horizon, 7):
                raise ValueError(f"Unexpected query action shape at sample {index}")
            for retrieved_index in range(num_retrieved):
                key = f"retrieved_{retrieved_index}_actions"
                if np.asarray(sample[key]).shape != (action_horizon, 7):
                    raise ValueError(f"Unexpected {key} shape at sample {index}")
    finally:
        dataset.close()
    return {
        "training_samples": len(dataset),
        "normalization_fields": len(stats),
        "init_checkpoint": str(init_checkpoint_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--expected-backend", choices=("dino", "progress"), default=None)
    parser.add_argument("--assets-dir", type=Path)
    parser.add_argument("--init-checkpoint-dir", type=Path)
    args = parser.parse_args()
    if (args.assets_dir is None) != (args.init_checkpoint_dir is None):
        parser.error("--assets-dir and --init-checkpoint-dir must be supplied together")
    summary = validate_corpus(args.corpus_dir, args.expected_backend)
    print("Validated LIBERO RICL corpus:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    if args.assets_dir is not None:
        training_summary = validate_training_inputs(
            args.corpus_dir,
            assets_dir=args.assets_dir,
            init_checkpoint_dir=args.init_checkpoint_dir,
        )
        print("Validated training inputs:")
        for key, value in training_summary.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
