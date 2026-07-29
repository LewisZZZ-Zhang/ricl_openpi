"""Shared LIBERO-100 retrieval-corpus access for RICL training and inference."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np


def _flip_libero_image(image: np.ndarray) -> np.ndarray:
    """Match the 180-degree rotation used by the LIBERO policy evaluator."""
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected an HWC image, got {image.shape}")
    return np.ascontiguousarray(image[::-1, ::-1])


def libero_action_chunk(actions: np.ndarray, step_idx: int, action_horizon: int) -> np.ndarray:
    """Return a fixed-length LIBERO action chunk, padding motion with zeros at episode end."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected LIBERO actions with shape [steps, 7], got {actions.shape}")
    if not 0 <= step_idx < len(actions):
        raise IndexError(f"{step_idx=} is outside an episode with {len(actions)} actions")

    chunk = actions[step_idx : step_idx + action_horizon]
    if len(chunk) == action_horizon:
        return chunk

    # LIBERO actions are 6-D Cartesian deltas plus an absolute gripper command.
    padding = np.zeros((action_horizon - len(chunk), actions.shape[1]), dtype=np.float32)
    padding[:, -1] = actions[-1, -1]
    return np.concatenate((chunk, padding), axis=0)


@dataclass(frozen=True)
class ContextBank:
    embeddings: np.ndarray
    demo_indices: np.ndarray
    step_indices: np.ndarray

    def search(self, query_embedding: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Exact L2 KNN for one query. Per-task LIBERO banks are intentionally small."""
        query_embedding = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if query_embedding.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                f"Embedding dimension mismatch: query={query_embedding.shape[0]}, "
                f"bank={self.embeddings.shape[1]}"
            )
        if not 0 < k <= len(self.embeddings):
            raise ValueError(f"Requested {k} neighbors from a bank with {len(self.embeddings)} frames")
        squared_distances = np.sum((self.embeddings - query_embedding) ** 2, axis=1)
        candidate_indices = np.argpartition(squared_distances, k - 1)[:k]
        ordered = candidate_indices[np.argsort(squared_distances[candidate_indices], kind="stable")]
        return np.sqrt(squared_distances[ordered]), ordered.astype(np.int32, copy=False)


class LiberoRiclCorpus:
    """Read-only retrieval corpus backed by the original LIBERO HDF5 demonstrations."""

    def __init__(self, corpus_dir: str | pathlib.Path):
        self.root = pathlib.Path(corpus_dir).expanduser().resolve()
        metadata_path = self.root / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"RICL LIBERO corpus metadata was not found at {metadata_path}. "
                "Run preprocessing/build_libero_ricl_corpus.py first."
            )
        self.metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("format_version") != 1:
            raise ValueError(f"Unsupported RICL LIBERO corpus format: {self.metadata.get('format_version')}")
        self.tasks = {int(task["task_id"]): task for task in self.metadata["tasks"]}
        self._banks: dict[int, ContextBank] = {}
        self._h5_files: dict[str, h5py.File] = {}
        self._actions: dict[tuple[str, str], np.ndarray] = {}

    @property
    def max_distance(self) -> float:
        return float(self.metadata.get("max_distance", 1.0))

    @property
    def embedding_type(self) -> str:
        return str(self.metadata["embedding_type"])

    def task(self, task_id: int) -> dict[str, Any]:
        try:
            return self.tasks[int(task_id)]
        except KeyError as exc:
            known = sorted(self.tasks)
            raise KeyError(f"Unknown LIBERO retrieval task {task_id}; available task ids: {known}") from exc

    def context_bank(self, task_id: int) -> ContextBank:
        task_id = int(task_id)
        if task_id not in self._banks:
            task = self.task(task_id)
            embeddings = np.load(self.root / task["context_embeddings_path"], mmap_mode="r")
            with np.load(self.root / task["context_refs_path"], allow_pickle=False) as refs:
                demo_indices = np.asarray(refs["demo_indices"], dtype=np.int32)
                step_indices = np.asarray(refs["step_indices"], dtype=np.int32)
            embeddings = np.asarray(embeddings, dtype=np.float32)
            if len(embeddings) != len(demo_indices) or len(embeddings) != len(step_indices):
                raise ValueError(f"Corrupt context bank for task {task_id}")
            self._banks[task_id] = ContextBank(embeddings, demo_indices, step_indices)
        return self._banks[task_id]

    def search(self, task_id: int, query_embedding: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        return self.context_bank(task_id).search(query_embedding, k)

    def retrieved_ref(self, task_id: int, bank_index: int) -> tuple[str, int]:
        task = self.task(task_id)
        bank = self.context_bank(task_id)
        demo_index = int(bank.demo_indices[bank_index])
        context_demo_ids = task["context_demo_ids"]
        if not 0 <= demo_index < len(context_demo_ids):
            raise IndexError(f"Invalid context-demo index {demo_index} for task {task_id}")
        return str(context_demo_ids[demo_index]), int(bank.step_indices[bank_index])

    def read_frame(self, task_id: int, demo_id: str, step_idx: int) -> dict[str, np.ndarray]:
        task = self.task(task_id)
        source_hdf5 = str(task["source_hdf5"])
        h5_file = self._h5(source_hdf5)
        prefix = f"data/{demo_id}"
        if prefix not in h5_file:
            raise KeyError(f"{demo_id!r} is missing from {source_hdf5}")
        obs = h5_file[f"{prefix}/obs"]
        return {
            "top_image": _flip_libero_image(obs["agentview_rgb"][step_idx]),
            "wrist_image": _flip_libero_image(obs["eye_in_hand_rgb"][step_idx]),
            "state": np.concatenate((obs["ee_states"][step_idx], obs["gripper_states"][step_idx])).astype(
                np.float32
            ),
            "actions": self._trajectory_actions(source_hdf5, demo_id),
        }

    def read_action_chunk(self, task_id: int, demo_id: str, step_idx: int, action_horizon: int) -> np.ndarray:
        task = self.task(task_id)
        source_hdf5 = str(task["source_hdf5"])
        actions = self._trajectory_actions(source_hdf5, demo_id)
        return libero_action_chunk(actions, step_idx, action_horizon)

    def read_trajectory(self, task_id: int, demo_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Read all state/action rows for normalization without materializing any images."""
        task = self.task(task_id)
        source_hdf5 = str(task["source_hdf5"])
        prefix = f"data/{demo_id}"
        h5_file = self._h5(source_hdf5)
        obs = h5_file[f"{prefix}/obs"]
        states = np.concatenate((obs["ee_states"][:], obs["gripper_states"][:]), axis=1).astype(np.float32)
        actions = self._trajectory_actions(source_hdf5, demo_id)
        if len(states) != len(actions):
            raise ValueError(f"State/action length mismatch for task={task_id}, demo={demo_id}")
        return states, actions

    def close(self) -> None:
        for h5_file in self._h5_files.values():
            h5_file.close()
        self._h5_files.clear()
        self._actions.clear()

    def _h5(self, source_hdf5: str) -> h5py.File:
        if source_hdf5 not in self._h5_files:
            self._h5_files[source_hdf5] = h5py.File(source_hdf5, "r")
        return self._h5_files[source_hdf5]

    def _trajectory_actions(self, source_hdf5: str, demo_id: str) -> np.ndarray:
        key = (source_hdf5, demo_id)
        if key not in self._actions:
            self._actions[key] = np.asarray(self._h5(source_hdf5)[f"data/{demo_id}/actions"][:], dtype=np.float32)
        return self._actions[key]

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
