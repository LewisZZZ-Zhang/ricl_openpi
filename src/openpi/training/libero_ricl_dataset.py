"""PyTorch dataset that turns a LIBERO RICL corpus into retrieved/query samples."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import SupportsIndex

import numpy as np

from openpi.policies.libero_retrieval import LiberoRiclCorpus, libero_action_chunk


@dataclass(frozen=True)
class _SampleRef:
    task_id: int
    demo_id: str
    step_idx: int
    neighbors_path: str


class RiclLiberoDataset:
    """Load RICL training samples from a precomputed, task-scoped LIBERO corpus."""

    def __init__(
        self,
        corpus_dir: str | PathLike[str],
        *,
        num_retrieved_observations: int,
        action_horizon: int,
        use_action_interpolation: bool,
        lamda: float,
    ) -> None:
        self.corpus = LiberoRiclCorpus(corpus_dir)
        expected_retrieved = int(self.corpus.metadata["num_retrieved"])
        if num_retrieved_observations != expected_retrieved:
            raise ValueError(
                f"Model requests {num_retrieved_observations} retrieved observations, "
                f"but the corpus contains {expected_retrieved}"
            )
        expected_horizon = int(self.corpus.metadata["action_horizon"])
        if action_horizon != expected_horizon:
            raise ValueError(f"Model action horizon is {action_horizon}, corpus was built with {expected_horizon}")

        self.num_retrieved_observations = num_retrieved_observations
        self.action_horizon = action_horizon
        self.use_action_interpolation = use_action_interpolation
        self.lamda = float(lamda)
        self._neighbor_cache: dict[str, dict[str, np.ndarray]] = {}
        self._samples: list[_SampleRef] = []

        for task in self.corpus.metadata["tasks"]:
            if not task["is_train"]:
                continue
            neighbors_dir = task.get("neighbors_dir")
            if neighbors_dir is None:
                raise ValueError(f"Training task {task['task_id']} has no precomputed neighbours")
            for demo_id in task["query_demo_ids"]:
                neighbors_path = self.corpus.root / neighbors_dir / f"{demo_id}.npz"
                if not neighbors_path.exists():
                    raise FileNotFoundError(f"Missing retrieval neighbours: {neighbors_path}")
                with np.load(neighbors_path, allow_pickle=False) as neighbors:
                    length = int(neighbors["retrieved_bank_indices"].shape[0])
                self._samples.extend(
                    _SampleRef(int(task["task_id"]), str(demo_id), step_idx, str(neighbors_path))
                    for step_idx in range(length)
                )
        if not self._samples:
            raise ValueError("The RICL LIBERO corpus has no training query frames")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: SupportsIndex) -> dict:
        sample = self._samples[index.__index__()]
        neighbors = self._neighbors(sample.neighbors_path)
        bank_indices = neighbors["retrieved_bank_indices"][sample.step_idx]
        if bank_indices.shape != (self.num_retrieved_observations,):
            raise ValueError(f"Unexpected neighbour shape {bank_indices.shape}")
        if self.corpus.retrieval_backend == "progress":
            bank = self.corpus.progress_bank(sample.task_id)
            retrieved_demo_indices = bank.demo_indices[bank_indices]
            if len(np.unique(retrieved_demo_indices)) != len(retrieved_demo_indices):
                raise ValueError("Progress retrieval must select at most one frame from each context demo")

        task = self.corpus.task(sample.task_id)
        data: dict[str, object] = {}
        for retrieved_number, bank_index in enumerate(bank_indices):
            demo_id, step_idx = self.corpus.retrieved_ref(sample.task_id, int(bank_index))
            frame = self.corpus.read_frame(sample.task_id, demo_id, step_idx)
            prefix = f"retrieved_{retrieved_number}_"
            data[f"{prefix}top_image"] = frame["top_image"]
            data[f"{prefix}wrist_image"] = frame["wrist_image"]
            data[f"{prefix}state"] = frame["state"]
            data[f"{prefix}actions"] = libero_action_chunk(frame["actions"], step_idx, self.action_horizon)
            data[f"{prefix}prompt"] = task["prompt"]

        query = self.corpus.read_frame(sample.task_id, sample.demo_id, sample.step_idx)
        data["query_top_image"] = query["top_image"]
        data["query_wrist_image"] = query["wrist_image"]
        data["query_state"] = query["state"]
        data["query_actions"] = libero_action_chunk(query["actions"], sample.step_idx, self.action_horizon)
        data["query_prompt"] = task["prompt"]

        if self.use_action_interpolation:
            distances = neighbors["relative_distances"][sample.step_idx]
            distances = np.asarray(distances, dtype=np.float32) / self.corpus.max_distance
            data["exp_lamda_distances"] = np.exp(-self.lamda * distances).reshape(-1, 1)
        return data

    def _neighbors(self, path: str) -> dict[str, np.ndarray]:
        if path not in self._neighbor_cache:
            with np.load(path, allow_pickle=False) as neighbors:
                self._neighbor_cache[path] = {key: neighbors[key] for key in neighbors.files}
        return self._neighbor_cache[path]

    def close(self) -> None:
        self.corpus.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
