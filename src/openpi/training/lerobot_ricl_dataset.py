"""Dataset reader for preprocessed LeRobot split-action RICL corpora."""

from __future__ import annotations

import contextlib
import json
from os import PathLike
import pathlib
from typing import SupportsIndex

import h5py
import numpy as np


class RiclLeRobotDataset:
    """Construct retrieved/query samples using the fixed 80/20 episode split."""

    def __init__(
        self,
        corpus_dir: str | PathLike[str],
        *,
        num_retrieved_observations: int,
        action_horizon: int,
        use_action_interpolation: bool,
        lamda: float,
    ) -> None:
        self.root = pathlib.Path(corpus_dir).expanduser().resolve()
        metadata_path = self.root / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"LeRobot RICL corpus metadata was not found at {metadata_path}. "
                "Run preprocessing/build_lerobot_ricl_corpus.py first."
            )
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("format_version") != 2:
            raise ValueError(f"Unsupported LeRobot RICL corpus format: {self.metadata.get('format_version')}")
        if int(self.metadata["state_dim"]) != 14 or int(self.metadata["action_dim"]) != 16:
            raise ValueError(
                "LeRobot RICL expects a 14-D state and 16-D action corpus; "
                f"got {self.metadata['state_dim']}-D state and {self.metadata['action_dim']}-D actions"
            )
        if int(self.metadata["num_retrieved"]) != num_retrieved_observations:
            raise ValueError(
                f"Model requests {num_retrieved_observations} retrieved observations, "
                f"but the corpus contains {self.metadata['num_retrieved']}"
            )
        if int(self.metadata["action_horizon"]) != action_horizon:
            raise ValueError(
                f"Model action horizon is {action_horizon}, corpus was built with {self.metadata['action_horizon']}"
            )

        self.num_retrieved_observations = num_retrieved_observations
        self.action_horizon = action_horizon
        self.use_action_interpolation = use_action_interpolation
        self.lamda = float(lamda)
        self.max_distance = float(self.metadata.get("max_distance", 1.0))
        if self.max_distance <= 0:
            raise ValueError(f"Corpus max_distance must be positive, got {self.max_distance}")

        records = self.metadata["train_episodes"] + self.metadata["context_episodes"]
        self._episode_paths = {int(record["episode_index"]): self.root / record["path"] for record in records}
        for path in self._episode_paths.values():
            if not path.exists():
                raise FileNotFoundError(f"Missing processed LeRobot RICL episode: {path}")

        with np.load(self.root / self.metadata["query_refs_path"], allow_pickle=False) as refs:
            query_episode_indices = np.asarray(refs["episode_indices"], dtype=np.int32)
            query_step_indices = np.asarray(refs["step_indices"], dtype=np.int32)
        if query_episode_indices.shape != query_step_indices.shape:
            raise ValueError("Corrupt LeRobot RICL query refs")
        self._samples = list(zip(query_episode_indices.tolist(), query_step_indices.tolist(), strict=True))
        if not self._samples:
            raise ValueError("The LeRobot RICL corpus has no training query frames")

        with np.load(self.root / self.metadata["context_refs_path"], allow_pickle=False) as refs:
            self._context_episode_indices = np.asarray(refs["episode_indices"], dtype=np.int32)
            self._context_step_indices = np.asarray(refs["step_indices"], dtype=np.int32)
        if self._context_episode_indices.shape != self._context_step_indices.shape:
            raise ValueError("Corrupt LeRobot RICL context refs")

        self._neighbors: dict[int, dict[str, np.ndarray]] = {}
        self._episode_files: dict[int, h5py.File] = {}

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: SupportsIndex) -> dict[str, object]:
        episode_index, step_index = self._samples[index.__index__()]
        neighbors = self._load_neighbors(episode_index)
        bank_indices = neighbors["retrieved_bank_indices"][step_index]
        if bank_indices.shape != (self.num_retrieved_observations,):
            raise ValueError(f"Unexpected LeRobot RICL neighbor shape {bank_indices.shape}")

        data: dict[str, object] = {}
        for retrieved_number, raw_bank_index in enumerate(bank_indices):
            bank_index = int(raw_bank_index)
            if not 0 <= bank_index < len(self._context_episode_indices):
                raise IndexError(f"Context bank index {bank_index} is out of range")
            retrieved_episode = int(self._context_episode_indices[bank_index])
            retrieved_step = int(self._context_step_indices[bank_index])
            data.update(self._read_block(retrieved_episode, retrieved_step, f"retrieved_{retrieved_number}_"))

        data.update(self._read_block(episode_index, step_index, "query_"))
        if self.use_action_interpolation:
            distances = np.asarray(neighbors["relative_distances"][step_index], dtype=np.float32)
            expected_shape = (self.num_retrieved_observations + 1,)
            if distances.shape != expected_shape:
                raise ValueError(f"Expected relative distances with shape {expected_shape}, got {distances.shape}")
            data["exp_lamda_distances"] = np.exp(-self.lamda * distances / self.max_distance).reshape(-1, 1)
        return data

    def _read_block(self, episode_index: int, step_index: int, prefix: str) -> dict[str, object]:
        episode = self._episode(episode_index)
        end = step_index + self.action_horizon
        if not 0 <= step_index < len(episode["state"]) or end > len(episode["actions"]):
            raise IndexError(f"Invalid action chunk [{step_index}:{end}] for episode {episode_index}")
        return {
            f"{prefix}top_image": np.asarray(episode["top_image"][step_index]),
            f"{prefix}wrist_image": np.asarray(episode["wrist_image"][step_index]),
            f"{prefix}right_image": np.asarray(episode["right_image"][step_index]),
            f"{prefix}state": np.asarray(episode["state"][step_index], dtype=np.float32),
            f"{prefix}actions": np.asarray(episode["actions"][step_index:end], dtype=np.float32),
            f"{prefix}prompt": str(episode.attrs.get("prompt", self.metadata["task_prompt"])),
        }

    def _load_neighbors(self, episode_index: int) -> dict[str, np.ndarray]:
        if episode_index not in self._neighbors:
            path = self.root / self.metadata["neighbors_dir"] / f"episode_{episode_index:06d}.npz"
            if not path.exists():
                raise FileNotFoundError(f"Missing LeRobot RICL retrieval neighbors: {path}")
            with np.load(path, allow_pickle=False) as neighbors:
                self._neighbors[episode_index] = {key: np.asarray(neighbors[key]) for key in neighbors.files}
        return self._neighbors[episode_index]

    def _episode(self, episode_index: int) -> h5py.File:
        if episode_index not in self._episode_files:
            self._episode_files[episode_index] = h5py.File(self._episode_paths[episode_index], "r")
        return self._episode_files[episode_index]

    def close(self) -> None:
        for episode in self._episode_files.values():
            episode.close()
        self._episode_files.clear()

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_episode_files"] = {}
        return state

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()
