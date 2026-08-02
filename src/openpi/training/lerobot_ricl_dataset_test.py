import json
import pickle

import h5py
import numpy as np

from openpi.policies import lerobot_ricl_policy
from openpi.training.lerobot_ricl_dataset import RiclLeRobotDataset


def _write_episode(path, *, offset: float) -> None:
    length = 12
    with h5py.File(path, "w") as episode:
        episode.attrs["prompt"] = "test lerobot task"
        episode.create_dataset("state", data=np.full((length, 14), offset, dtype=np.float32))
        episode.create_dataset("actions", data=np.full((length, 16), offset + 1, dtype=np.float32))
        for key, value in (("top_image", 10), ("right_image", 20), ("wrist_image", 30)):
            episode.create_dataset(key, data=np.full((length, 8, 8, 3), value, dtype=np.uint8))


def _write_corpus(root) -> None:
    episodes_dir = root / "episodes"
    neighbors_dir = root / "neighbors"
    episodes_dir.mkdir()
    neighbors_dir.mkdir()
    _write_episode(episodes_dir / "episode_000000.h5", offset=0)
    _write_episode(episodes_dir / "episode_000001.h5", offset=2)
    np.savez(root / "query_refs.npz", episode_indices=np.array([0]), step_indices=np.array([1]))
    np.savez(root / "context_refs.npz", episode_indices=np.array([1]), step_indices=np.array([2]))
    np.savez(
        neighbors_dir / "episode_000000.npz",
        retrieved_bank_indices=np.zeros((3, 4), dtype=np.int32),
        relative_distances=np.array([[0, 1, 2, 3, 4]] * 3, dtype=np.float32),
    )
    metadata = {
        "format_version": 2,
        "state_dim": 14,
        "action_dim": 16,
        "action_horizon": 10,
        "num_retrieved": 4,
        "max_distance": 4.0,
        "task_prompt": "test lerobot task",
        "train_episodes": [{"episode_index": 0, "path": "episodes/episode_000000.h5"}],
        "context_episodes": [{"episode_index": 1, "path": "episodes/episode_000001.h5"}],
        "query_refs_path": "query_refs.npz",
        "context_refs_path": "context_refs.npz",
        "neighbors_dir": "neighbors",
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_dataset_and_lerobot_transforms(tmp_path) -> None:
    _write_corpus(tmp_path)
    dataset = RiclLeRobotDataset(
        tmp_path,
        num_retrieved_observations=4,
        action_horizon=10,
        use_action_interpolation=True,
        lamda=10.0,
    )

    sample = dataset[0]
    assert sample["query_state"].shape == (14,)
    assert sample["query_actions"].shape == (10, 16)
    assert sample["retrieved_0_actions"].shape == (10, 16)
    assert sample["exp_lamda_distances"].shape == (5, 1)

    inputs = lerobot_ricl_policy.RiclLeRobotInputs(
        action_dim=16,
        num_retrieved_observations=4,
    )(sample)
    assert inputs["query_image"]["base_0_rgb"].shape == (8, 8, 3)
    assert inputs["query_image"]["base_1_rgb"][0, 0, 0] == 20
    assert inputs["query_image"]["wrist_0_rgb"][0, 0, 0] == 30

    padded = lerobot_ricl_policy.PadRiclLeRobotStates(
        action_dim=16,
        num_retrieved_observations=4,
    )(inputs)
    assert padded["query_state"].shape == (16,)
    np.testing.assert_array_equal(padded["query_state"][-2:], np.zeros(2, dtype=np.float32))

    restored_dataset = pickle.loads(pickle.dumps(dataset))
    assert restored_dataset[0]["query_actions"].shape == (10, 16)
    restored_dataset.close()
    dataset.close()
