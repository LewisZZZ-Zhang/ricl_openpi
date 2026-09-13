import json

import h5py
import numpy as np

from openpi.policies.libero_retrieval import (
    LiberoRiclCorpus,
    libero_action_chunk,
    progress_neighbors,
    relative_progress_values,
)
from openpi.policies.policy import RiclLiberoPolicy


def _write_demo(h5_file, demo_id: str, image_offset: int) -> None:
    demo = h5_file.create_group(f"data/{demo_id}")
    obs = demo.create_group("obs")
    images = np.arange(2 * 2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 2, 3) + image_offset
    obs.create_dataset("agentview_rgb", data=images)
    obs.create_dataset("eye_in_hand_rgb", data=images + 40)
    obs.create_dataset("ee_states", data=np.arange(12, dtype=np.float32).reshape(2, 6))
    obs.create_dataset("gripper_states", data=np.arange(4, dtype=np.float32).reshape(2, 2))
    actions = np.arange(14, dtype=np.float32).reshape(2, 7)
    demo.create_dataset("actions", data=actions)


def test_libero_corpus_reads_flipped_frames_and_topk(tmp_path):
    h5_path = tmp_path / "libero_demo.hdf5"
    with h5py.File(h5_path, "w") as h5_file:
        h5_file.create_group("data")
        _write_demo(h5_file, "demo_0", 0)

    corpus_dir = tmp_path / "corpus"
    task_dir = corpus_dir / "embeddings" / "task_000"
    task_dir.mkdir(parents=True)
    np.save(task_dir / "context_embeddings.npy", np.asarray([[0.0, 0.0], [3.0, 4.0]], dtype=np.float16))
    np.savez(task_dir / "context_refs.npz", demo_indices=np.asarray([0, 0]), step_indices=np.asarray([0, 1]))
    (corpus_dir / "metadata.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "embedding_type": "CLS",
                "max_distance": 1.0,
                "num_retrieved": 1,
                "tasks": [
                    {
                        "task_id": 0,
                        "source_hdf5": str(h5_path),
                        "context_demo_ids": ["demo_0"],
                        "context_embeddings_path": "embeddings/task_000/context_embeddings.npy",
                        "context_refs_path": "embeddings/task_000/context_refs.npz",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    corpus = LiberoRiclCorpus(corpus_dir)
    try:
        distances, indices = corpus.search(0, np.asarray([0.2, 0.1], dtype=np.float32), 1)
        assert indices.tolist() == [0]
        assert distances[0] > 0
        frame = corpus.read_frame(0, "demo_0", 0)
        with h5py.File(h5_path, "r") as h5_file:
            expected_top = np.asarray(h5_file["data/demo_0/obs/agentview_rgb"][0])[::-1, ::-1]
        np.testing.assert_array_equal(frame["top_image"], expected_top)
        assert frame["state"].shape == (8,)
    finally:
        corpus.close()


def test_libero_action_chunk_uses_zero_motion_and_last_gripper_at_episode_end():
    actions = np.arange(14, dtype=np.float32).reshape(2, 7)
    chunk = libero_action_chunk(actions, step_idx=1, action_horizon=4)
    np.testing.assert_array_equal(chunk[0], actions[1])
    np.testing.assert_array_equal(chunk[1:, :-1], np.zeros((3, 6), dtype=np.float32))
    np.testing.assert_array_equal(chunk[1:, -1], np.full(3, actions[-1, -1], dtype=np.float32))


def test_relative_progress_values_include_both_endpoints():
    np.testing.assert_allclose(relative_progress_values(3), [0.0, 0.5, 1.0])
    np.testing.assert_allclose(relative_progress_values(1), [1.0])


def test_progress_neighbors_select_nearest_distinct_demos_with_random_ties():
    context_progress = np.asarray([0.0, 0.5, 1.0, 0.0, 0.4, 1.0, 0.0, 0.6, 1.0], dtype=np.float32)
    demo_indices = np.repeat(np.arange(3, dtype=np.int32), 3)
    indices, relative = progress_neighbors(
        [0.5, 1.0], context_progress, demo_indices, k=2, rng=np.random.default_rng(7)
    )

    assert indices[0, 0] == 1
    assert demo_indices[indices[0, 1]] in {1, 2}
    assert len(set(demo_indices[indices[0]].tolist())) == 2
    assert len(set(demo_indices[indices[1]].tolist())) == 2
    np.testing.assert_allclose(relative[0], [0.0, 0.1, 0.0], atol=1e-6)
    np.testing.assert_allclose(relative[1], [0.0, 0.0, 0.0], atol=1e-6)


def test_progress_neighbors_give_tied_context_demos_equal_opportunity():
    context_progress = np.full(10, 0.5, dtype=np.float32)
    demo_indices = np.arange(10, dtype=np.int32)
    indices, _ = progress_neighbors(
        np.full(10_000, 0.5, dtype=np.float32),
        context_progress,
        demo_indices,
        k=4,
        rng=np.random.default_rng(7),
    )
    selected_demos = demo_indices[indices]
    usage = np.bincount(selected_demos.reshape(-1), minlength=10)
    assert np.all(np.abs(usage - 4_000) < 200), usage


def test_libero_progress_corpus_searches_scalar_bank(tmp_path):
    h5_path = tmp_path / "libero_demo.hdf5"
    with h5py.File(h5_path, "w") as h5_file:
        h5_file.create_group("data")
        _write_demo(h5_file, "demo_0", 0)
        _write_demo(h5_file, "demo_1", 1)

    corpus_dir = tmp_path / "corpus"
    task_dir = corpus_dir / "progress" / "task_000"
    task_dir.mkdir(parents=True)
    np.save(task_dir / "context_progress.npy", np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32))
    np.savez(
        task_dir / "context_refs.npz",
        demo_indices=np.asarray([0, 0, 1, 1], dtype=np.int32),
        step_indices=np.asarray([0, 1, 0, 1], dtype=np.int32),
    )
    (corpus_dir / "metadata.json").write_text(
        json.dumps(
            {
                "format_version": 2,
                "retrieval_backend": "progress",
                "progress_semantics": "per_demo_relative_progress_v1",
                "progress_range": [0.0, 1.0],
                "progress_tie_break": "uniform_random_v1",
                "max_distance": 1.0,
                "num_retrieved": 2,
                "tasks": [
                    {
                        "task_id": 0,
                        "source_hdf5": str(h5_path),
                        "context_demo_ids": ["demo_0", "demo_1"],
                        "context_progress_path": "progress/task_000/context_progress.npy",
                        "context_refs_path": "progress/task_000/context_refs.npz",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    corpus = LiberoRiclCorpus(corpus_dir)
    try:
        distances, indices = corpus.search_progress(0, 0.8, 2, rng=np.random.default_rng(7))
        assert set(indices.tolist()) == {1, 3}
        np.testing.assert_allclose(distances, [0.2, 0.2])
        assert {corpus.retrieved_ref(0, int(index)) for index in indices} == {
            ("demo_0", 1),
            ("demo_1", 1),
        }
    finally:
        corpus.close()


def test_ricl_policy_uses_progress_predictor_and_strips_runtime_fields(tmp_path):
    h5_path = tmp_path / "libero_demo.hdf5"
    with h5py.File(h5_path, "w") as h5_file:
        h5_file.create_group("data")
        _write_demo(h5_file, "demo_0", 0)
        _write_demo(h5_file, "demo_1", 1)
    corpus_dir = tmp_path / "corpus"
    task_dir = corpus_dir / "progress" / "task_000"
    task_dir.mkdir(parents=True)
    np.save(task_dir / "context_progress.npy", np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32))
    np.savez(
        task_dir / "context_refs.npz",
        demo_indices=np.asarray([0, 0, 1, 1], dtype=np.int32),
        step_indices=np.asarray([0, 1, 0, 1], dtype=np.int32),
    )
    (corpus_dir / "metadata.json").write_text(
        json.dumps(
            {
                "format_version": 2,
                "retrieval_backend": "progress",
                "progress_semantics": "per_demo_relative_progress_v1",
                "progress_range": [0.0, 1.0],
                "progress_tie_break": "uniform_random_v1",
                "max_distance": 1.0,
                "num_retrieved": 2,
                "tasks": [
                    {
                        "task_id": 0,
                        "prompt": "test task",
                        "source_hdf5": str(h5_path),
                        "context_demo_ids": ["demo_0", "demo_1"],
                        "context_progress_path": "progress/task_000/context_progress.npy",
                        "context_refs_path": "progress/task_000/context_refs.npz",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class Predictor:
        def predict(self, observation, task_id):
            assert observation["retrieval_episode_id"] == 3
            assert task_id == 0
            return 0.9

    policy = RiclLiberoPolicy.__new__(RiclLiberoPolicy)
    policy._corpus = LiberoRiclCorpus(corpus_dir)
    policy._knn_k = 2
    policy._progress_predictor = Predictor()
    policy._progress_rng = np.random.default_rng(7)
    policy._use_action_interpolation = False
    policy._action_horizon = 3
    try:
        result = policy.retrieve(
            {
                "query_top_image": np.zeros((2, 2, 3), dtype=np.uint8),
                "query_wrist_image": np.zeros((2, 2, 3), dtype=np.uint8),
                "query_state": np.zeros(8, dtype=np.float32),
                "query_prompt": "test task",
                "retrieval_task_id": np.asarray(0, dtype=np.int32),
                "retrieval_episode_id": np.asarray(3, dtype=np.int32),
                "vfe_query_proprio": np.zeros(9, dtype=np.float32),
            }
        )
        assert "retrieval_task_id" not in result
        assert "retrieval_episode_id" not in result
        assert "vfe_query_proprio" not in result
        assert result["retrieved_0_actions"].shape == (3, 7)
        assert result["retrieved_1_actions"].shape == (3, 7)
        assert result["retrieved_0_state"].shape == (8,)
    finally:
        policy._corpus.close()
