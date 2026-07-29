import json

import h5py
import numpy as np

from openpi.policies.libero_retrieval import LiberoRiclCorpus, libero_action_chunk


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
