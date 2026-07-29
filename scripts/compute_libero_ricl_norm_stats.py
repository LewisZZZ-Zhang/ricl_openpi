"""Compute and save query/retrieved normalization statistics for a LIBERO RICL corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

from openpi.policies.libero_retrieval import LiberoRiclCorpus
from openpi.shared import normalize


def compute_norm_stats(corpus_dir: Path, output_dir: Path) -> None:
    corpus = LiberoRiclCorpus(corpus_dir)
    state_stats = normalize.RunningStats()
    action_stats = normalize.RunningStats()
    try:
        for task in corpus.metadata["tasks"]:
            if not task["is_train"]:
                continue
            # Query and context demos share the same training distribution. Including both
            # yields stable statistics while preserving the no-evaluation-data contract.
            demo_ids = list(task["context_demo_ids"]) + list(task["query_demo_ids"])
            for demo_id in demo_ids:
                states, actions = corpus.read_trajectory(int(task["task_id"]), str(demo_id))
                state_stats.update(states)
                action_stats.update(actions)
    finally:
        corpus.close()

    base_stats = {"state": state_stats.get_statistics(), "actions": action_stats.get_statistics()}
    num_retrieved = int(corpus.metadata["num_retrieved"])
    ricl_stats = {
        **{f"retrieved_{index}_{name}": stats for index in range(num_retrieved) for name, stats in base_stats.items()},
        **{f"query_{name}": stats for name, stats in base_stats.items()},
    }
    normalize.save(output_dir, ricl_stats)
    print(f"Wrote normalization statistics for {num_retrieved} retrieved blocks to {output_dir / 'norm_stats.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute normalization statistics for RICL-LIBERO training.")
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("assets/libero_ricl"))
    args = parser.parse_args()
    compute_norm_stats(args.corpus_dir, args.output_dir)


if __name__ == "__main__":
    main()
