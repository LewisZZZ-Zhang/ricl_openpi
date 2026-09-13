"""Serve a trained RICL-LIBERO policy over the OpenPI websocket protocol."""

from __future__ import annotations

import dataclasses
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.policies.libero_retrieval import LiberoRiclCorpus
from openpi.policies.vfe_progress_client import VfeProgressClient
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    config: str = "pi0_fast_libero_ricl"
    checkpoint_dir: str = tyro.MISSING
    corpus_dir: str = tyro.MISSING
    port: int = 8000
    record: bool = False
    vfe_host: str = "127.0.0.1"
    vfe_port: int | None = None


def main(args: Args) -> None:
    corpus = LiberoRiclCorpus(args.corpus_dir)
    retrieval_backend = corpus.retrieval_backend
    corpus.close()
    progress_predictor = None
    if retrieval_backend == "progress":
        if args.vfe_port is None:
            raise ValueError("A progress retrieval corpus requires --vfe-port for online inference")
        progress_predictor = VfeProgressClient(args.vfe_host, args.vfe_port)
    policy = _policy_config.create_trained_libero_ricl_policy(
        _config.get_config(args.config),
        args.checkpoint_dir,
        args.corpus_dir,
        progress_predictor=progress_predictor,
    )
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    logging.info("Creating RICL-LIBERO server (host: %s, ip: %s)", hostname, socket.gethostbyname(hostname))
    websocket_policy_server.WebsocketPolicyServer(policy=policy, host="0.0.0.0", port=args.port).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
