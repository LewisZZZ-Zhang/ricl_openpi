"""Client for the localhost VFE progress service."""

from __future__ import annotations

import pickle
import socket
import struct
from typing import Any


MAX_MESSAGE_BYTES = 128 * 1024 * 1024


def _recv_exact(sock: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("VFE progress server closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class VfeProgressClient:
    def __init__(self, host: str, port: int, timeout_seconds: float = 600.0):
        self.host = str(host)
        self.port = int(port)
        self.timeout_seconds = float(timeout_seconds)
        if not 0 < self.port <= 65535:
            raise ValueError(f"Invalid VFE progress server port: {self.port}")

    def _request(self, request: dict[str, Any]) -> dict[str, Any]:
        payload = pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL)
        if len(payload) > MAX_MESSAGE_BYTES:
            raise ValueError(f"VFE progress request is too large: {len(payload)} bytes")
        with socket.create_connection((self.host, self.port), timeout=self.timeout_seconds) as sock:
            sock.settimeout(self.timeout_seconds)
            sock.sendall(struct.pack("!Q", len(payload)) + payload)
            response_size = struct.unpack("!Q", _recv_exact(sock, 8))[0]
            if response_size > MAX_MESSAGE_BYTES:
                raise ValueError(f"VFE progress response is too large: {response_size} bytes")
            response = pickle.loads(_recv_exact(sock, response_size))
        if not response.get("ok"):
            raise RuntimeError(f"VFE progress inference failed: {response.get('error', 'unknown error')}")
        return response

    def health(self) -> bool:
        return bool(self._request({"type": "health"})["ok"])

    def predict(self, observation: dict[str, Any], task_id: int) -> float:
        response = self._request({"type": "predict", "task_id": int(task_id), "observation": observation})
        return float(response["progress"])
