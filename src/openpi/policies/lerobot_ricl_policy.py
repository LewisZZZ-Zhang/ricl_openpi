"""Input and output transforms for LeRobot split-action RICL policies."""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RiclLeRobotInputs(transforms.DataTransformFn):
    """Map three LeRobot cameras and proprioception to RICL's block schema."""

    action_dim: int
    num_retrieved_observations: int
    state_dim: int = 14

    def __call__(self, data: dict) -> dict:
        prefixes = [f"retrieved_{i}_" for i in range(self.num_retrieved_observations)] + ["query_"]
        inputs: dict[str, object] = {}
        for prefix in prefixes:
            state = np.asarray(data[f"{prefix}state"], dtype=np.float32)
            if state.shape[-1] != self.state_dim:
                raise ValueError(f"Expected {self.state_dim}-D LeRobot RICL state, got {state.shape}")
            inputs[f"{prefix}state"] = state
            inputs[f"{prefix}image"] = {
                "base_0_rgb": _parse_image(data[f"{prefix}top_image"]),
                "base_1_rgb": _parse_image(data[f"{prefix}right_image"]),
                "wrist_0_rgb": _parse_image(data[f"{prefix}wrist_image"]),
            }
            inputs[f"{prefix}image_mask"] = {
                "base_0_rgb": np.True_,
                "base_1_rgb": np.True_,
                "wrist_0_rgb": np.True_,
            }
            inputs[f"{prefix}prompt"] = data[f"{prefix}prompt"]

        for prefix in prefixes:
            key = f"{prefix}actions"
            if key not in data:
                continue
            actions = np.asarray(data[key], dtype=np.float32)
            if actions.shape[-1] != self.action_dim:
                raise ValueError(f"Expected {self.action_dim}-D LeRobot RICL actions, got {actions.shape}")
            inputs[key] = actions
        if "exp_lamda_distances" in data:
            inputs["exp_lamda_distances"] = np.asarray(data["exp_lamda_distances"], dtype=np.float32)
        if "inference_time" in data:
            inputs["inference_time"] = data["inference_time"]
        return inputs


@dataclasses.dataclass(frozen=True)
class PadRiclLeRobotStates(transforms.DataTransformFn):
    """Pad normalized proprioceptive states to the model action/state width."""

    action_dim: int
    num_retrieved_observations: int

    def __call__(self, data: dict) -> dict:
        prefixes = [f"retrieved_{i}_" for i in range(self.num_retrieved_observations)] + ["query_"]
        for prefix in prefixes:
            data[f"{prefix}state"] = transforms.pad_to_dim(data[f"{prefix}state"], self.action_dim)
        return data


@dataclasses.dataclass(frozen=True)
class RiclLeRobotOutputs(transforms.DataTransformFn):
    action_dim: int = 16

    def __call__(self, data: dict) -> dict:
        return {"query_actions": np.asarray(data["query_actions"][:, : self.action_dim])}
