"""Backward-compatible aliases for the generic LeRobot RICL policy transforms."""

from openpi.policies.lerobot_ricl_policy import PadRiclLeRobotStates
from openpi.policies.lerobot_ricl_policy import RiclLeRobotInputs
from openpi.policies.lerobot_ricl_policy import RiclLeRobotOutputs

RiclPnpEggplantInputs = RiclLeRobotInputs
PadRiclPnpEggplantStates = PadRiclLeRobotStates
RiclPnpEggplantOutputs = RiclLeRobotOutputs
