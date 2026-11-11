from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


aloha_agilex_config = {
    # Video: current frame only; keys must match "video" entries in meta/modality.json
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cam_high", "cam_left_wrist", "cam_right_wrist"],
    ),
    # State: current proprioceptive reading; keys must match "state" entries in meta/modality.json
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_joint",  # left arm joint positions
            "right_joint",  # right arm joint positions
        ],
    ),
    # Action: 16-step prediction horizon; one ActionConfig per modality key
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),  # predict 16 future steps
        modality_keys=[
            "left_joint",
            "right_joint",
        ],
        action_configs=[
            # left_joint: RELATIVE = delta from current state (better generalization)
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,  # joint-space, not end-effector
                format=ActionFormat.DEFAULT,
            ),
            # right_joint: RELATIVE = delta from current state (better generalization)
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,  # joint-space, not end-effector
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    # Language: task instruction from annotation field in the dataset
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(aloha_agilex_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
