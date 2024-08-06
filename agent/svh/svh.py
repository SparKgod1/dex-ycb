from copy import deepcopy

import numpy as np
import sapien
import torch

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.pose import vectorize_pose



@register_agent()
class SVHHandRight(BaseAgent):
    uid = "svh"
    urdf_path = f"../svh free root urdf/schunk_svh_hand_right_glb.urdf"
    urdf_config = dict(
        _materials=dict(
            tip=dict(static_friction=2.0, dynamic_friction=1.0, restitution=0.0)
        ),
        link={
            "right_hand_c": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
            "right_hand_t": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
            "right_hand_s": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
            "right_hand_r": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
            "right_hand_q": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
        },
    )
    keyframes = dict(
        start_up=Keyframe(
            qpos=np.array(
                [
                    0.31177029,
                    0.78997695,
                    0.09484848,
                    1.44094574,
                    -0.31045592,
                    -0.68973529,
                    0.24706309,
                    0.24706309,
                    0.00971482,
                    0.42615992,
                    0.00971482,
                    0.01942964,
                    0.40538633,
                    0.57583535,
                    0.43259919,
                    0.64781165,
                    0.64548451,
                    0.56715286,
                    0.71601611,
                    0.61745884,
                    0.88024647,
                    0.87708435,
                    0.59267474,
                    0.74852325,
                    0.92049501,
                    0.91856964
                ]
            ),
            pose=sapien.Pose([0, 0, 0], q=[1, 0, 0, 0]),
        ),
    )

    def __init__(self, *args, **kwargs):
        self.joint_names = [
            'dummy_x_translation_joint',
            'dummy_y_translation_joint',
            'dummy_z_translation_joint',
            'dummy_x_rotation_joint',
            'dummy_y_rotation_joint',
            'dummy_z_rotation_joint',
            'right_hand_Thumb_Opposition',
            'right_hand_j5',
            'right_hand_index_spread',
            'right_hand_Thumb_Flexion',
            'right_hand_ring_spread',
            'right_hand_Finger_Spread',
            'right_hand_Index_Finger_Proximal',
            'right_hand_Middle_Finger_Proximal',
            'right_hand_j3',
            'right_hand_Ring_Finger',
            'right_hand_Pinky',
            'right_hand_Index_Finger_Distal',
            'right_hand_Middle_Finger_Distal',
            'right_hand_j4',
            'right_hand_j12',
            'right_hand_j13',
            'right_hand_j14',
            'right_hand_j15',
            'right_hand_j16',
            'right_hand_j17'
        ]

        self.joint_stiffness = 4e2
        self.joint_damping = 5e1
        self.joint_force_limit = 4e2

        # Order: thumb finger, index finger, middle finger, ring finger
        self.tip_link_names = [
            "right_hand_c",  # "thtip"
            "right_hand_t",  # "fftip"
            "right_hand_s",  # "mftip"
            "right_hand_r",  # "rftip"
            "right_hand_q",  # "lftip"
        ]

        self.palm_link_names = [
            "right_hand_e1",
            "right_hand_e2"
        ]

        super().__init__(*args, **kwargs)

    def _after_init(self):
        self.tip_links = sapien_utils.get_objs_by_names(
            self.robot.get_links(), self.tip_link_names
        )
        self.palm_links = sapien_utils.get_objs_by_names(
            self.robot.get_links(), self.palm_link_names
        )

    @property
    def _controller_configs(self):
        joint_pos = PDJointPosControllerConfig(
            self.joint_names,
            None,
            None,
            self.joint_stiffness,
            self.joint_damping,
            self.joint_force_limit,
            normalize_action=False,
        )
        joint_delta_pos = PDJointPosControllerConfig(
            self.joint_names,
            -2,
            2,
            self.joint_stiffness,
            self.joint_damping,
            self.joint_force_limit,
            use_delta=True,
            normalize_action=True
        )
        joint_target_delta_pos = deepcopy(joint_delta_pos)
        joint_target_delta_pos.use_target = True

        controller_configs = dict(
            pd_joint_delta_pos=dict(arm=joint_delta_pos),
            pd_joint_pos=dict(arm=joint_pos),
            pd_joint_target_delta_pos=dict(arm=joint_target_delta_pos),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def get_proprioception(self):
        """
        Get the proprioceptive state of the agent.
        """
        obs = super().get_proprioception()
        obs.update(
            {
                "tip_poses": self.tip_poses.reshape(-1, len(self.tip_links) * 7),
                "palm_poses": self.palm_poses.reshape(-1, len(self.palm_links) * 7),
            }
        )

        return obs

    @property
    def tip_poses(self):
        """
        Get the tip pose for each of the finger, four fingers in total
        """
        tip_poses = [vectorize_pose(link.pose) for link in self.tip_links]
        return torch.stack(tip_poses, dim=-2)

    @property
    def palm_poses(self):
        """
        Get the tip pose for each of the finger, four fingers in total
        """
        tip_poses = [vectorize_pose(link.pose) for link in self.palm_links]
        return torch.stack(tip_poses, dim=-2)
