from typing import Any, Dict, List, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots.svh import SVHHandRight
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilderDEXYCB
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

# TODO:Randomly modify items in the scene.
_YCB_CLASSES = {
    1: '002_master_chef_can',
    2: '003_cracker_box',
    3: '004_sugar_box',
    4: '005_tomato_soup_can',
    5: '006_mustard_bottle',
    6: '007_tuna_fish_can',
    7: '008_pudding_box',
    8: '009_gelatin_box',
    9: '010_potted_meat_can',
    10: '011_banana',
    11: '019_pitcher_base',
    12: '021_bleach_cleanser',
    13: '024_bowl',
    14: '025_mug',
    15: '035_power_drill',
    16: '036_wood_block',
    17: '037_scissors',
    18: '040_large_marker',
    19: '051_large_clamp',
    20: '052_extra_large_clamp',
    21: '061_foam_brick',
}


@register_env("DEX_YCB-v1", max_episode_steps=80)
class DexYCBEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["svh"]

    agent: Union[SVHHandRight]

    def __init__(
            self,
            *args,
            robot_uids="svh",
            robot_init_qpos_noise=0,
            num_envs=1,
            reconfiguration_freq=None,
            **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise

        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1, 2, 1], [-1.0, -2, -1])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilderDEXYCB(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self._objs: List[Actor] = []
        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:002_master_chef_can"
        )
        builder.set_scene_idxs([0])
        self._objs.append(builder.build(name=f"002_master_chef_can-0"))
        self.obj = Actor.merge(self._objs, name="ycb_object")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            xyz = torch.tensor([0.5962, 0.3497, 0.0496])
            qs = torch.tensor([0.80665, -0.0001985, 0.000058281, 0.59102])
            self.obj.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            # grasp init pos refer to dex_ycb data
            qpos = np.array(
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
            )
            self.agent.reset(qpos)
            self.agent.robot.set_root_pose(sapien.Pose([0, 0, 0], q=[1, 0, 0, 0]))

    def evaluate(self):
        height = self.obj.pose.p[0][2]
        threshold = 0.1 + 0.0496
        success = height.clone().detach() > threshold

        k1 = 1
        tip_min_dis = 0.20
        palm_min_dis = 0.20
        # tip reach
        tip_to_obj_dist = torch.sum(
            torch.stack(
                [torch.linalg.norm(self.obj.pose.p - link.pose.p)
                 for link in self.agent.tip_links]
            )
        )
        if tip_to_obj_dist < tip_min_dis:
            tip_reaching_reward = torch.ones(1)
        else:
            tip_reaching_reward = (1 / (1 + torch.exp(k1 * (tip_to_obj_dist - tip_min_dis))) * 2)

        # palm reach
        palm_to_obj_dist = torch.sum(
            torch.stack(
                [torch.linalg.norm(self.obj.pose.p - link.pose.p)
                 for link in self.agent.palm_links]
            )
        )
        if palm_to_obj_dist < palm_min_dis:
            palm_reaching_reward = torch.ones(1)
        else:
            palm_reaching_reward = (1 / (1 + torch.exp(k1 * (palm_to_obj_dist - palm_min_dis))) * 2)

        return dict(
            success=success,
            tip_to_obj_dist=tip_to_obj_dist,
            tip_reaching_reward=tip_reaching_reward,
            palm_to_obj_dist=palm_to_obj_dist,
            palm_reaching_reward=palm_reaching_reward,
            )

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tip_to_obj_dist=info["tip_to_obj_dist"].reshape(1, 1),
            tip_reaching_reward=info["tip_reaching_reward"].reshape(1, 1),
            palm_to_obj_dist=info["palm_to_obj_dist"].reshape(1, 1),
            palm_reaching_reward=info["palm_reaching_reward"].reshape(1, 1),
            obj_pose=self.obj.pose.p.reshape(-1, 3),
        )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # can's r = 0.03609339892864227
        # reach grasp pose reward
        lamda_1_1 = 1
        lamda_1_2 = 1
        k1 = 1
        tip_min_dis = 0.20
        palm_min_dis = 0.20
        # tip reach
        tip_to_obj_dist = torch.sum(
            torch.stack(
                [torch.linalg.norm(self.obj.pose.p - link.pose.p)
                 for link in self.agent.tip_links]
            )
        )
        if tip_to_obj_dist < tip_min_dis:
            tip_reaching_reward = torch.ones(1)
        else:
            tip_reaching_reward = (1 / (1 + torch.exp(k1 * (tip_to_obj_dist - tip_min_dis))) * 2)
        reward = tip_reaching_reward.view(1).clone() * lamda_1_1
        # print("tip dis", tip_to_obj_dist)
        # print("tip distance reward", tip_reaching_reward)

        # palm reach
        palm_to_obj_dist = torch.sum(
            torch.stack(
                [torch.linalg.norm(self.obj.pose.p - link.pose.p)
                 for link in self.agent.palm_links]
            )
        )
        if palm_to_obj_dist < palm_min_dis:
            palm_reaching_reward = torch.ones(1)
        else:
            palm_reaching_reward = (1 / (1 + torch.exp(k1 * (palm_to_obj_dist - palm_min_dis))) * 2)    # 非常激烈的奖励，使得手掌快速靠近物体
        reward += palm_reaching_reward.view(1).clone() * lamda_1_2
        # print("palm dis", palm_to_obj_dist)
        # print("palm distance reward", palm_reaching_reward)

        # contact reward
        lamda_2 = 2
        is_contact = torch.zeros(1)
        contact_reward = torch.zeros(1)
        for link in self.agent.tip_links:
            force = self.scene.get_pairwise_contact_forces(
                link, self.obj
            )
            if force.ne(0).any():
                contact_reward += torch.tensor(0.2)
        if contact_reward >= 0.6:
            is_contact = torch.ones(1)
        reward += contact_reward.clone() * lamda_2
        # print("contact reward", contact_reward)
        # print("contact?", is_contact)

        # grasp force reward
        lamda_3 = 1
        min_force = 0.5
        tip_contact_forces = []
        for link in self.agent.tip_links:
            force = self.scene.get_pairwise_contact_forces(
                link, self.obj
            )
            tip_contact_forces.append(force)
        forces = [torch.linalg.norm(force, axis=1) for force in tip_contact_forces]
        force_sum = torch.sum(torch.stack(forces))
        # print("f", force_sum)
        if force_sum > min_force:
            grasp_force_reward = torch.ones(1)
        else:
            grasp_force_reward = torch.tanh(torch.log(torch.tensor(min_force))-torch.log(min_force - force_sum))
        reward += grasp_force_reward.clone() * is_contact.clone() * lamda_3
        # print("grasp force reward", grasp_force_reward)

        # TODO:is_grasped

        # lift reward
        lamda_4 = 1
        lift_reward = torch.zeros(1)
        if self.obj.pose.p[0][2] > 0.0496:
            lift_reward = torch.ones(1)
        reward += lift_reward.clone() * is_contact.clone() * lamda_4
        # print("lift_reward", lift_reward)

        # lift_up reward
        lamda_5 = 1
        height = 0.35
        threshold = height + 0.0496
        k2 = 10
        if self.obj.pose.p[0][2] > threshold:
            lift_up_reward = torch.ones(1)
        else:
            lift_up_reward = 1 / (1 + torch.exp(k2 * (threshold - self.obj.pose.p[0][2]))) * 2
        reward += lift_up_reward.clone() * is_contact.clone() * lamda_5
        # print("obj_z", self.obj.pose.p[0][2])
        # print("lift up reward", lift_up_reward)

        # static reward
        # static_reward = 1 - torch.tanh(
        #     5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        # )
        # print(5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1))
        # reward += static_reward
        # print("static reward", static_reward)
        return reward

    def compute_normalized_dense_reward(
            self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 6
