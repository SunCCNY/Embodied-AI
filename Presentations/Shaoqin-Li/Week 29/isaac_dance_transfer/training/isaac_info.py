"""Dump Isaac-side ground truth for the harness: joint/body order, defaults, gains, and one observation with the state that produced it."""

import argparse
import json
import sys

from isaaclab.app import AppLauncher

import cli_args  # noqa: F401

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Tracking-Flat-R1-v0")
parser.add_argument("--motion_file", type=str, required=True)
parser.add_argument("--out", type=str, default="isaac_info.json")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args_cli).app

import os

import gymnasium as gym
import torch

from isaaclab_tasks.utils.hydra import hydra_task_config

import whole_body_tracking.tasks  # noqa: F401


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = 1
    env_cfg.commands.motion.motion_file = os.path.abspath(args_cli.motion_file)
    env_cfg.commands.motion.pose_range = {}
    env_cfg.commands.motion.velocity_range = {}
    env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
    env_cfg.observations.policy.enable_corruption = False
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    obs, _ = env.reset()
    u = env.unwrapped
    robot = u.scene["robot"]
    cmd = u.command_manager.get_term("motion")
    cmd.time_steps[:] = 0
    # re-apply the reset state for frame 0 so state and command agree
    cmd._resample_command(torch.tensor([0], device=u.device))
    cmd.time_steps[:] = 0
    obs = u.observation_manager.compute()
    pol = obs["policy"][0].cpu().numpy().tolist()
    d = robot.data
    info = {
        "joint_names": list(robot.joint_names),
        "body_names": list(robot.body_names),
        "default_joint_pos": d.default_joint_pos[0].cpu().numpy().tolist(),
        "joint_stiffness": d.joint_stiffness[0].cpu().numpy().tolist(),
        "joint_damping": d.joint_damping[0].cpu().numpy().tolist(),
        "joint_armature": d.joint_armature[0].cpu().numpy().tolist(),
        "effort_limit": robot.root_physx_view.get_dof_max_forces()[0].cpu().numpy().tolist(),
        "action_scale": u.action_manager.get_term("joint_pos")._scale[0].cpu().numpy().tolist(),
        "action_joint_names": list(u.action_manager.get_term("joint_pos")._joint_names),
        "total_mass": float(robot.root_physx_view.get_masses()[0].sum()),
        "anchor_body_name": cmd.cfg.anchor_body_name,
        "cmd_body_names": list(cmd.cfg.body_names),
        "obs_terms": list(u.observation_manager.active_terms["policy"]),
        "obs_dims": [list(x) for x in u.observation_manager.group_obs_term_dim["policy"]],
        "obs0": pol,
        "state0": {
            "root_pos_w": d.root_link_pos_w[0].cpu().numpy().tolist(),
            "root_quat_w": d.root_link_quat_w[0].cpu().numpy().tolist(),
            "root_lin_vel_w": d.root_lin_vel_w[0].cpu().numpy().tolist(),
            "root_ang_vel_w": d.root_ang_vel_w[0].cpu().numpy().tolist(),
            "joint_pos": d.joint_pos[0].cpu().numpy().tolist(),
            "joint_vel": d.joint_vel[0].cpu().numpy().tolist(),
            "env_origin": u.scene.env_origins[0].cpu().numpy().tolist(),
            "time_step": int(cmd.time_steps[0]),
        },
    }
    with open(args_cli.out, "w") as f:
        json.dump(info, f, indent=1)
    print("WROTE", args_cli.out, flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
