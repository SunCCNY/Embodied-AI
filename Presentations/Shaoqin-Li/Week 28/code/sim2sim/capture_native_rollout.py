"""Run the trained R1 tracking policy in mjlab's own native env (single instance,
no domain randomization at eval) and log every applied action + resulting joint
state, so we can replay the exact same action sequence open-loop through plain
MuJoCo and see if physics alone reproduces the same trajectory.
"""
import numpy as np
import torch

from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.envs import ManagerBasedRlEnv

TASK = "Mjlab-Tracking-Flat-Unitree-R1"
CHECKPOINT = "logs/rsl_rl/r1_tracking/2026-09-23_19-03-01_r1_tennis_run1/model_2999.pt"
MOTION_FILE = "/tmp/motion_tennis.npz"

env_cfg = load_env_cfg(TASK, play=True)
agent_cfg = load_rl_cfg(TASK)

motion_cmd = env_cfg.commands["motion"]
motion_cmd.motion_file = MOTION_FILE
motion_cmd.sampling_mode = "start"
env_cfg.scene.num_envs = 1
env_cfg.observations["actor"].enable_corruption = False
env_cfg.events.pop("push_robot", None)
motion_cmd.pose_range = {}
motion_cmd.velocity_range = {}

device = "cuda:0" if torch.cuda.is_available() else "cpu"
env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

from dataclasses import asdict
runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
runner.load(CHECKPOINT, map_location=device)
policy = runner.get_inference_policy(device=device)

command = env.unwrapped.command_manager.get_term("motion")
robot = env.unwrapped.scene["robot"]

obs = env.get_observations()

actions_log = []
joint_pos_log = []
joint_vel_log = []
root_pos_log = []
root_quat_log = []
obs_log = []

n_steps = 519  # matches the motion length at 50fps -> but env runs at 125Hz;
                # just capture a fixed generous number of control steps.
n_steps = 660

for t in range(n_steps):
    obs_log.append(obs["actor"].cpu().numpy().copy())
    with torch.no_grad():
        action = policy(obs)
    actions_log.append(action.cpu().numpy().copy())
    joint_pos_log.append(robot.data.joint_pos.cpu().numpy().copy())
    joint_vel_log.append(robot.data.joint_vel.cpu().numpy().copy())
    root_pos_log.append(robot.data.root_link_pos_w.cpu().numpy().copy())
    root_quat_log.append(robot.data.root_link_quat_w.cpu().numpy().copy())
    obs, _, dones, _ = env.step(action)
    if dones.any():
        print(f"episode ended at step {t}")
        break

np.savez(
    "/tmp/native_rollout.npz",
    actions=np.concatenate(actions_log, axis=0),
    joint_pos=np.concatenate(joint_pos_log, axis=0),
    joint_vel=np.concatenate(joint_vel_log, axis=0),
    root_pos=np.concatenate(root_pos_log, axis=0),
    root_quat=np.concatenate(root_quat_log, axis=0),
    obs=np.concatenate(obs_log, axis=0),
)
print("saved /tmp/native_rollout.npz, steps:", len(actions_log))
print("action range overall:", np.min(actions_log), np.max(actions_log))
