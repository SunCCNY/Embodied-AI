from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_r1_flat_env_cfg,
  unitree_r1_flat_finetune_env_cfg,
  unitree_r1_rough_env_cfg,
)
from .rl_cfg import unitree_r1_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-R1-Rough",
  env_cfg=unitree_r1_rough_env_cfg(),
  play_env_cfg=unitree_r1_rough_env_cfg(play=True),
  rl_cfg=unitree_r1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-R1-Flat",
  env_cfg=unitree_r1_flat_env_cfg(),
  play_env_cfg=unitree_r1_flat_env_cfg(play=True),
  rl_cfg=unitree_r1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Fine-tune tasks (resume from a finished Unitree-R1-Flat checkpoint).
# -Delay-FT: 0-50 ms command delay randomization; -FT: matched control, no delay.
register_mjlab_task(
  task_id="Unitree-R1-Flat-Delay-FT",
  env_cfg=unitree_r1_flat_finetune_env_cfg(action_delay_max_lag=10),
  play_env_cfg=unitree_r1_flat_finetune_env_cfg(play=True),
  rl_cfg=unitree_r1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-R1-Flat-FT",
  env_cfg=unitree_r1_flat_finetune_env_cfg(),
  play_env_cfg=unitree_r1_flat_finetune_env_cfg(play=True),
  rl_cfg=unitree_r1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
