from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass
from isaaclab.utils.buffers import DelayBuffer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DelayedJointPositionAction(JointPositionAction):
    """Joint position targets reach the actuators `lag` physics steps late. One lag per env, uniform in
    [min_lag, max_lag], resampled every `update_period` physics steps with a per-env random phase
    (same scheme as the mjlab R1 delay task)."""

    cfg: "DelayedJointPositionActionCfg"

    def __init__(self, cfg: "DelayedJointPositionActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._buf = DelayBuffer(cfg.max_lag, self.num_envs, self.device)
        self._phase = torch.randint(0, cfg.update_period, (self.num_envs,), device=self.device)
        self._resample(torch.arange(self.num_envs, device=self.device))

    def _resample(self, ids: torch.Tensor):
        lags = torch.randint(self.cfg.min_lag, self.cfg.max_lag + 1, (len(ids),), device=self.device, dtype=torch.int)
        self._buf.set_time_lag(lags, ids)

    def apply_actions(self):
        self._phase += 1
        due = (self._phase >= self.cfg.update_period).nonzero(as_tuple=True)[0]
        if len(due) > 0:
            self._phase[due] = 0
            self._resample(due)
        delayed = self._buf.compute(self.processed_actions)
        self._asset.set_joint_position_target(delayed, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else torch.as_tensor(env_ids, device=self.device)
        self._buf.reset(ids)
        self._resample(ids)


@configclass
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = DelayedJointPositionAction
    min_lag: int = 0
    max_lag: int = 10
    update_period: int = 200
