import os, sys; sys.path.insert(0, os.path.expanduser("~/unitree_rl_mjlab"))
import torch
import src.tasks  # registers
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
for task in ["Unitree-R1-Flat-Delay-FT", "Unitree-R1-Flat-FT"]:
    cfg = load_env_cfg(task)
    cfg.scene.num_envs = 64
    print(task, "curriculum:", list(cfg.curriculum.keys()), "x range", cfg.commands["twist"].ranges.lin_vel_x)
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    env.reset()
    robot = env.scene["robot"]
    acts = robot._actuators
    print(" actuators:", [type(a).__name__ for a in acts])
    seen = set()
    for t in range(450):
        a = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device="cuda:0")
        env.step(a)
        if hasattr(acts[0], "_delay_buffers"):
            lags = [a_._delay_buffers["position"].current_lags.clone() for a_ in acts]
            assert all(torch.equal(lags[0], l) for l in lags), "lags differ across groups!"
            seen.update(lags[0].tolist())
    print(" lags seen:", sorted(seen))
    env.close()
