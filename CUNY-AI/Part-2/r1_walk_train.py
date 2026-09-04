"""PPO training for the R1 walking policy (single-env, CPU -- GTX 1650 laptop).

Usage:
  python3 r1_walk_train.py --steps 10000000 --name run1
  python3 r1_walk_train.py --steps 30000    --name smoke   # smoke test

Saves checkpoints (model + VecNormalize stats) under runs/<name>/ every
--save-freq steps, plus tensorboard logs. Resumable-ish: latest checkpoint +
vecnormalize can be loaded by the eval / sim2sim scripts.
"""
import os
import argparse
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from r1_walk_env import R1WalkEnv

HERE = os.path.dirname(os.path.abspath(__file__))

try:
    import tensorboard  # noqa: F401
    _HAS_TB = True
except ImportError:
    _HAS_TB = False


class SaveCallback(BaseCallback):
    """Save model + VecNormalize stats periodically and track best ep reward."""
    def __init__(self, save_freq, save_dir, vecnorm):
        super().__init__()
        self.save_freq = save_freq
        self.save_dir = save_dir
        self.vecnorm = vecnorm
        self.best = -np.inf

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            self.model.save(os.path.join(self.save_dir, "latest"))
            self.vecnorm.save(os.path.join(self.save_dir, "vecnormalize.pkl"))
            # keep a numbered snapshot too, so a good-behaviour policy is never
            # overwritten by a later higher-reward-but-worse one (lesson from walk1)
            k = self.num_timesteps // 1000
            self.model.save(os.path.join(self.save_dir, f"ckpt_{k}k"))
            self.vecnorm.save(os.path.join(self.save_dir, f"vn_{k}k.pkl"))
            # ep_rew_mean from the monitor
            if len(self.model.ep_info_buffer) > 0:
                mean_r = np.mean([e["r"] for e in self.model.ep_info_buffer])
                mean_l = np.mean([e["l"] for e in self.model.ep_info_buffer])
                if mean_r > self.best:
                    self.best = mean_r
                    self.model.save(os.path.join(self.save_dir, "best"))
                    self.vecnorm.save(os.path.join(self.save_dir, "vecnormalize_best.pkl"))
                print(f"[{self.num_timesteps}] ep_rew_mean={mean_r:.2f} "
                      f"ep_len_mean={mean_l:.0f} best={self.best:.2f}", flush=True)
        return True


def make_env(seed, jitter_dr=False, fixed_cmd=None, zero_rew=None):
    def _f():
        from stable_baselines3.common.monitor import Monitor
        return Monitor(R1WalkEnv(seed=seed, jitter_dr=jitter_dr, fixed_cmd=fixed_cmd,
                                 zero_rew=zero_rew))
    return _f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10_000_000)
    ap.add_argument("--name", type=str, default="run1")
    ap.add_argument("--save-freq", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-from", default=None, help="warm-start policy weights from this .zip")
    ap.add_argument("--init-vn", default=None, help="load obs-norm stats from this .pkl")
    ap.add_argument("--n-envs", type=int, default=1,
                    help="parallel envs via SubprocVecEnv (each gets seed+rank); "
                         "total rollout size stays 4096 (n_steps = 4096 // n_envs) "
                         "so parallelism/batch-diversity is the ONLY changed variable")
    ap.add_argument("--fixed-cmd", type=float, nargs=3, default=None,
                    help="hold the velocity command fixed, e.g. --fixed-cmd 0 0 0 "
                         "turns the walk task into a standing task")
    ap.add_argument("--jitter-dr", action="store_true",
                    help="OPTION 3: per-step intermittent latency spikes (JITTER_DR in "
                         "r1_walk_env.py) instead of/on top of the per-episode-constant "
                         "0..MAX_LATENCY draw already in dr=True")
    ap.add_argument("--zero-rew", type=str, default=None,
                    help="EXPERIMENT 3: comma-separated reward terms to drop from "
                         "the training objective, e.g. --zero-rew heading,lateral. "
                         "The unmodified reward is still logged as info['full_rew'] "
                         "so every ablation can be scored on one fixed yardstick.")
    args = ap.parse_args()
    zero_rew = [s.strip() for s in args.zero_rew.split(",")] if args.zero_rew else None

    save_dir = os.path.join(HERE, "runs", args.name)
    os.makedirs(save_dir, exist_ok=True)

    fns = [make_env(args.seed + i, jitter_dr=args.jitter_dr, fixed_cmd=args.fixed_cmd,
                    zero_rew=zero_rew)
           for i in range(args.n_envs)]
    venv = SubprocVecEnv(fns) if args.n_envs > 1 else DummyVecEnv(fns)
    vn_path = os.path.join(save_dir, "vecnormalize.pkl")
    if args.resume and os.path.exists(vn_path):
        venv = VecNormalize.load(vn_path, venv)
        venv.training = True
    elif args.init_vn and os.path.exists(args.init_vn):
        # warm start: carry obs-normalization stats from the source run;
        # reward norm re-adapts to the (changed) reward on its own.
        venv = VecNormalize.load(args.init_vn, venv)
        venv.training = True
        print("loaded obs-norm stats from", args.init_vn, flush=True)
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True,
                            clip_obs=10.0, clip_reward=10.0, gamma=0.99)

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    # keep the TOTAL rollout at 4096 steps regardless of n_envs, so multi-env
    # runs change batch diversity (independent trajectories), not batch size
    n_steps = max(4096 // args.n_envs, 256)
    override = {"n_steps": n_steps}
    model_path = os.path.join(save_dir, "latest.zip")
    if args.resume and os.path.exists(model_path):
        model = PPO.load(model_path, env=venv, device="cpu", custom_objects=override)
        print("resumed from", model_path, flush=True)
    elif args.init_from and os.path.exists(args.init_from):
        model = PPO.load(args.init_from, env=venv, device="cpu", custom_objects=override)
        print("warm-started policy weights from", args.init_from, flush=True)
    else:
        model = PPO(
            "MlpPolicy", venv, device="cpu", verbose=0,
            n_steps=n_steps, batch_size=256, n_epochs=5,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.004, learning_rate=3e-4, vf_coef=0.5,
            max_grad_norm=1.0, policy_kwargs=policy_kwargs,
            # Only log to tensorboard if it is actually installed. SB3 raises
            # ImportError at learn() time otherwise, which killed training on a
            # clean student install where tensorboard is not a dependency.
            tensorboard_log=(os.path.join(save_dir, "tb") if _HAS_TB else None),
        )

    cb = SaveCallback(args.save_freq, save_dir, venv)
    print(f"training '{args.name}' for {args.steps:,} steps -> {save_dir}", flush=True)
    print(f"  seed={args.seed}  n_envs={args.n_envs}  n_steps={n_steps}  "
          f"zero_rew={zero_rew or 'none (baseline)'}", flush=True)
    model.learn(total_timesteps=args.steps, callback=cb,
                reset_num_timesteps=not args.resume, progress_bar=False)
    model.save(os.path.join(save_dir, "latest"))
    venv.save(vn_path)
    print("done. saved to", save_dir, flush=True)


if __name__ == "__main__":
    main()
