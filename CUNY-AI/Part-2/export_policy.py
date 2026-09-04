"""Export a trained PPO policy to a plain .npz so it can run with NumPy alone.

The policy is a 45 -> 256 -> 256 -> 12 MLP with tanh activations, plus the
VecNormalize observation statistics. None of that needs PyTorch to evaluate, so
exporting lets the student labs keep the same light dependency as Lab 1
(mujoco + numpy) instead of pulling in a ~2.5 GB torch install.

    python export_policy.py runs/exp2_stand best vecnormalize_best.pkl out.npz
"""
import sys, os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym


def export(run_dir, ckpt, vn_name, out):
    model = PPO.load(os.path.join(run_dir, ckpt), device="cpu")
    sd = model.policy.state_dict()
    g = lambda k: sd[k].cpu().numpy()

    class _Stub(gym.Env):
        observation_space = model.observation_space
        action_space = model.action_space
        def reset(self, **kw): return np.zeros(model.observation_space.shape, np.float32), {}
        def step(self, a): return np.zeros(model.observation_space.shape, np.float32), 0.0, False, False, {}

    vn = VecNormalize.load(os.path.join(run_dir, vn_name), DummyVecEnv([lambda: _Stub()]))

    np.savez(
        out,
        w0=g("mlp_extractor.policy_net.0.weight"), b0=g("mlp_extractor.policy_net.0.bias"),
        w1=g("mlp_extractor.policy_net.2.weight"), b1=g("mlp_extractor.policy_net.2.bias"),
        w2=g("action_net.weight"), b2=g("action_net.bias"),
        obs_mean=vn.obs_rms.mean.astype(np.float64),
        obs_var=vn.obs_rms.var.astype(np.float64),
        clip_obs=np.float64(vn.clip_obs), epsilon=np.float64(vn.epsilon),
        act_low=model.action_space.low.astype(np.float64),
        act_high=model.action_space.high.astype(np.float64),
    )
    print(f"wrote {out}  ({os.path.getsize(out)/1024:.0f} KB)")
    return model, vn


def numpy_policy(npz):
    """The whole inference path, in NumPy. This is what the labs will ship."""
    d = np.load(npz)
    def act(obs):
        o = np.clip((obs - d["obs_mean"]) / np.sqrt(d["obs_var"] + d["epsilon"]),
                    -d["clip_obs"], d["clip_obs"])
        h = np.tanh(d["w0"] @ o + d["b0"])
        h = np.tanh(d["w1"] @ h + d["b1"])
        # SB3's predict() clips the action to the action space -- without this the
        # numpy policy disagrees by up to 4.6 on saturated joints.
        return np.clip(d["w2"] @ h + d["b2"], d["act_low"], d["act_high"])
    return act


if __name__ == "__main__":
    run_dir, ckpt, vn_name, out = sys.argv[1:5]
    model, vn = export(run_dir, ckpt, vn_name, out)

    # verify: numpy must reproduce sb3's deterministic action exactly
    act = numpy_policy(out)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        raw = rng.normal(0, 1.5, size=model.observation_space.shape[0]).astype(np.float64)
        norm = np.clip((raw - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + vn.epsilon),
                       -vn.clip_obs, vn.clip_obs).astype(np.float32)
        ref, _ = model.predict(norm[None, :], deterministic=True)
        worst = max(worst, float(np.abs(act(raw) - ref[0]).max()))
    print(f"max |numpy - sb3| over 200 random observations: {worst:.3e}")
    print("MATCH" if worst < 1e-5 else "MISMATCH — do not ship")
