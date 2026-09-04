"""Experiment 2 evaluation -- what did the standing policy actually learn?

Three questions the training curve cannot answer:

  1. Does it stand?  ep_rew_mean says nothing about whether the episode ended
     at the 20 s cap or on a fall.
  2. Can it take a push?  Experiment 1's PD controller survived 0.15 m/s and
     fell at 0.20 m/s.  The push protocol here is deliberately IDENTICAL --
     a base velocity impulse injected at t = 1 s -- so the two numbers are
     directly comparable.  Anything else would be comparing different tests.
  3. What is it being paid for?  The reward is a sum of terms; without
     decomposing it, a high score could just be an alive bonus being farmed.

Randomization is OFF (dr=False, push=False) so the only disturbance is the one
we inject.
"""
import argparse
import os
from collections import defaultdict

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from r1_walk_env import R1WalkEnv

HERE = os.path.dirname(os.path.abspath(__file__))


def make(name, ckpt, vn):
    d = os.path.join(HERE, "runs", name)
    model = PPO.load(os.path.join(d, ckpt), device="cpu")
    raw = R1WalkEnv(push=False, dr=False, fixed_cmd=[0.0, 0.0, 0.0], seed=None)
    venv = DummyVecEnv([lambda: raw])
    venv = VecNormalize.load(os.path.join(d, vn), venv)
    venv.training = False
    venv.norm_reward = False
    return model, venv, raw


def episode(model, venv, raw, push=0.0, push_at_s=1.0):
    """One episode.  Returns metrics plus the per-term reward totals."""
    obs = venv.reset()
    terms = defaultdict(float)
    h0 = raw.data.xpos[1][2]
    min_h = h0
    max_tilt, steps, total_r = 0.0, 0, 0.0
    push_step = int(push_at_s / raw.dt)
    fell = False

    while True:
        if push and steps == push_step:
            raw.data.qvel[0] += push          # same protocol as experiment 1

        # Read state BEFORE step().  DummyVecEnv auto-resets on done, so reading
        # raw.data after the loop measures a FRESH episode -- which reported a
        # -0.38 cm "height drop" for episodes that had actually toppled.  Fixed
        # 2026-08-14; the tilt figure was never affected (accumulated in-loop).
        min_h = min(min_h, raw.data.xpos[1][2])
        zaxis = raw.data.xmat[1].reshape(3, 3)[:, 2]
        max_tilt = max(max_tilt, np.degrees(np.arccos(np.clip(zaxis[2], -1, 1))))

        act, _ = model.predict(obs, deterministic=True)
        obs, r, done, infos = venv.step(act)
        info = infos[0]
        total_r += float(r[0])
        steps += 1

        for k, v in info.get("rew", {}).items():
            terms[k] += float(v)

        if done[0]:
            # SB3 reports truncation in the info dict; anything else is a real fall
            fell = not info.get("TimeLimit.truncated", False)
            break

    return {"steps": steps, "seconds": steps * raw.dt, "fell": fell,
            "max_tilt": max_tilt, "height_drop_cm": 100 * (h0 - min_h),
            "reward": total_r, "terms": dict(terms)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="exp2_stand")
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--vn", default="vecnormalize_best.pkl")
    ap.add_argument("--episodes", type=int, default=10)
    args = ap.parse_args()

    model, venv, raw = make(args.name, args.ckpt, args.vn)
    cap = raw.max_steps * raw.dt
    print(f"=== Experiment 2 eval: {args.name}/{args.ckpt} "
          f"(episode cap {cap:.0f}s, no DR, no random push) ===\n")

    # ---------- 1. undisturbed standing ----------
    rows = [episode(model, venv, raw) for _ in range(args.episodes)]
    surv = np.array([r["seconds"] for r in rows])
    fell = np.array([r["fell"] for r in rows])
    print(f"--- undisturbed, {args.episodes} episodes ---")
    print(f"  survival        : {surv.mean():5.2f}s +/- {surv.std():4.2f}  "
          f"(min {surv.min():.2f}, max {surv.max():.2f})")
    print(f"  reached the cap : {(~fell).sum()}/{len(rows)}  ({100*(~fell).mean():.0f}%)")
    print(f"  max tilt        : {np.mean([r['max_tilt'] for r in rows]):5.2f} deg")
    print(f"  height drop     : {np.mean([r['height_drop_cm'] for r in rows]):5.2f} cm")
    print(f"  episode reward  : {np.mean([r['reward'] for r in rows]):8.1f}")

    # ---------- 2. what is it paid for ----------
    tot = defaultdict(float)
    for r in rows:
        for k, v in r["terms"].items():
            tot[k] += v / len(rows)
    print(f"\n--- reward decomposition (mean per episode) ---")
    gross = sum(abs(v) for v in tot.values()) or 1.0
    for k, v in sorted(tot.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {k:18s} {v:9.1f}   ({100*abs(v)/gross:4.1f}% of gross)")

    # ---------- 3. push threshold, same protocol as experiment 1 ----------
    print(f"\n--- push rejection (impulse at t=1s; PD baseline was 0.15 ok / 0.20 fail) ---")
    print(f"  {'push m/s':>9}{'survived':>10}{'mean s':>9}{'max tilt':>10}")
    for p in [0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]:
        trials = [episode(model, venv, raw, push=p) for _ in range(5)]
        ok = sum(not t["fell"] for t in trials)
        print(f"  {p:>9}{ok:>7}/5{np.mean([t['seconds'] for t in trials]):>9.2f}"
              f"{np.mean([t['max_tilt'] for t in trials]):>10.1f}")


if __name__ == "__main__":
    main()
