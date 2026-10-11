"""Paper figures from the result files in this folder (run from results/: python make_figures.py).

fig_delay_seeds.png   survival vs added command delay, 3 training seeds per arm (seed study)
fig_stress.png        survival per stress scenario, tennis policy before / after delay training
fig_ablation.png      sim2sim joint error when one raw-MJCF setting is kept (floor/timestep ablation)
Colors: reference categorical slots 1-3 (blue, orange, aqua); every value is also printed, so color is never the only cue.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
for f in ("arial.ttf", "arialbd.ttf"):
    p = "/mnt/c/Windows/Fonts/" + f
    if os.path.exists(p):
        fm.fontManager.addfont(p)
plt.rcParams.update({"font.family": "Arial", "font.size": 11, "axes.edgecolor": "#8a8984", "axes.linewidth": 0.8,
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def surv(rs):
    return 100.0 * sum(r["survived"] for r in rs) / len(rs)


def load(p):
    return json.load(open(os.path.join(HERE, p)))


# ---------- 1. delay vs survival, seed study ----------
SC = [("train_DR", 0), ("train_DR+latency1", 20), ("train_DR+latency2", 40), ("train_DR+latency3", 60)]
fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=200)
for arm, color, label in (("control", BLUE, "+3,000 iterations, no delay"), ("delay", ORANGE, "+3,000 iterations, random 0–50 ms delay")):
    runs = [load(f"sweeps/seed_study/results_v5_dr_subset_seed_{arm}_s{s}.json") for s in (42, 43, 44)]
    xs = [ms for _, ms in SC]
    per = [[surv(r[k]) for k, _ in SC] for r in runs]
    mean = [sum(c) / 3 for c in zip(*per)]
    for p in per:
        ax.plot(xs, p, color=color, lw=1, alpha=0.35, marker="o", ms=4, zorder=2)
    ax.plot(xs, mean, color=color, lw=2.2, marker="o", ms=8, mec="white", mew=2, zorder=3, label=label)
    for x, m in zip(xs, mean):
        if x < 40:
            continue                                   # label only where the arms differ
        ax.annotate(f"{m:.0f}%", (x, m), textcoords="offset points", xytext=(9, 4 if arm == "delay" else -12),
                    fontsize=9.5, color=INK2)
ax.set_xticks([0, 20, 40, 60]); ax.set_xticklabels(["0 ms", "20 ms", "40 ms", "60 ms"])
ax.set_ylim(-5, 108); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_xlabel("Command delay added in the sim2sim test (on top of training randomization)")
ax.set_ylabel("Episodes surviving the full clip")
ax.set_title("Delay training, not extra training, restores delay tolerance", loc="left", fontsize=12, fontweight="bold", color=INK)
ax.legend(frameon=False, fontsize=9.5, loc="lower left")
ax.text(1.0, -0.27, "Tennis policy, mjlab. 3 training seeds per arm (thin lines), mean bold. 0 ms = 30 episodes, others 20.",
        transform=ax.transAxes, ha="right", fontsize=8, color=INK2)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_delay_seeds.png"), facecolor="white"); plt.close(fig)

# ---------- 2. stress scenarios, tennis policy before / after delay training ----------
before = load("sweeps/mjlab_tennis_baseline_full.json")
after = {**load("sweeps/mjlab_tennis_delay_latency.json"), **load("sweeps/mjlab_tennis_delay_stress.json")}
ROWS = [("train_DR", "Training randomization"), ("train_DR+kp-20%", "PD gains −20%"), ("train_DR+payload2kg", "2 kg torso payload"),
        ("train_DR+latency1", "+20 ms delay"), ("train_DR+push3x", "Pushes ×3"), ("train_DR+fric0.15", "Foot friction 0.15"),
        ("train_DR+no_lin_vel", "No base linear velocity"), ("train_DR+latency2", "+40 ms delay")]
fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
h = 0.36
for i, (k, name) in enumerate(ROWS):
    y = len(ROWS) - 1 - i
    for dy, src, color in ((h / 2 + 0.02, before, BLUE), (-h / 2 - 0.02, after, ORANGE)):
        if k in src:
            v = surv(src[k])
            ax.barh(y + dy, v, height=h, color=color, zorder=2)
            ax.text(v + 1.2, y + dy, f"{v:.0f}%", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([n for _, n in ROWS][::-1])
ax.set_xlim(0, 112); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE, label="Tennis policy"), Patch(color=ORANGE, label="After delay training")],
          frameon=False, fontsize=9.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
ax.set_xlabel("Episodes surviving the full clip (sim2sim, plain MuJoCo)")
ax.set_title("Which perturbations break the policy", loc="left", fontsize=12, fontweight="bold", color=INK, pad=26)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_stress.png"), facecolor="white"); plt.close(fig)

# ---------- 3. floor / timestep ablation ----------
ABL = [("exact training model", 0.5769), ("contact dimension", 0.5759), ("solver iterations", 0.5769), ("foot friction", 0.5816),
       ("contact priority", 0.5769), ("all four above", 0.5823), ("timestep (2 ms, 125 Hz)", 1.7189), ("no floor", 6.9690)]
fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
for i, (name, v) in enumerate(ABL):
    y = len(ABL) - 1 - i
    ax.barh(y, v, height=0.6, color=ORANGE if v > 1 else BLUE, zorder=2)
    ax.text(v + 0.08, y, f"{v:.2f}", va="center", fontsize=9, color=INK2)
ax.set_yticks(range(len(ABL))); ax.set_yticklabels([n for n, _ in ABL][::-1])
ax.set_xlim(0, 7.8); ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_xlabel("Joint error, nominal rollout (rad, L2 norm over 24 joints)")
ax.set_title("Only the floor and the timestep matter", loc="left", fontsize=12, fontweight="bold", color=INK)
ax.text(1.0, -0.25, "Raw-MJCF setting restored one at a time in the training-equivalent model. Tennis policy.",
        transform=ax.transAxes, ha="right", fontsize=8, color=INK2)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_ablation.png"), facecolor="white"); plt.close(fig)
print("wrote fig_delay_seeds.png, fig_stress.png, fig_ablation.png")

# ---------- 4. dance: left shoulder pinned at joint limits ----------
import numpy as np
d = np.load(os.path.join(HERE, "diagnostics/dance_left_shoulder.npz"))  # from 4_sim2sim/diag_shoulder_limits.py (SAVE=...)
t = np.arange(len(d["q"])) / 50.0
fig, axs = plt.subplots(3, 1, figsize=(6.4, 5.2), dpi=200, sharex=True)
for j, ax in enumerate(axs):
    lo, hi = d["rng"][j]
    pinned = d["pinned"]
    for s, e in zip(np.flatnonzero(np.diff(np.r_[0, pinned.astype(int)]) == 1), np.flatnonzero(np.diff(np.r_[pinned.astype(int), 0]) == -1)):
        ax.axvspan(t[s], t[e], color=ORANGE, alpha=0.12, lw=0)
    ax.axhline(lo, color=INK2, lw=0.8, ls=(0, (4, 3))); ax.axhline(hi, color=INK2, lw=0.8, ls=(0, (4, 3)))
    ax.plot(t, d["ref"][:, j], color=INK2, lw=1.4, label="Reference (GMR)")
    ax.plot(t, d["q"][:, j], color=BLUE, lw=1.6, label="R1 policy in MuJoCo")
    ax.set_ylabel(str(d["names"][j]).replace("left_shoulder_", "").replace("_joint", "") + " (rad)", fontsize=9.5)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
axs[0].legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
axs[0].set_title("Dance: the left-shoulder error sits where the reference is pinned at a limit", loc="left",
                 fontsize=11.5, fontweight="bold", color=INK, pad=22)
axs[-1].set_xlabel("Time in clip (s)")
fig.text(0.99, 0.005, "Left shoulder, nominal rollout, mjlab dance policy. Dashed = joint limits; shaded = reference frames at a limit (27%).",
         ha="right", fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(os.path.join(OUT, "fig_dance_shoulder_limits.png"), facecolor="white"); plt.close(fig)
print("wrote fig_dance_shoulder_limits.png")
