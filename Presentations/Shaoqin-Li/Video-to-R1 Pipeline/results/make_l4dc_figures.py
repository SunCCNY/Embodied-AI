"""L4DC paper figure: survival vs constant command delay, control vs delay-randomized arm (run from results/)."""
import glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "l4dc_2026-10-19")
OUT = os.path.join(HERE, "..", "paper", "l4dc2027", "figures")
for f in ("arial.ttf", "arialbd.ttf"):
    p = "/mnt/c/Windows/Fonts/" + f
    if os.path.exists(p):
        fm.fontManager.addfont(p)
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.edgecolor": "#8a8984", "axes.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, INK2, GRID = "#2a78d6", "#eb6834", "#52514e", "#e6e5e0"
M = [("tennis", "Tennis serve"), ("woman_home_fixed", "Dance (home)"), ("rdne_boxing_moving", "Shadow boxing"),
     ("street_hiphop_moving", "Street dance"), ("dancer_truespeed", "Dance (studio)")]
DS = list(range(0, 90, 10))
R = {}
for f in glob.glob(os.path.join(DATA, "*.json")):
    t = os.path.basename(f)[:-5]
    for m, _ in M:
        if t.startswith(m + "_"):
            arm, s = t[len(m) + 1:].split("_s")
            R[(m, arm, int(s))] = json.load(open(f))


def curve(r):
    return [100 * sum(x["survived"] for x in r[f"train_DR+delay{d}ms"]) / len(r[f"train_DR+delay{d}ms"]) for d in DS]


fig, axs = plt.subplots(2, 3, figsize=(6.6, 3.9), sharex=True, sharey=True)
allc = {"ctrl": [], "dly": []}
for ax, (m, name) in zip(axs.flat, M):
    for arm, col in (("ctrl", ORANGE), ("dly", BLUE)):
        cs = [curve(R[(m, arm, s)]) for s in (42, 43, 44)]
        allc[arm] += cs
        for c in cs:
            ax.plot(DS, c, color=col, lw=0.8, alpha=0.45)
        ax.plot(DS, [sum(c[i] for c in cs) / 3 for i in range(len(DS))], color=col, lw=2, marker="o", ms=3)
    ax.axvspan(0, 50, color=GRID, alpha=0.55, lw=0, zorder=0)
    ax.axhline(50, color=INK2, lw=0.6, ls=":")
    ax.set_title(name, fontsize=9)
ax = axs.flat[5]
for arm, col, lab in (("ctrl", ORANGE, "control (no delay)"), ("dly", BLUE, "delay randomized")):
    cs = allc[arm]
    ax.plot(DS, [sum(c[i] for c in cs) / len(cs) for i in range(len(DS))], color=col, lw=2, marker="o", ms=3, label=lab)
ax.axvspan(0, 50, color=GRID, alpha=0.55, lw=0, zorder=0)
ax.axhline(50, color=INK2, lw=0.6, ls=":")
ax.set_title("All motions (mean)", fontsize=9)
for ax in axs[1]:
    ax.set_xlabel("constant command delay (ms)")
for ax in axs[:, 0]:
    ax.set_ylabel("survival (%)")
axs[0, 0].set_xticks(range(0, 90, 20)); axs[0, 0].set_ylim(-3, 103)
h, l = axs.flat[5].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8.5)
fig.tight_layout(rect=(0, 0.05, 1, 1))
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "fig_delay_margin.pdf"))
fig.savefig(os.path.join(OUT, "fig_delay_margin.png"), dpi=200)
print("ok")
