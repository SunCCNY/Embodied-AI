"""Week 31 deck assets: charts, terminal panels (real captured output), video poster frames.

Run with the deck venv: <scratchpad>/w31/.venv/bin/python make_assets.py
All numbers come from the results READMEs / sweep files in video_to_r1_pipeline/results/.
"""
import os
import subprocess
from pathlib import Path

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
A = HERE / "assets"
A.mkdir(exist_ok=True)
PIPE = Path.home() / "claude_code_projects/embodied_ai_r1/video_to_r1_pipeline"
RES = PIPE / "results"
FF = imageio_ffmpeg.get_ffmpeg_exe()

FONTS = Path.home() / ".local/share/fonts"
for f in ("calibri.ttf", "calibrib.ttf"):
    if (FONTS / f).exists():
        font_manager.fontManager.addfont(str(FONTS / f))

# validated (dataviz validate_palette.js, light, all-pairs): matches the paper figure
# control/plain = orange, delay = blue, base = aqua (aqua < 3:1 -> every line direct-labelled)
C_DELAY, C_CTRL, C_BASE = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#22252A", "#5A6068", "#E4E2DC", "#FCFBF8"

plt.rcParams.update({
    "font.family": "Calibri", "font.size": 15, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
})


def line_chart(path, x, series, ymax, ylabel, shade_to=None, label_y=None):
    fig, ax = plt.subplots(figsize=(7.6, 4.3), dpi=200)
    if shade_to is not None:
        ax.axvspan(0, shade_to, color="#EEEDE9", zorder=0, lw=0)
        ax.text(shade_to / 2, ymax * 1.045, f"delay range used in training (0–{shade_to} ms)",
                ha="center", va="bottom", color=INK2, fontsize=12)
    for name, ys, c in series:
        ax.plot(x, ys, color=c, lw=2.6, solid_capstyle="round", solid_joinstyle="round", zorder=3)
        ax.scatter(x, ys, s=46, color=c, edgecolor=SURF, linewidth=1.6, zorder=4)
    # direct labels at the line ends (text in ink, colored key dot beside it)
    for name, ys, c in series:
        y = label_y[name] if label_y else ys[-1]
        ax.scatter([x[-1] + 3.2], [y], s=46, color=c, clip_on=False, zorder=5)
        ax.text(x[-1] + 5.2, y, name, va="center", ha="left", color=INK, fontsize=14, clip_on=False)
    ax.set_xlim(-2, x[-1] + 2)
    ax.set_ylim(-ymax * 0.03, ymax * 1.03)
    ax.set_xticks(x)
    ax.set_xlabel("command delay held constant for the whole run (ms)")
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    fig.subplots_adjust(left=0.1, right=0.74, bottom=0.17, top=0.88)
    fig.savefig(path)
    plt.close(fig)


def charts():
    d = [0, 20, 30, 40, 50, 60, 80]
    line_chart(A / "chart_walkrun_delay.png", d, [
        ("delay-trained", [50, 50, 50, 50, 50, 50, 46], C_DELAY),
        ("+3,000 plain (control)", [50, 50, 50, 48, 16, 0, 0], C_CTRL),
        ("base (Unitree recipe)", [50, 50, 50, 44, 15, 0, 0], C_BASE),
    ], 50, "robots that survived (of 50)", shade_to=50,
        label_y={"delay-trained": 46, "+3,000 plain (control)": 7, "base (Unitree recipe)": -1})
    d = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    line_chart(A / "chart_jj_delay.png", d, [
        ("delay-trained", [20, 20, 20, 20, 20, 20, 19, 15, 4], C_DELAY),
        ("plain", [20, 20, 19, 19, 2, 1, 0, 0, 0], C_CTRL),
    ], 20, "runs that survived (of 20)", shade_to=50)


def termshot(path, title, text, cols=None):
    mono = ImageFont.truetype(str(FONTS / "consola.ttf"), 30)
    bold = ImageFont.truetype(str(FONTS / "consolab.ttf"), 30)
    lines = text.rstrip("\n").split("\n")
    cw = mono.getbbox("M")[2]
    cols = cols or max(len(l) for l in lines)
    lh, pad, bar = 40, 34, 56
    W = pad * 2 + cw * cols
    H = bar + pad + lh * len(lines) + pad - 6
    im = Image.new("RGB", (W, H), "#1E1F22")
    dr = ImageDraw.Draw(im)
    dr.rounded_rectangle([0, 0, W - 1, H - 1], 18, fill="#1E1F22")
    dr.rectangle([0, 0, W, bar], fill="#2B2D31")
    for i, c in enumerate(("#FF5F57", "#FEBC2E", "#28C840")):
        dr.ellipse([24 + i * 34, 18, 24 + i * 34 + 20, 38], fill=c)
    tf = ImageFont.truetype(str(FONTS / "consola.ttf"), 24)
    tw = dr.textlength(title, font=tf)
    dr.text(((W - tw) / 2, 15), title, font=tf, fill="#A9ADB4")
    y = bar + pad - 8
    for l in lines:
        if l.startswith("$ "):
            dr.text((pad, y), "$", font=bold, fill="#6FCF97")
            dr.text((pad + cw * 2, y), l[2:], font=bold, fill="#E8E8E8")
        else:
            dr.text((pad, y), l, font=mono, fill="#C9CDD3")
        y += lh
    # rounded mask
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, W - 1, H - 1], 18, fill=255)
    out = Image.new("RGB", (W, H), SURF)
    out.paste(im, (0, 0), m)
    out.save(path)


def run(cmd, cwd):
    """Run the real command and return '$ cmd' + its real output (ANSI codes stripped)."""
    out = subprocess.run(["bash", "-c", cmd], cwd=cwd, capture_output=True, text=True).stdout
    import re
    out = re.sub(r"\x1b\[[0-9;]*m", "", out)
    return f"$ {cmd}\n{out}"


def terminals():
    logs = Path.home() / "unitree_rl_mjlab/logs"
    t = run('grep -m1 "Number of environments" walk_run_flat_train.log', logs)
    t += run('grep -A40 "iteration 10000/" walk_run_flat_train.log | grep -E "reward:|fell_over|elapsed"', logs)
    termshot(A / "term_walkrun_train.png", "~/unitree_rl_mjlab/logs", t)

    s2s = RES / "walk_run_delay_2026-10-23/sim2sim_cpu"
    t = run('grep -E "^trained +0.00 +(40|50|60|80) " sim2sim_base.txt', s2s)
    t += run('grep -E "^trained +0.00 +(40|50|60|80) " sim2sim_delay.txt', s2s)
    termshot(A / "term_stand_delay.png", "walk_run_delay_2026-10-23/sim2sim_cpu   (vx_cmd 0 = standing)", t)

    sw = RES / "jump_clips_2026-10-23/sweeps"
    t = run('grep -E "scenario|delay(0|40|50|60|70)ms|push" l4dc_sweep_jumping_jacks_fixed_delay.txt '
            '| cut -c1-56', sw)
    termshot(A / "term_jj_sweep.png", "jump_clips_2026-10-23/sweeps   (delay-trained jumping jacks)", t)


VIDEOS = {
    "probe_tennis_to_dance": (RES / "hub_switch_probe_2026-10-20/videos/tennis_to_dance_direct.mp4", 14.0),
    "walk_model5000": (RES / "walk_run_eval_2026-10-22/walk_model5000.mp4", 8.0),
    "run_model10000": (RES / "walk_run_eval_2026-10-22/run_model10000.mp4", 17.5),
    "walk_run_delay": (RES / "walk_run_delay_2026-10-23/walk_run_delay_model12999.mp4", 15.0),
    "jumping_jacks": (RES / "jump_clips_2026-10-23/videos/jumping_jacks_side_by_side.mp4", 4.0),
    "jump_rope": (RES / "jump_clips_2026-10-23/videos/jump_rope_side_by_side.mp4", 6.0),
}


def posters():
    for name, (src, t) in VIDEOS.items():
        subprocess.run([FF, "-y", "-loglevel", "error", "-ss", str(t), "-i", str(src), "-frames:v", "1",
                        str(A / f"poster_{name}.png")], check=True)
        # a projector-friendly copy of the poster, also used as the still fallback
        Image.open(A / f"poster_{name}.png").convert("RGB").save(A / f"poster_{name}.jpg", quality=92)


if __name__ == "__main__":
    charts()
    terminals()
    posters()
    print("assets ->", A)
    for p in sorted(A.iterdir()):
        print(f"  {p.name}  {Image.open(p).size if p.suffix in ('.png', '.jpg') else ''}")
