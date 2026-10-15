"""Frame figures built from the result videos (run from results/; needs opencv + Pillow + Windows Arial).

fig_delay_before_after.png  40 ms delay: last frame of the failing run vs the delay-trained run at the same moment.
(fig_dance_video_vs_r1.png and the tennis video/R1 pairs come from 4_sim2sim/render_poster_frames.py.)
"""
import os

import cv2
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = "/mnt/c/Windows/Fonts/"


def grab(video, i):
    c = cv2.VideoCapture(os.path.join(HERE, "videos", video)); c.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, fr = c.read(); assert ok, (video, i)
    return Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))


fb, fr, fs = (ImageFont.truetype(FONT + n, s) for n, s in (("arialbd.ttf", 30), ("arial.ttf", 24), ("arial.ttf", 20)))
W, H, pad, colh, rowlab = 640, 480, 24, 46, 250
cw, ch = rowlab + 2 * W + pad * 3, pad + colh + 2 * (H + pad) + 40
c = Image.new("RGB", (cw, ch), "white"); d = ImageDraw.Draw(c)
d.text((rowlab + pad + W / 2, pad), "Trained without delay", font=fb, fill="#0b0b0b", anchor="ma")
d.text((rowlab + 2 * pad + W * 1.5, pad), "Trained with random 0–50 ms delay", font=fb, fill="#0b0b0b", anchor="ma")
ROWS = [("Tennis, mjlab", "t = 1.56 s", ("r1_tennis_sim2sim_latency2_FAIL.mp4", 77), ("r1_tennis_sim2sim_latency2_delaytrained.mp4", 77),
         "falls (2/20 survive)", "finishes (15/20 survive)"),
        ("Dance, Isaac Lab", "t = 1.18 s", ("r1_dance_isaac_latency2_nodelay.mp4", 58), ("r1_dance_isaac_latency2_delaytrained.mp4", 58),
         "falls (0/20 survive)", "finishes (15/20 survive)")]
for r, (name, t, a, b, la, lb) in enumerate(ROWS):
    y = pad + colh + r * (H + pad)
    d.text((pad, y + H / 2 - 30), name, font=fb, fill="#0b0b0b"); d.text((pad, y + H / 2 + 10), t, font=fr, fill="#52514e")
    for k, (src, lab) in enumerate(((a, la), (b, lb))):
        x = rowlab + pad + k * (W + pad); c.paste(grab(*src), (x, y))
        d.rectangle([x + 10, y + H - 48, x + 30 + d.textlength(lab, font=fr), y + H - 12], fill="white")
        d.text((x + 20, y + H - 44), lab, font=fr, fill="#0b0b0b")
d.text((cw - pad, ch - 30), "Sim2sim in plain MuJoCo, training randomization + 40 ms command delay. One episode per panel; "
       "survival counts are over 20 seeds. Grey = robot, green = reference.", font=fs, fill="#52514e", anchor="ra")
c.save(os.path.join(HERE, "figures", "fig_delay_before_after.png")); print("wrote fig_delay_before_after.png", c.size)

# ---------- dance: original video vs R1 policy (panels rendered by 4_sim2sim/render_poster_frames.py) ----------
# Times are VIDEO times (frame / 25 fps). The robot reference runs 1.2x faster (see README frame-rate note).
F = os.path.join(HERE, "figures")
fb, fr, fs = (ImageFont.truetype(FONT + n, s) for n, s in (("arialbd.ttf", 44), ("arial.ttf", 30), ("arial.ttf", 24)))
W, H, arrow, pad = 450, 800, 70, 30
cw, ch = pad * 2 + (W * 2 + arrow) * 2 + 80, pad + 70 + H + 110
c = Image.new("RGB", (cw, ch), "white"); d = ImageDraw.Draw(c); x = pad
for i, title, sub in ((320, "Lunge, video 12.8 s", "tracks the video closely"),
                      (560, "Arms overhead, video 22.4 s", "reference pinned at shoulder limits")):
    v = Image.open(f"{F}/dance_video_frame_{i}.png").resize((W, H)); r = Image.open(f"{F}/dance_r1_frame_{i}.png").resize((W, H))
    d.text((x, pad), title, font=fb, fill="#0b0b0b")
    c.paste(v, (x, pad + 70)); c.paste(r, (x + W + arrow, pad + 70))
    y = pad + 70 + H // 2; ax = x + W + 12
    d.polygon([(ax, y - 9), (ax + 28, y - 9), (ax + 28, y - 20), (ax + arrow - 24, y), (ax + 28, y + 20), (ax + 28, y + 9), (ax, y + 9)], fill="#673AB7")
    d.text((x, pad + 70 + H + 14), "Original video  →  R1 policy in MuJoCo: " + sub, font=fr, fill="#52514e")
    x += W * 2 + arrow + 80
d.text((cw - pad, ch - 34), "Video: “Woman Dancing” by Antoni Shkraba, Pexels.", font=fs, fill="#52514e", anchor="ra")
c.save(os.path.join(F, "fig_dance_video_vs_r1.png")); print("wrote fig_dance_video_vs_r1.png", c.size)
