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
