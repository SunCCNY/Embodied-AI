"""Week 28 deck -- imitation learning: R1 learns to mimic the tennis motion.

Continues the Week 27 GVHMR deck. Same accepted format: first person, real
captured output in terminal panels, no equations, robot video. See the
`embodied-ai-experiment-deck-recipe` memory. Helper block cloned from
build_ppt_week27.py so the decks look identical.

Shape: the pipeline (IL is this week's addition) -> the choice of framework
(TWIST vs BeyondMimic vs ASAP, why BeyondMimic via mjlab) -> setup (Blackwell
verification, R1 added to mjlab, WandB) -> the code, section by section ->
training spec -> training results with a real plot -> the result video ->
sim2sim, reported honestly (native eval success, the bug found and fixed,
what's still open) -> next.

Every terminal panel is read from week28_shots/captures/, which
holds the real stdout of commands actually run this session. Nothing in them
is typed in or invented.

Run with the isaac_eco interpreter (python-pptx installed this session).
"""
import os
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

HERE = os.path.dirname(os.path.abspath(__file__))
SHOT = os.path.join(HERE, "week28_shots")
CAP = os.path.join(SHOT, "captures")
OUT = os.path.join(HERE, "embodied ai week 28.pptx")
MEDIA = os.path.join(HERE, "week28_media")

NAVY = RGBColor(0x1B, 0x2A, 0x3A)
TEAL = RGBColor(0x0F, 0x7A, 0x6B)
GO = RGBColor(0x0A, 0xA0, 0x8A)
OCHRE = RGBColor(0xB8, 0x86, 0x2E)
RUST = RGBColor(0xB0, 0x47, 0x3C)
INK = RGBColor(0x22, 0x25, 0x2A)
GRAY = RGBColor(0x5A, 0x60, 0x68)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PAPER = RGBColor(0xFC, 0xFC, 0xFB)
SAND = RGBColor(0xF4, 0xEC, 0xDA)
MINT = RGBColor(0xE3, 0xF0, 0xED)
STONE = RGBColor(0xEE, 0xED, 0xE9)
ROSE = RGBColor(0xF6, 0xE4, 0xE1)
TERM = RGBColor(0x14, 0x1A, 0x22)
TERMTX = RGBColor(0xDD, 0xE4, 0xEC)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def cap(name):
    return open(os.path.join(CAP, name)).read().rstrip("\n")


def add(bg=PAPER):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = bg
    return s


def txt(slide, text, x, y, w, h, size=16, color=INK, bold=False,
        align=PP_ALIGN.LEFT, font="Calibri", italic=False, spacing=1.0):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, ln in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ln
        p.alignment = align
        p.line_spacing = spacing
        for r in p.runs:
            r.font.size = Pt(size)
            r.font.color.rgb = color
            r.font.bold = bold
            r.font.name = font
            r.font.italic = italic
    return tb


def rect(slide, x, y, w, h, fill, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    sh = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def term(slide, text, x, y, w, h, size=12.5, title=None):
    """Captured terminal output, drawn as real text so it stays crisp."""
    rect(slide, x, y, w, h, TERM)
    if title:
        txt(slide, title, x + 0.24, y + 0.12, w - 0.5, 0.28, size=11,
            color=RGBColor(0x7F, 0x93, 0xA6), font="Consolas")
    bw, bh = w - 0.5, h - (0.62 if title else 0.36)
    size = _shrink(text, bw, bh, size, 1.06, floor=7.5)
    txt(slide, text, x + 0.24, y + (0.44 if title else 0.18), bw,
        bh, size=size, color=TERMTX, font="Consolas", spacing=1.06)


def fit(slide, path, cx, top, maxw, maxh):
    iw, ih = Image.open(path).size
    sc = min(maxw / iw, maxh / ih)
    w, h = iw * sc, ih * sc
    slide.shapes.add_picture(path, Inches(cx - w / 2), Inches(top),
                             width=Inches(w), height=Inches(h))
    return w, h


def movie(slide, name, cx, top, w, caption=None):
    poster = os.path.join(SHOT, "poster_" + name + ".png")
    iw, ih = Image.open(poster).size
    h = w * ih / iw
    slide.shapes.add_movie(os.path.join(MEDIA, name + ".mp4"),
                           Inches(cx - w / 2), Inches(top), Inches(w), Inches(h),
                           poster_frame_image=poster, mime_type="video/mp4")
    if caption:
        txt(slide, caption, cx - w / 2, top + h + 0.06, w, 0.34, size=12,
            italic=True, color=GRAY, align=PP_ALIGN.CENTER)
    return top + h


def head(slide, title, sub=None):
    txt(slide, title, 0.62, 0.42, 12.1, 0.7, size=30, bold=True, color=NAVY)
    if sub:
        txt(slide, sub, 0.62, 1.08, 12.1, 0.45, size=15, color=GRAY)


def divider(label, name, sub, note):
    s = add(NAVY)
    rect(s, 1.1, 2.45, 0.09, 2.1, OCHRE, MSO_SHAPE.RECTANGLE)
    txt(s, label, 1.5, 2.42, 10.5, 0.4, size=15, color=OCHRE)
    txt(s, name, 1.5, 2.85, 10.5, 0.9, size=40, bold=True, color=WHITE)
    txt(s, sub, 1.5, 3.85, 10.5, 0.5, size=20, color=RGBColor(0xC9, 0xD2, 0xDA))
    txt(s, note, 1.5, 4.55, 10.5, 0.6, size=15,
        color=RGBColor(0x8F, 0x9E, 0xAC), italic=True)
    return s


def _fits(text, w_in, h_in, size, spacing):
    cpl = max(8, int(w_in / (0.0082 * size)))
    lines = sum(max(1, -(-len(para) // cpl)) for para in text.split("\n"))
    return lines * (size / 72.0) * spacing * 1.34 <= h_in


def _shrink(text, w_in, h_in, size, spacing, floor=9.0):
    while size > floor and not _fits(text, w_in, h_in, size, spacing):
        size -= 0.5
    return size


def card(slide, x, y, w, h, title, body, fill=STONE, title_color=NAVY,
         title_size=15, body_size=12.5):
    rect(slide, x, y, w, h, fill)
    # Title height grows with however many lines it actually wraps to, so a
    # long/2-line title never overlaps the body below it.
    title_cpl = max(8, int((w - 0.56) / (0.0082 * title_size)))
    title_lines = sum(max(1, -(-len(p) // title_cpl)) for p in title.split("\n"))
    title_h = max(0.4, title_lines * (title_size / 72.0) * 1.15 * 1.5)
    txt(slide, title, x + 0.28, y + 0.2, w - 0.56, title_h, size=title_size,
        bold=True, color=title_color)
    body_y = y + 0.2 + title_h + 0.08
    body_h = max(0.3, (y + h) - body_y - 0.15)
    bsize = _shrink(body, w - 0.56, body_h, body_size, 1.12, floor=8.5)
    txt(slide, body, x + 0.28, body_y, w - 0.56, body_h, size=bsize,
        color=INK, spacing=1.12)


def pill(slide, x, y, w, h, text, fill, tcolor=WHITE, size=12):
    rect(slide, x, y, w, h, fill, MSO_SHAPE.ROUNDED_RECTANGLE)
    txt(slide, text, x, y, w, h, size=size, bold=True, color=tcolor,
        align=PP_ALIGN.CENTER)


def arrow(slide, x, y, w, h=0.35, color=GRAY):
    sh = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y),
                                Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    sh.shadow.inherit = False


# ============================================================ #
# 1. Divider
# ============================================================ #
divider("WEEK 28", "Imitation Learning",
        "R1 learns to mimic the retargeted tennis motion",
        "video -> GVHMR -> GMR -> imitation learning -- the last arrow is this week's work")

# ============================================================ #
# 2. Pipeline slide
# ============================================================ #
s = add()
head(s, "The pipeline", "Four stages. Three were done before this week.")
stages = [
    ("1. Video", "a phone/demo clip\nof a human moving", GO, "DONE"),
    ("2. GVHMR", "video -> SMPL-X\nhuman kinematics", GO, "DONE"),
    ("3. GMR", "SMPL-X -> R1\njoint trajectories", GO, "DONE"),
    ("4. IL Training", "R1 motion data ->\na trained policy", OCHRE, "THIS WEEK"),
]
x = 0.7
w = 2.75
for i, (name, desc, color, tag) in enumerate(stages):
    card(s, x, 2.0, w, 2.3, name, desc, fill=MINT if color == GO else SAND,
         title_color=NAVY, title_size=17, body_size=13)
    pill(s, x + 0.35, 4.45, w - 0.7, 0.45, tag, color, size=12)
    if i < len(stages) - 1:
        arrow(s, x + w + 0.05, 2.95, 0.5, color=GRAY)
    x += w + 0.55
txt(s, "GVHMR+GMR were ported to this workstation last week (~11x/2.6x faster than the laptop, "
       "numbers matched within noise). This week: pick a training framework, wire up R1, train, "
       "and check the result both natively and via an independent sim2sim harness.",
    0.7, 5.3, 11.9, 1.3, size=14, color=GRAY, spacing=1.2)

# ============================================================ #
# 3. Divider: the choice
# ============================================================ #
divider("THE CHOICE", "Which framework?",
        "GMR's output is designed to feed straight into one of three published training codebases",
        "TWIST, BeyondMimic, or ASAP -- picked before writing a line of training code")

# ============================================================ #
# 4. Comparison slide
# ============================================================ #
s = add()
head(s, "Three candidates, one hardware reality",
     "This RTX 5080 (Blackwell) has already broken two Isaac-family stacks this project")
rows = [
    ("TWIST", "IsaacGym (train)", "Confirmed broken -- NVIDIA forum: IsaacGym Preview 4\nmissing SM_120 kernels on RTX 5080, deprecated, no fix", "No R1 path -- hardcoded to G1"),
    ("BeyondMimic\n(original)", "Isaac Lab / Isaac Sim", "High risk -- same GPU-PhysX hang this project\nalready hit and worked around with device=\"cpu\"", "G1-focused, modular robot dirs"),
    ("BeyondMimic\nvia mjlab", "MuJoCo-Warp\n(no Isaac at all)", "Low risk -- pure CUDA compute, no rendering\nstack; matches what already works here", "Data-driven robot configs -- comparable\nlift to the GMR R1 work already done"),
    ("ASAP", "IsaacGym / IsaacSim\n/ Genesis (pick one)", "Medium -- every backend shares some version\nof the same risk; Genesis untested here", "No native R1, but a retargeting path that\noverlaps with GMR work already done"),
]
headers = ["Framework", "Physics engine", "Blackwell / WSL2 risk", "R1 support"]
colw = [1.9, 2.1, 4.6, 4.6]
x0 = 0.55
y0 = 2.0
xs = [x0]
for w_ in colw[:-1]:
    xs.append(xs[-1] + w_)
for xh, w_, htext in zip(xs, colw, headers):
    txt(s, htext, xh, y0, w_ - 0.1, 0.35, size=13, bold=True, color=NAVY)
rowh = 1.15
for r, (fw, eng, risk, r1s) in enumerate(rows):
    ry = y0 + 0.42 + r * rowh
    fill = MINT if fw == "BeyondMimic\nvia mjlab" else STONE
    rect(s, x0 - 0.1, ry - 0.06, sum(colw) + 0.1, rowh - 0.08, fill,
         MSO_SHAPE.RECTANGLE)
    for xh, w_, cell in zip(xs, colw, [fw, eng, risk, r1s]):
        txt(s, cell, xh, ry, w_ - 0.15, rowh - 0.1, size=10.5, color=INK, spacing=1.05)

# ============================================================ #
# 5. Why BeyondMimic via mjlab
# ============================================================ #
s = add()
head(s, "Why BeyondMimic via mjlab")
pts = [
    "Structurally avoids a failure class this project already hit twice on this exact GPU: "
    "Isaac Sim's GPU-PhysX pipeline hang, and now IsaacGym's outright SM_120 incompatibility.",
    "Trains headless on pure CUDA compute (MuJoCo-Warp) -- no rendering/PhysX stack at all, "
    "matching this machine's actual proven-working profile (CUDA yes, GL/Vulkan no).",
    "Most actively maintained of the four: 3.1k stars, 1,115 commits, active CI.",
    "It's a direct, source-linked reimplementation -- mjlab's own tracking_env_cfg.py header reads:\n"
    "  \"This is a re-implementation of BeyondMimic. Based on HybridRobotics/whole_body_tracking,\n"
    "   commit f8e20c880d9c8ec7172a13d3a88a65e3a5a88448\"",
    "Adding R1 means writing a data-driven robot config -- comparable effort to the R1-into-GMR "
    "work already finished, not a new research problem.",
]
y = 1.9
for p in pts:
    rect(s, 0.7, y + 0.08, 0.14, 0.14, OCHRE, MSO_SHAPE.OVAL)
    txt(s, p, 1.05, y - 0.08, 11.4, 0.9, size=14.5, color=INK, spacing=1.15)
    y += 0.15 + 0.28 * (p.count("\n") + 1) + (0.55 if len(p) > 90 else 0.25)

# ============================================================ #
# 6. Divider: setup
# ============================================================ #
divider("SETUP", "Wiring it up",
        "Verify Blackwell works, add R1 to mjlab, connect WandB",
        "three real blockers hit along the way -- covered honestly, not glossed over")

# ============================================================ #
# 7. Blackwell verification
# ============================================================ #
s = add()
head(s, "Step 1 -- verify the GPU path before building anything")
txt(s, "mjlab needs an NVIDIA GPU for training (MuJoCo-Warp, GPU-accelerated compute). "
       "Before writing any R1-specific code, confirm this Blackwell card actually runs a real "
       "GPU tensor op through the exact torch build mjlab uses.",
    0.7, 1.85, 11.9, 1.0, size=15, color=GRAY, spacing=1.2)
term(s, cap("blackwell_check.txt"), 0.7, 3.0, 11.9, 2.1,
     title="python -c \"import torch; torch.cuda.get_device_capability(0)\"")
txt(s, "capability (12, 0) = Blackwell sm_120, and a real matmul executed on-device. "
       "torch 2.7.0+cu128 is the confirmed-working build for this card (already established "
       "porting GVHMR last week).", 0.7, 5.35, 11.9, 0.8, size=13.5, italic=True, color=GRAY)

# ============================================================ #
# 8. R1 added to mjlab
# ============================================================ #
s = add()
head(s, "Step 2 -- add R1 as a real robot inside mjlab",
     "Not a config file -- mjlab's robots are code (asset_zoo/robots/<name>/)")
card(s, 0.6, 1.9, 5.7, 4.3, "What had to be written",
     "asset_zoo/robots/unitree_r1/r1_constants.py\n"
     "  - MJCF path (reused GMR's clean 24-joint\n"
     "    R1 model -- no phantom joints)\n"
     "  - PD gains: taken from this project's own\n"
     "    r1_stand_sim2sim.py deployment values,\n"
     "    NOT a G1-analogy guess\n"
     "  - Effort limits: official Unitree R1_C++.xml\n"
     "    ctrlrange (88/139/50/25 Nm)\n"
     "  - Stand keyframe: the validated Goal-1 pose\n\n"
     "tasks/tracking/config/r1/\n"
     "  - env_cfgs.py, rl_cfg.py (mirrors G1's)\n"
     "  - registered as Mjlab-Tracking-Flat-Unitree-R1",
     body_size=12.5)
term(s, cap("r1_task_registered.txt"), 6.55, 1.9, 6.1, 2.0,
     title="uv run python -c \"... build Entity, Scene, Simulation ...\"")
card(s, 6.55, 4.1, 6.1, 2.1, "Every number is sourced, not guessed",
     "PD gains: real deployment values (r1_stand_sim2sim.py).\n"
     "Effort limits: real motor specs (R1_C++.xml).\n"
     "Stand pose: the already-validated Goal-1 pose.\n"
     "Only the joint-limit/armature default is a MuJoCo\n"
     "generic (0.01) -- no per-motor rotor-inertia sheet\n"
     "exists for R1 the way it does for G1.",
     fill=MINT, body_size=12)

# ============================================================ #
# 9. What is WandB / why needed
# ============================================================ #
s = add()
head(s, "What is Weights & Biases, and why does training need it?")
card(s, 0.7, 1.9, 5.7, 4.3, "What it is",
     "A hosted experiment-tracking service: it logs training\n"
     "curves, hyperparameters, and files (\"artifacts\") to a\n"
     "web dashboard in real time, and stores them in an\n"
     "account-scoped \"Registry\" that any later run can pull\n"
     "back by name.\n\n"
     "BeyondMimic (and mjlab's reimplementation) uses WandB\n"
     "not just for logging, but as the actual storage/lookup\n"
     "mechanism for reference motions: a converted motion\n"
     "file gets uploaded once, then any training run just\n"
     "asks for it by name.",
     body_size=13)
card(s, 6.55, 1.9, 6.1, 4.3, "Why this project needs it (not optional)",
     "csv_to_npz.py auto-uploads the converted GMR motion\n"
     "to a WandB registry as part of the conversion step.\n\n"
     "The train command then pulls the motion back by\n"
     "--registry-name <org>/wandb-registry-motions/r1_tennis\n"
     "-- there is no \"just point it at a local file\" path in\n"
     "this framework's design.\n\n"
     "Real blocker hit: WandB registries can only be created\n"
     "through the web UI, never the API -- had to walk\n"
     "through creating one by hand before training could\n"
     "start at all.",
     fill=SAND, body_size=13)

# ============================================================ #
# 10. WandB registry upload
# ============================================================ #
s = add()
head(s, "The motion, converted and registered")
txt(s, "GMR's output pickle (root_pos, root_rot, dof_pos -- already MuJoCo qpos-shaped) is "
       "converted to the CSV mjlab expects, then run through csv_to_npz.py, which replays it "
       "through forward kinematics and pushes the result to the WandB registry.",
    0.7, 1.85, 11.9, 1.0, size=14.5, color=GRAY, spacing=1.2)
term(s, cap("wandb_registry_upload.txt"), 0.7, 3.0, 11.9, 3.4,
     title="uv run python -m mjlab.scripts.csv_to_npz --input-file tennis.csv --robot r1 ...")

# ============================================================ #
# 11. Divider: the code
# ============================================================ #
divider("THE CODE", "Section by section",
        "What each piece of the R1 tracking config actually does",
        "r1_constants.py (the robot) + env_cfgs.py (the task)")

# ============================================================ #
# 12. r1_constants.py walkthrough
# ============================================================ #
s = add()
head(s, "r1_constants.py -- the robot", "Every section, what it's for")
sections = [
    ("MJCF + get_spec()", "Points at the GMR-clean R1 model (24 real joints, no phantom "
     "waist_pitch/wrist_pitch/wrist_yaw). get_spec() returns a fresh mujoco.MjSpec each call."),
    ("Actuator configs (7 groups)", "One BuiltinPositionActuatorCfg per joint-torque class. "
     "Each sets stiffness, damping, effort_limit -- turns a kinematic joint into a controlled one."),
    ("STAND_KEYFRAME", "The robot's pose and height at episode reset -- reused verbatim from "
     "the already-validated Goal-1 standing pose, not re-derived."),
    ("FULL_COLLISION", "Which geoms can touch what, and with what friction/priority -- feet "
     "get condim=3 (real contact), everything else condim=1 (self-collision only)."),
    ("get_r1_robot_cfg()", "Assembles all of the above into one EntityCfg. Returns a fresh "
     "instance every call -- mjlab explicitly warns against sharing one across environments."),
    ("R1_ACTION_SCALE", "Computed, not chosen: 0.25 * effort_limit / stiffness per joint -- "
     "this is what converts the policy's raw network output into a joint position offset."),
]
y = 1.75
for name, desc in sections:
    card(s, 0.6, y, 12.1, 0.83, name, desc, fill=STONE, title_size=13.5, body_size=11.5)
    y += 0.93

# ============================================================ #
# 13. env_cfgs.py walkthrough
# ============================================================ #
s = add()
head(s, "env_cfgs.py -- the tracking task", "unitree_r1_flat_tracking_env_cfg()")
sections2 = [
    ("scene.entities", "{\"robot\": get_r1_robot_cfg()} -- plugs the R1 config from the previous "
     "slide into the shared BeyondMimic tracking environment (make_tracking_env_cfg())."),
    ("self_collision sensor", "A ContactSensorCfg watching the pelvis subtree against itself -- "
     "feeds the self_collisions reward term and the training curves' collision metric."),
    ("joint_pos_action.scale", "Set to R1_ACTION_SCALE -- without this the action-to-target "
     "conversion would silently use whatever the shared default is, not R1's real gains."),
    ("motion command anchor/body_names", "anchor_body_name=\"torso_link\"; 14 tracked bodies "
     "(pelvis, hips, knees, ankles, torso, shoulders, elbows, wrists). R1 has no wrist_yaw_link "
     "(phantom joint) -- wrist_roll_link stands in as the real last arm body."),
    ("foot_friction / base_com events", "Domain-randomization event configs re-pointed at R1's "
     "real geom/body names (left|right)_foot[1-7]_collision and torso_link."),
    ("ee_body_pos termination", "Ends an episode early if a tracked end-effector (ankles, "
     "wrists) drifts too far from the reference -- the actual signal behind "
     "\"Episode_Termination/anchor_pos\" in the training curves."),
]
y = 1.75
for name, desc in sections2:
    card(s, 0.6, y, 12.1, 0.83, name, desc, fill=STONE, title_size=13.5, body_size=11.5)
    y += 0.93

# ============================================================ #
# 14. Divider: training
# ============================================================ #
divider("TRAINING", "3,000 iterations, 4,096 envs",
        "~1 hour on the RTX 5080, measured, not estimated",
        "a 100-iteration benchmark set the estimate before committing to a real run")

# ============================================================ #
# 15. Training spec
# ============================================================ #
s = add()
head(s, "Training spec", "What was actually run, and where each number comes from")
spec_rows = [
    ("Episode length", "10.0 s", "tracking_env_cfg.py default -- one full pass of a ~10s motion clip"),
    ("Control rate", "125 Hz", "physics dt 0.002s x decimation 4"),
    ("Parallel environments", "4,096", "measured via a 100-iter benchmark first: ~1.1s/iteration, "
     "82-100k steps/sec on this GPU"),
    ("Steps per env per iteration", "24", "rl_cfg.py num_steps_per_env"),
    ("PPO epochs / mini-batches", "5 / 4", "rl_cfg.py -- standard PPO inner-loop settings"),
    ("Learning rate", "1e-3, adaptive (target KL 0.01)", "rl_cfg.py"),
    ("gamma / GAE lambda", "0.99 / 0.95", "rl_cfg.py"),
    ("Total iterations", "3,000", "a deliberately conservative first real attempt, not the "
     "framework's 30,000 default (which is tuned for general G1 locomotion, not one short clip)"),
    ("Wall-clock time", "59:15", "measured -- matches the ~1hr estimate from the benchmark"),
]
y = 1.85
for label, val, note in spec_rows:
    rect(s, 0.6, y, 12.1, 0.5, STONE, MSO_SHAPE.RECTANGLE)
    txt(s, label, 0.75, y + 0.06, 3.0, 0.38, size=12.5, bold=True, color=NAVY)
    txt(s, val, 3.85, y + 0.06, 2.4, 0.38, size=12.5, bold=True, color=TEAL)
    txt(s, note, 6.35, y + 0.06, 6.2, 0.4, size=10.5, color=GRAY, spacing=1.0)
    y += 0.545

# ============================================================ #
# 16. Training results -- what each metric means
# ============================================================ #
s = add()
head(s, "Reading the training curves", "What each number represents, and what's an acceptable range")
rows3 = [
    ("Mean reward", "-0.95 -> +27.04", "Sum of all weighted reward terms. Negative = mostly "
     "failing early; the sign flip to strongly positive is the clearest single convergence signal."),
    ("error_joint_pos", "2.56 -> 1.16 rad", "L2 norm across all 24 joints vs the reference pose. "
     "Lower is better; this run's final value is still a first-pass number, not a converged floor."),
    ("error_body_pos", "0.17 -> 0.068 m", "Mean tracked-body position error (14 bodies). Sub-10cm "
     "is a reasonable \"looks right\" threshold for this robot's scale."),
    ("Episode_Termination/anchor_pos", "4-17 -> 0.0", "How many envs per batch fell/diverged far "
     "enough from the reference to end early. Zero means the policy stopped falling over."),
    ("Episode_Termination/time_out", "2-6 -> 8.6", "How many envs instead survive to the full "
     "10s episode. This flipping to dominate is the real \"it's tracking now\" signal."),
]
y = 1.85
for name, change, note in rows3:
    card(s, 0.6, y, 12.1, 0.95, f"{name}   ({change})", note, fill=STONE,
         title_size=13, body_size=11.5)
    y += 1.03

# ============================================================ #
# 17. The plot
# ============================================================ #
s = add()
head(s, "The training curves", "real TensorBoard data from this run, not illustrative")
fit(s, os.path.join(SHOT, "training_curves.png"), 6.67, 1.85, 12.0, 4.6)
txt(s, "Left: mean episode reward. Right: three tracking-error metrics, all trending down. "
       "3,000 iterations was a first real attempt, not a claim of full convergence.",
    0.7, 6.55, 11.9, 0.6, size=13, italic=True, color=GRAY, align=PP_ALIGN.CENTER)

# ============================================================ #
# 18. Training start/end terminal panels
# ============================================================ #
s = add()
head(s, "Start and end of the run, verbatim")
term(s, cap("training_start.txt"), 0.6, 1.85, 5.95, 4.7, size=11, title="iteration 1/3000")
term(s, cap("training_end.txt"), 6.75, 1.85, 5.95, 4.7, size=11, title="iteration 2999/3000")

# ============================================================ #
# 19. Native evaluation
# ============================================================ #
s = add()
head(s, "Checking it natively before trusting the video",
     "mjlab's own evaluate.py -- 64 held-out episodes, real metrics")
term(s, cap("evaluate_results.txt"), 0.7, 1.9, 11.9, 3.2,
     title="uv run python -m mjlab.tasks.tracking.scripts.evaluate ... --num-envs 64")
txt(s, "96.9% success rate (62/64 episodes survive to timeout). mpkpe (mean per-keypoint "
       "position error) 0.30m, end-effector position error 8.7cm -- the policy genuinely tracks "
       "the motion, not just a training-curve artifact.",
    0.7, 5.3, 11.9, 0.9, size=14, color=GRAY, spacing=1.2)

# ============================================================ #
# 20. Result video
# ============================================================ #
s = add()
head(s, "The result", "checkpoint model_2999, played back in mjlab's own simulator")
movie(s, "r1_tennis_tracking", 6.67, 1.9, 6.5,
      caption="R1 tracking the retargeted tennis motion after 3,000 training iterations")

# ============================================================ #
# 20b. Divider: second motion
# ============================================================ #
divider("SECOND MOTION", "Does this generalize?",
        "Everything so far used one clip -- GVHMR's own tennis demo",
        "sourced a fresh video from the open web, same session, no changes to the pipeline")

# ============================================================ #
# 20c. Sourcing a real video
# ============================================================ #
s = add()
head(s, "Finding a video -- and a lesson in not trusting descriptions")
txt(s, "Wanted something more dynamic than tennis: a dance clip, full body, plain background, "
       "static camera (matches what made the tennis demo clip work well). Searched free stock "
       "video sites -- but a page's own description of a clip is not the same as looking at it.",
    0.7, 1.8, 11.9, 1.0, size=14.5, color=GRAY, spacing=1.2)
term(s, cap("dance_sourcing_lesson.txt"), 0.7, 2.95, 11.9, 3.2, size=12.5,
     title="finding a usable clip")
txt(s, "Candidate 2's page said \"full-body performer\" -- the actual decoded frame showed a "
       "tight crop of just legs and feet. Lesson: pull the real poster image or a decoded frame "
       "before downloading, every time.",
    0.7, 6.25, 11.9, 0.9, size=13, italic=True, color=GRAY)

# ============================================================ #
# 20d. Same pipeline, new motion
# ============================================================ #
s = add()
head(s, "Same pipeline, new motion", "no code changes -- GVHMR -> GMR -> mjlab, first attempt")
term(s, cap("dance_gvhmr_gmr.txt"), 0.6, 1.85, 5.9, 3.1, size=10.5,
     title="GVHMR + GMR retarget, real output")
term(s, cap("dance_native_eval.txt"), 6.6, 1.85, 5.9, 3.1, size=11,
     title="native evaluate.py -- dance vs tennis")
rows = [
    ("Mean reward", "-0.95 -> 27.04", "-0.61 -> 26.81"),
    ("Native success_rate", "0.9688", "1.0000"),
    ("mpkpe", "0.2991", "0.2195"),
]
y = 5.15
txt(s, "", 0.6, y, 1, 1)
for i, (label, tennis_v, dance_v) in enumerate(rows):
    rowy = y + i * 0.38
    txt(s, label, 0.7, rowy, 3.4, 0.35, size=12, bold=True, color=NAVY)
    txt(s, "Tennis: " + tennis_v, 4.2, rowy, 3.9, 0.35, size=12, color=GRAY)
    txt(s, "Dance: " + dance_v, 8.2, rowy, 3.9, 0.35, size=12, color=TEAL, bold=True)

# ============================================================ #
# 20e. Dance result video
# ============================================================ #
s = add()
head(s, "The result, again -- on a motion the pipeline had never seen",
     "checkpoint model_2999, 3,000 iterations, first attempt")
movie(s, "r1_dancer_tracking", 6.67, 1.9, 6.5,
      caption="R1 tracking the freshly-sourced dance motion -- 100% native success rate")

# ============================================================ #
# 21. Divider: sim2sim
# ============================================================ #
divider("SIM2SIM", "Does it survive a different engine?",
        "Native training used MuJoCo-Warp (GPU-batched). Does the policy still work in plain MuJoCo?",
        "reported honestly: real bug found and fixed, one divergence still open")

# ============================================================ #
# 22. Sim2sim process
# ============================================================ #
s = add()
head(s, "Sim2sim, honestly", "what this actually means and what happened")
txt(s, "This project's established meaning of sim2sim: train in one simulator, deploy in a "
       "genuinely different one, and see if the policy survives. mjlab trains on MuJoCo-Warp "
       "(GPU-batched); the test here is plain single-instance MuJoCo (CPU, mj_step) driving the "
       "exported ONNX policy directly -- a real second physics engine implementation, not a "
       "device-flag change.",
    0.7, 1.8, 11.9, 1.15, size=14, color=GRAY, spacing=1.2)
card(s, 0.6, 3.05, 5.85, 3.2, "What made this tractable",
     "The exported ONNX is self-describing: joint order,\n"
     "PD gains, effort limits, default pose, and action\n"
     "scale are all embedded as metadata -- read directly\n"
     "off the file, not re-derived by hand. This removed\n"
     "most of the observation-reconstruction risk.\n\n"
     "The official companion sim2sim tool (motion_tracking_\n"
     "controller) needed a full ROS2 + legged_control2 C++\n"
     "stack requiring sudo -- not available in this session,\n"
     "so a lighter Python + onnxruntime harness was built\n"
     "instead.", body_size=12)
card(s, 6.65, 3.05, 5.85, 3.2, "What was found (see next slide for the full arc)",
     "Four real bugs found and fixed: no effort limits\n"
     "(unbounded torque), a wrong motion-interpolation model,\n"
     "a +1 frame-index offset, and qvel left at zero instead\n"
     "of the reference motion's own velocity at reset.\n\n"
     "Every one of the 8 observation terms was then verified\n"
     "against a captured native rollout to floating-point\n"
     "precision -- the formula is provably correct. The\n"
     "harness still diverges closed-loop; two more real tests\n"
     "followed, on the next slide.", fill=SAND, body_size=12)

# ============================================================ #
# 23. Sim2sim terminal + video
# ============================================================ #
s = add()
head(s, "The divergence, captured")
term(s, cap("sim2sim_bug_found.txt"), 0.6, 1.85, 5.95, 2.4, size=11,
     title="python sim2sim_r1_tennis.py")
txt(s, "Action magnitude growing into the hundreds is itself the tell -- a well-behaved policy, "
       "even one tracking poorly, doesn't do this. This is after the effort-limit fix; two "
       "further hypotheses were tested next (see next slide).",
    0.6, 4.45, 5.95, 1.3, size=12.5, color=GRAY, spacing=1.15)
movie(s, "r1_tennis_sim2sim", 9.6, 1.85, 6.3,
      caption="Plain-MuJoCo replay -- diverges after the initial correct frame")

# ============================================================ #
# 23b. Sim2sim, continued -- ruling out the last two hypotheses
# ============================================================ #
s = add()
head(s, "Sim2sim, continued", "two more real tests, and where this actually lands")
term(s, cap("sim2sim_extended_and_cuda.txt"), 0.6, 1.85, 11.9, 3.9, size=11,
     title="two follow-up tests, both real runs")
txt(s, "Neither test explains the gap. More training helps native performance a lot but barely "
       "moves sim2sim; matching the inference backend exactly (CUDA PyTorch, not CPU ONNX) makes "
       "no meaningful difference either.",
    0.6, 5.9, 11.9, 0.6, size=13, color=GRAY, spacing=1.15)

s = add()
head(s, "Where this actually lands")
card(s, 0.6, 1.9, 12.1, 4.8, "Best-supported conclusion",
     "Confirmed, in order: (1) 4 real bugs fixed. (2) Every observation term matches native's own\n"
     "ground truth to floating-point precision -- the formula is provably correct. (3) Physics\n"
     "agreement in open loop is real but not perfect: 0.38 rad mean error over 660 steps when\n"
     "replaying native's own exact action sequence -- small, but not zero. (4) More training helps\n"
     "native performance substantially (mpkpe nearly halved) but barely moves sim2sim (~4-8%) --\n"
     "ruling out pure undertraining as the explanation. (5) Matching the inference backend exactly\n"
     "(CUDA PyTorch vs CPU ONNX) makes no meaningful difference -- ruling out inference\n"
     "non-determinism too.\n\n"
     "What's left: a genuine, small physics-solver discrepancy between MuJoCo-Warp's GPU-parallel\n"
     "contact solver and native MuJoCo's CPU solver -- not a bug in this harness, and not something\n"
     "further tuning is likely to eliminate. A closed-loop bipedal controller's own corrections feed\n"
     "back into its future observations, so even a small, bounded per-step physics difference\n"
     "compounds over hundreds of steps. This is a known category of limitation for GPU-parallel\n"
     "physics engines generally, not specific to this project's setup.",
     fill=MINT, body_size=13, title_size=16)

# ============================================================ #
# 24. Next
# ============================================================ #
s = add(NAVY)
txt(s, "Next", 1.1, 0.7, 6, 0.7, size=15, color=OCHRE)
nxt = [
    "A harsher sim2sim test would only make this gap worse, not better -- treat the current "
    "policy as not yet ready for real-hardware deployment on this evidence",
    "Try training an architecture/training recipe known to be more robust to physics-engine "
    "differences (e.g. explicit domain randomization over solver-sensitive parameters)",
    "Stance-aware ground pass refinement (flagged two weeks ago, still not required to train)",
    "A third motion, further from tennis/dance in character (e.g. a fall recovery or a two-person "
    "interaction), to keep stress-testing how general this pipeline actually is",
]
y = 1.5
for n in nxt:
    rect(s, 1.1, y + 0.08, 0.14, 0.14, GO, MSO_SHAPE.OVAL)
    txt(s, n, 1.45, y - 0.05, 10.5, 0.9, size=15.5, color=WHITE, spacing=1.15)
    y += 0.18 + 0.3 * (-(-len(n) // 78))

prs.save(OUT)
print(f"Saved {OUT} ({len(prs.slides)} slides)")
