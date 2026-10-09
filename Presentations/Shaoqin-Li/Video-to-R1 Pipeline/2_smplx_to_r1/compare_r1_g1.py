"""R1 vs G1 on identical LAFAN1 clips -- the control the week-26 sweep was missing.

Week 26 measured the R1 alone. That makes every number unanchored: a reading like
"ground: -45.63 cm below the floor" cannot distinguish "the R1 is a poor retargeting
target" from "kinematic retargeting does this to every robot, the G1 included".

Both robots are run through the same script, the same clips, the same window and the
same four checks. The only thing that differs is `tgt_robot`, and therefore which
`*_to_g1.json` / `*_to_r1.json` config GMR loads. The G1 config is upstream and
unmodified, which is what makes it a control rather than another variable.

    python compare_r1_g1.py
    python compare_r1_g1.py --out measurements/week27_r1_vs_g1.txt
"""
import argparse
import json

R1_JSON = "measurements/week27_r1_recheck.json"
G1_JSON = "measurements/week27_g1_baseline.json"
R1_WEEK26 = "measurements/week26_posture_sweep.json"

# metric key, label, unit, how many decimals, is "closer to zero" better?
METRICS = [
    ("lowest_cm",             "lowest point",   "cm",    2),
    ("contact_pct",           "floor contact",  "%",     1),
    ("peak_joint_speed",      "peak speed",     "rad/s", 1),
    ("limit_violation_frames","limit viol.",    "fr",    0),
    ("ik_fps",                "IK rate",        "fps",   1),
]


def load(p):
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    r1, g1 = load(R1_JSON), load(G1_JSON)
    lines = []
    def w(s=""):
        print(s); lines.append(s)

    # --- control: did the geom-selector change move any R1 number? -------------
    try:
        old = load(R1_WEEK26)
        drift = []
        for cat in old:
            if cat not in r1:
                drift.append(f"{cat}: missing from recheck"); continue
            for k, *_ in METRICS:
                a, b = old[cat].get(k), r1[cat].get(k)
                if a is None or b is None:
                    continue
                if abs(float(a) - float(b)) > (1e-6 if k != "ik_fps" else 1e9):
                    drift.append(f"{cat}.{k}: week26 {a} -> recheck {b}")
        w("CONTROL -- R1 re-run under the physics-based geom selector")
        w("  (ik_fps is timing noise and is not compared)")
        if drift:
            w("  R1 NUMBERS MOVED:")
            for d in drift:
                w(f"    {d}")
        else:
            w("  every R1 metric identical to measurements/week26_posture_sweep.json")
        w("")
    except FileNotFoundError:
        w("CONTROL -- week26 baseline not found, skipped\n")

    cats = [c for c in r1 if c in g1]
    w(f"R1 vs G1 -- {len(cats)} LAFAN1 categories, identical clips and window")
    w("")

    for key, label, unit, dp in METRICS:
        w(f"  {label} ({unit})")
        w(f"    {'category':18s} {'R1':>10s} {'G1':>10s} {'R1-G1':>10s}")
        w("    " + "-" * 50)
        dr1 = dg1 = 0.0
        for c in cats:
            a, b = float(r1[c][key]), float(g1[c][key])
            dr1 += a; dg1 += b
            w(f"    {c:18s} {a:10.{dp}f} {b:10.{dp}f} {a-b:+10.{dp}f}")
        w(f"    {'mean':18s} {dr1/len(cats):10.{dp}f} {dg1/len(cats):10.{dp}f} "
          f"{(dr1-dg1)/len(cats):+10.{dp}f}")
        w("")

    # --- the three claims that need a baseline to mean anything ----------------
    w("What the baseline settles")
    w("")
    v1 = sum(int(r1[c]["limit_violation_frames"]) for c in cats)
    v2 = sum(int(g1[c]["limit_violation_frames"]) for c in cats)
    n = sum(int(r1[c].get("frames", 600)) for c in cats)
    w(f"  joint limits   R1 {v1} violating frames, G1 {v2}, out of ~{n} each")
    verdict = ("both zero: a property of the method, not of the R1"
               if v1 == v2 == 0 else "they differ -- robot-specific")
    w(f"                 -> {verdict}")

    # A "which categories cross -10 cm" test is dominated by whichever clips sit
    # near the threshold, so report the magnitudes as well -- they are what say
    # whether the two robots sink by the same amount.
    sink1 = [c for c in cats if float(r1[c]["lowest_cm"]) < -10]
    sink2 = [c for c in cats if float(g1[c]["lowest_cm"]) < -10]
    m1 = sum(float(r1[c]["lowest_cm"]) for c in cats) / len(cats)
    m2 = sum(float(g1[c]["lowest_cm"]) for c in cats) / len(cats)
    worst = max(cats, key=lambda c: abs(float(r1[c]["lowest_cm"]) - float(g1[c]["lowest_cm"])))
    w(f"  floor sinking  R1 sinks >10cm in {sorted(sink1)}")
    w(f"                 G1 sinks >10cm in {sorted(sink2)}")
    w(f"                 mean lowest point  R1 {m1:+.2f} cm   G1 {m2:+.2f} cm   "
      f"difference {m1 - m2:+.2f} cm")
    w(f"                 largest single gap: {worst} "
      f"(R1 {float(r1[worst]['lowest_cm']):+.2f} vs G1 {float(g1[worst]['lowest_cm']):+.2f} cm)")
    if abs(m1 - m2) < 2.0:
        w("                 -> BOTH robots sink, by the same amount on average.")
        w("                    Inherent to kinematic retargeting, not an R1 defect.")
        if set(sink1) != set(sink2):
            w("                    (The category lists differ only because clips sit either")
            w("                     side of the -10 cm line -- do not read that as a robot")
            w("                     difference; the means are what matter.)")
    else:
        w(f"                 -> the robots differ by {abs(m1 - m2):.2f} cm on average")

    fast1 = [c for c in cats if float(r1[c]["peak_joint_speed"]) > 20]
    fast2 = [c for c in cats if float(g1[c]["peak_joint_speed"]) > 20]
    w(f"  servo speed    R1 over 20 rad/s in {sorted(fast1)}")
    w(f"                 G1 over 20 rad/s in {sorted(fast2)}")
    verdict = ("same set: the motion demands it, not the robot"
               if set(fast1) == set(fast2) else "differs by robot")
    w(f"                 -> {verdict}")

    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
