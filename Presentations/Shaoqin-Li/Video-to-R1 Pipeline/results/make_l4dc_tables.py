"""LaTeX table bodies for the L4DC paper from l4dc_2026-10-19/*.json (run from results/)."""
import glob, json, os, statistics as st
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "l4dc_2026-10-19")
M = [("tennis", "Tennis serve"), ("woman_home_fixed", "Dance (home)"), ("rdne_boxing_moving", "Shadow boxing"),
     ("street_hiphop_moving", "Street dance"), ("dancer_truespeed", "Dance (studio)")]
R = {}
for f in glob.glob(os.path.join(DATA, "*.json")):
    t = os.path.basename(f)[:-5]
    for m, _ in M:
        if t.startswith(m + "_"):
            arm, s = t[len(m) + 1:].split("_s"); R[(m, arm, int(s))] = json.load(open(f))
S = (42, 43, 44)
n = lambda r, sc: sum(x["survived"] for x in r[sc])
pct = lambda r, sc: 100 * n(r, sc) / len(r[sc])
dl = lambda d: f"train_DR+delay{d}ms"
def rng(v): return f"{st.mean(v):.1f}" + (f" ({min(v)}--{max(v)})" if min(v) != max(v) else "")
print("% attribution table")
for m, name in M:
    b = R[(m, "base", 42)]
    c = [[n(R[(m, a, s)], dl(d)) for s in S] for a in ("ctrl", "dly") for d in (40, 60)]
    print(f"{name} & {n(b, dl(40))} & {rng(c[0])} & {rng(c[2])} & {n(b, dl(60))} & {rng(c[1])} & {rng(c[3])} \\\\")
print("% cost / perturbation table (mean over 3 seeds; c / d)")
SC = ["train_DR+push3x", "train_DR+fric0.15", "train_DR+kp-20%", "train_DR+payload2kg", "train_DR+no_lin_vel"]
def err(r):
    xs = [x["mean_err"] for x in r["train_DR"] if x["survived"]]; return st.mean(xs)
tot = {k: [] for k in ["err", "nom"] + SC}
for m, name in M:
    cells = []
    e = [st.mean(err(R[(m, a, s)]) for s in S) for a in ("ctrl", "dly")]
    nm = [st.mean(pct(R[(m, a, s)], "train_DR") for s in S) for a in ("ctrl", "dly")]
    cells.append(f"{e[0]:.2f} / {e[1]:.2f}"); cells.append(f"{nm[0]:.0f} / {nm[1]:.0f}")
    tot["err"].append(e); tot["nom"].append(nm)
    for sc in SC:
        v = [st.mean(pct(R[(m, a, s)], sc) for s in S) for a in ("ctrl", "dly")]
        tot[sc].append(v); cells.append(f"{v[0]:.0f} / {v[1]:.0f}")
    print(f"{name} & " + " & ".join(cells) + " \\\\")
print("\\midrule")
cells = []
for k in ["err", "nom"] + SC:
    c = st.mean(v[0] for v in tot[k]); d = st.mean(v[1] for v in tot[k])
    cells.append(f"{c:.2f} / {d:.2f}" if k == "err" else f"{c:.0f} / {d:.0f}")
print("Mean over motions & " + " & ".join(cells) + " \\\\")
