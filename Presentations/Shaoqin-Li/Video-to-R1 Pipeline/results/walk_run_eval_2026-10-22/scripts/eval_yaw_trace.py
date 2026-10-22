import sys, math, torch
sys.path.insert(0, __import__("os").path.dirname(__file__))
from eval_walk_run import make_env, load_policy, set_cmd
from eval_stop_flight import yaw
ckpt, n = sys.argv[1], int(sys.argv[2])
sched = [(3,0),(7,0.5),(11,1.0),(15,1.5),(19,2.0),(23,0)]
w, a = make_env(n, False, "cuda:0"); _, pol = load_policy(w, a, ckpt, "cuda:0")
r = w.unwrapped.scene["robot"]; dt = w.unwrapped.step_dt
obs, _ = w.reset(); y0 = yaw(r.data.root_link_quat_w)[0].item(); seg = 0
with torch.inference_mode():
  for t in range(int(23/dt)):
    while t*dt >= sched[seg][0]: seg += 1
    set_cmd(w, torch.full((n,), float(sched[seg][1]), device="cuda:0"))
    obs, _, d, _ = w.step(pol(obs))
    if t % 50 == 49:
      ys = torch.rad2deg(yaw(r.data.root_link_quat_w))
      print(f"t={((t+1)*dt):5.1f} cmd={sched[seg][1]:.1f} env0_yaw={ys[0].item():7.1f} " + (f"spread={ys.std().item():.1f}" if n>1 else ""))
