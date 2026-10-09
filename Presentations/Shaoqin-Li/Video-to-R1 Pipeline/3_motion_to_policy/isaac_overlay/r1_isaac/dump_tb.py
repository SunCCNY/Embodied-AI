import sys, json
from tensorboard.backend.event_processing import event_accumulator as ea
d = sys.argv[1]
a = ea.EventAccumulator(d, size_guidance={"scalars": 0}); a.Reload()
out = {t: [(e.step, e.value) for e in a.Scalars(t)] for t in a.Tags()["scalars"]}
print(list(out)[:40], file=sys.stderr)
json.dump(out, open(sys.argv[2], "w"))
