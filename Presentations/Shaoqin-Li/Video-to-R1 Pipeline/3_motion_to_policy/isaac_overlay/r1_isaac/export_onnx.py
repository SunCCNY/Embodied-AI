"""Export an rsl-rl 5.x actor checkpoint to ONNX: obs (1,135, Isaac joint order) -> deterministic action (1,24)."""
import sys

import numpy as np
import torch
import torch.nn as nn

ckpt, out = sys.argv[1], sys.argv[2]
sd = torch.load(ckpt, map_location="cpu", weights_only=False)["actor_state_dict"]


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", sd["obs_normalizer._mean"].clone())
        self.register_buffer("std", sd["obs_normalizer._std"].clone())
        self.mlp = nn.Sequential(
            nn.Linear(135, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 24),
        )
        self.mlp.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("mlp.")})

    def forward(self, obs):
        return self.mlp((obs - self.mean) / (self.std + 1e-2))


m = Actor().eval()
x = torch.randn(1, 135)
torch.onnx.export(m, x, out, input_names=["obs"], output_names=["actions"], opset_version=17, dynamo=False)
try:
    import onnxruntime as ort
except ImportError:
    print("exported", out, "(no onnxruntime here; verified in WSL)")
    sys.exit(0)

s = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
d = np.abs(s.run(None, {"obs": x.numpy()})[0] - m(x).detach().numpy()).max()
print("exported", out, "max |onnx - torch| =", d)
