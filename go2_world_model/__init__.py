"""CPU/GPU inference for the Scaling Quadruped World Models checkpoint family."""
import json
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file


class ResidualBody(nn.Module):
    def __init__(self, features, width, depth):
        super().__init__()
        self.inp = nn.Linear(features, width)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())
            for _ in range(depth - 1)
        )
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 286)

    def forward(self, x):
        x = self.inp(x)
        for block in self.blocks:
            x = x + block(x)
        return self.head(self.norm(x))


class Go2Model(nn.Module):
    def __init__(self, width, depth, inputs):
        super().__init__()
        if inputs not in {"proprio", "terrain-scan"}:
            raise ValueError("Expected proprio or terrain-scan input configuration")
        self.inputs = inputs
        self.enc = nn.GRU(42, 128, batch_first=True)
        self.body = ResidualBody(182 + (187 if inputs == "terrain-scan" else 0), width, depth)
        self.cell = nn.GRUCell(256, 128)
        for group, count in [("f", 30), ("a", 12), ("d", 30)]:
            self.register_buffer("mu_" + group, torch.zeros(count))
            self.register_buffer("sd_" + group, torch.ones(count))

    def encode(self, observations, commands):
        readings = (observations - self.mu_f) / self.sd_f
        actions = (commands - self.mu_a) / self.sd_a
        return self.enc(torch.cat((readings, actions), -1))[1][-1]

    def step(self, observation, command, previous, hidden, scan=None):
        features = [(observation - self.mu_f) / self.sd_f,
                    (command - self.mu_a) / self.sd_a, (previous - self.mu_a) / self.sd_a]
        if scan is not None:
            features.append(scan)
        out = self.body(torch.cat([*features, hidden], -1))
        delta = out[:, :30]
        updated = observation + self.mu_d + self.sd_d * delta
        up = updated[:, :3]
        updated = torch.cat([up / up.norm(dim=1, keepdim=True).clamp_min(1e-6), updated[:, 3:]], -1)
        return delta, updated, self.cell(out[:, 30:], hidden)

    @torch.inference_mode()
    def forecast(self, history, history_actions, future_actions, scans=None):
        """Return [batch, T, 30] readings one tick after each supplied command."""
        if history.ndim != 3 or history.shape[-1] != 30 or history.shape[1] < 2:
            raise ValueError("history must have shape [B, K+1, 30] with K >= 1")
        b, n, _ = history.shape
        if history_actions.shape != (b, n-1, 12):
            raise ValueError("history_actions must have shape [B, K, 12]")
        if future_actions.ndim != 3 or future_actions.shape[0] != b or future_actions.shape[2] != 12:
            raise ValueError("future_actions must have shape [B, T, 12]")
        t = future_actions.shape[1]
        if self.inputs == "terrain-scan" and (scans is None or scans.shape != (b, t, 187)):
            raise ValueError("terrain-scan models require aligned scans of shape [B, T, 187]")
        if self.inputs == "proprio" and scans is not None:
            raise ValueError("proprio models take history and joint targets only")
        if t == 0:
            return history[:, :0]
        hidden = self.encode(history[:, :-1], history_actions)
        reading, previous = history[:, -1], history_actions[:, -1]
        predictions = []
        for i, command in enumerate(future_actions.unbind(1)):
            _, reading, hidden = self.step(reading, command, previous, hidden,
                                           None if scans is None else scans[:, i])
            predictions.append(reading)
            previous = command
        return torch.stack(predictions, dim=1)


def load_model(directory, device="cpu"):
    """Load a release directory containing a config and a SafeTensors or original PyTorch checkpoint."""
    directory = Path(directory)
    cfg = json.loads((directory / "config.json").read_text())
    if (cfg["format_version"], cfg["architecture"]) != (1, "go2_residual_gru"):
        raise ValueError("Unsupported model format")
    model = Go2Model(cfg["width"], cfg["depth"], cfg["inputs"])
    filename = cfg.get("weight_file")
    if filename is None:
        filename = "model.pt" if (directory / "model.pt").is_file() else "model.safetensors"
    if filename == "model.pt":
        state = torch.load(directory / filename, map_location="cpu", weights_only=True)["model"]
    elif filename == "model.safetensors":
        state = load_file(directory / filename)
    else:
        raise ValueError(f"Unknown checkpoint filename: {filename}")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
