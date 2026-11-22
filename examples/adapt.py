"""Repeat the 32 input-projection updates using the saved flat-ground cohort."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import save_file
from go2_world_model import Go2Model, load_model


def adapt(parent_dir, output):
    cfg = json.loads((parent_dir / "config.json").read_text())
    parent = load_model(parent_dir)
    if parent.inputs != "terrain-scan":
        raise ValueError("Start with the longest-trained scan checkpoint for this width")
    torch.manual_seed(1)
    model = Go2Model(cfg["width"], cfg["depth"], "proprio")
    weights = parent.state_dict()
    projection = weights["body.inp.weight"]
    weights["body.inp.weight"] = torch.cat((projection[:, :54], projection[:, 241:]), dim=1)
    model.load_state_dict(weights)
    del parent, weights, projection
    model.requires_grad_(False)
    model.body.inp.requires_grad_(True)
    opt = torch.optim.AdamW(model.body.inp.parameters(), lr=1e-4, weight_decay=0)
    with np.load(Path(__file__).with_name("flat-train.npz")) as data:
        x, a = (torch.from_numpy(data[k]) for k in ["observation", "commands"])
    gen = torch.Generator().manual_seed(3)
    for _ in range(32):
        world = torch.randint(len(x), (16, 1), generator=gen)
        start = torch.randint(147, (16, 1), generator=gen)
        ticks = start + torch.arange(55)
        obs, act = x[world, ticks], a[world, ticks[:, :-1]]
        teacher = hidden = model.encode(obs[:, :50], act[:, :50])
        reading = obs[:, 50]
        teacher_loss = rollout_loss = 0.
        for i in range(50, 54):
            cmd, prev = act[:, i], act[:, i-1]
            delta, _, teacher = model.step(obs[:, i], cmd, prev, teacher)
            truth = (obs[:, i+1] - obs[:, i] - model.mu_d) / model.sd_d
            teacher_loss += (delta-truth).square().mean()
            _, reading, hidden = model.step(reading, cmd, prev, hidden)
            error = (reading - obs[:, i+1]) / model.sd_f
            rollout_loss += F.smooth_l1_loss(error, torch.zeros_like(error))
        opt.zero_grad(set_to_none=True)
        (teacher_loss / 4 + rollout_loss / 4).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
    output.mkdir(parents=True, exist_ok=False)
    cfg.update(inputs="proprio", scan_dim=0, total_parameters=sum(p.numel() for p in model.parameters()))
    (output/"config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    save_file(model.state_dict(), str(output/"model.safetensors"), metadata={"format":"pt"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent_dir", type=Path)
    parser.add_argument("new_output_dir", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(12)
    adapt(args.parent_dir, args.new_output_dir)
