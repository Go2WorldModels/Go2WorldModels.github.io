"""Score a local checkpoint and persistence on all 32 published validation worlds."""
import argparse
from pathlib import Path
import numpy as np
import torch
from go2_world_model import load_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--zero-scan", action="store_true", help="zero the scan at every forecast tick")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = load_model(args.checkpoint)
    with np.load(Path(__file__).with_name("flat-validation.npz")) as data:
        obs, act = (torch.from_numpy(data[k]) for k in ["observation","commands"])
        scans = torch.from_numpy(data["flat_scan_value"][:,50:,None]).expand(-1,-1,187)
    if args.zero_scan and model.inputs != "terrain-scan":
        parser.error("--zero-scan evaluates the original scan model")
    scan_input = (torch.zeros_like(scans) if args.zero_scan else scans) if model.inputs == "terrain-scan" else None
    pred = model.forecast(obs[:,:51], act[:,:50], act[:,50:], scan_input)
    for name, result in [("model",pred), ("persistence",obs[:,50:51])]:
        error = ((result-obs[:,51:])/model.sd_f).square().mean((0,2)).sqrt()
        print(f"{name}: 1 s {error[49]:.4f}, 3 s {error[149]:.4f}")
