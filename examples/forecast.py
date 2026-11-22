"""Forecast one held-out flat-ground sequence and optionally plot two observables."""
import argparse
from pathlib import Path
import numpy as np
import torch
from go2_world_model import load_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = load_model(args.checkpoint)
    with np.load(Path(__file__).with_name("flat-validation.npz")) as data:
        x = torch.from_numpy(data["observation"][:1])
        a = torch.from_numpy(data["commands"][:1])
        sc = torch.from_numpy(data["flat_scan_value"][:1,50:,None]).expand(-1,-1,187)
    pred = model.forecast(x[:,:51], a[:,:50], a[:,50:], sc if model.inputs == "terrain-scan" else None)
    actual = x[:,51:]
    print(f"{pred.shape[1]} ticks, joint-angle RMSE {((pred[:,:,6:18]-actual[:,:,6:18])**2).mean().sqrt():.5f} rad")
    if args.plot:
        import matplotlib.pyplot as plt
        t = np.arange(1,151)/50
        fig, axes = plt.subplots(2,1,sharex=True,layout="constrained")
        for ax, i, units in zip(axes,[3,6],["Body gyro x (rad/s)", "Front-left hip (rad)"]):
            ax.plot(t, actual[0,:,i], label="Recorded simulation")
            ax.plot(t, pred[0,:,i], label="Forecast")
            ax.axhline(float(x[0,50,i]), ls=":", color="gray", label="Persistence")
            ax.set_ylabel(units)
        axes[0].legend(); axes[-1].set_xlabel("Forecast time (s)")
        fig.savefig(args.plot)
