# Scaling Quadruped World Models

**Ramakrishna Kompella · Gödel Machines**

Forecast 30 joint and body readings at 50 Hz. The 20 scan-conditioned checkpoints cover five widths and four training exposures. Five additional checkpoints forecast from proprioception and joint commands after projection-only adaptation. The model outputs joint angles, joint velocities, body gyro and the body-up vector. Code and examples run offline with local weights. [Checkpoint repository](https://huggingface.co/GoedelMachines/go2-world-models) · [Paper](docs/paper.pdf)

## First forecast

Install with Python 3.10 or later in a virtual environment from this repository’s root, then download the two files for the adapted model. For an exact replay, set the Hub CLI’s `--revision` to the commit you used or verify the weights against the independent [checkpoint manifests and SHA-256 hashes](checkpoints.json).

```sh
python -m pip install '.[hub,analysis]'
hf download GoedelMachines/go2-world-models --include 'proprio/config.json' 'proprio/model.pt' 'proprio/model.safetensors' 'manifest.json' --local-dir checkpoints
python examples/forecast.py checkpoints/proprio --plot forecast.png
python examples/evaluate.py checkpoints/proprio
```

The example runs on a recorded, held-out **simulated** flat-ground segment. The evaluator runs all 32 validation worlds. The adapted flagship gives endpoint NRMSE of 0.613 and 1.009 at one and three seconds. The same loader accepts a local root or sweep directory for a scan-conditioned model. Those examples automatically use the aligned terrain measurement at each step.

```sh
hf download GoedelMachines/go2-world-models --include 'config.json' 'model.pt' 'model.safetensors' --local-dir checkpoints
python examples/forecast.py checkpoints
python examples/evaluate.py checkpoints
python examples/evaluate.py checkpoints --zero-scan
python examples/adapt.py checkpoints rebuilt-proprio
python examples/evaluate.py rebuilt-proprio
```

With the recorded scan the flagship scores 0.131 and 0.136. With zero scans the scores are 0.831 and 0.948. Adaptation starts from the original scan weights, removes the scan columns, and updates only the input projection for 32 steps. Its 64 training segments and 32 validation segments have disjoint world IDs. Both sets retain their original start ticks. Each flat scan consists of 187 copies of the supplied scalar. For arbitrary terrain supply all 187 encoded values.

## Input contract

`model.forecast(history, history_actions, future_actions, scans=None)` takes FP32 PyTorch tensors on the model’s device and returns `[B,T,30]` in physical units. Use the history and target order in [schema.json](schema.json). The 50 historical commands drive the first 50 readings to the following readings. Forecast row zero is the reading one tick after future command zero. For the scan family `scans[b,t]` is the terrain encoding at the pose where future command `t` takes effect. These future scans were available in the simulated evaluation. To forecast on robot logs without scans, select a proprioceptive checkpoint.

## Project page

The static project page, three synchronized real Go2 / 120M neural / MuJoCo comparison videos, the original camera recordings, experimental plots, paper, and citation are in [docs](docs/index.html). The site works from a local checkout. For GitHub Pages use the `main` branch and `/docs` publishing source. To serve a local preview, run `python -m http.server --directory docs 8000` and open localhost on port 8000.
