# Online forward-backward training

`scripts/fb/fb_training.py` trains the local `metamotivo.fb.FBAgent` from scratch
using only simulator interaction. No dataset or pretrained model is needed.
The FB and actor losses are Metamotivo's implementations, imported from the
sibling checkout rather than copied. Metamotivo's CC BY-NC 4.0 license applies
to that dependency; see `metamotivo/LICENSE`.

## Setup and launch

From `air-hockey-rl`, use your existing simulator/PyTorch environment and install:

```bash
python -m pip install -e ../metamotivo tensorboard pytest wandb
python -m scripts.fb.fb_training --args-file configs/fb/fb_online_realplant.yaml
```

The tested local environment is `.venv` (inherits the machine's CUDA PyTorch).
For a fresh environment, install the simulator dependencies from this repo as
well. Box2D must use the repo's `pybox2d` Git source, not the old PyPI 2.3.2
release, which fails on Python 3.12. On this ARM host, build it with GCC/G++:

```bash
CC=gcc CXX=g++ python -m pip install 'box2d @ git+https://github.com/pybox2d/pybox2d.git'
```

All YAML values can be overridden on the command line, for example:

```bash
.venv/bin/python -m scripts.fb.fb_training \
  --args-file configs/fb/fb_online_realplant.yaml \
  --output-dir runs/fb/my_seed1 --seed 1 --device cuda
```

Simulator paths are resolved from the repository root. Output paths and
`--args-file` are relative to the working directory. An existing run directory
with `args.yaml` is rejected to prevent accidental overwrites.

## Collection and learning

The preset uses `sysid_v2_hist4_realplant.yaml` plus the existing external
**two-control-step action latency** wrapper, in both training and evaluation.
`--action-latency-steps 0` disables only that wrapper. The simulator is unchanged.
The 30 history features are augmented with the last issued action (32 inputs),
matching TD3's policy input convention. The latent adds task conditioning inside
Metamotivo's networks. The hidden action queue is not fully observed; this
retains the existing TD3-style partial observability.

The first 5,000 steps use uniform random actions. Subsequent actions come from
the noisy FB actor with a random spherical latent held for up to 150 steps and
resampled at episode boundaries. Training samples a uniform online ring buffer;
50% of training latents use achieved-state backward embeddings, as in upstream
FB. Neither task rewards nor task-inferred latents guide collection or gradients.
The default is one FB+actor update per interaction after warmup, batch size 256,
z dimension 50, and width 256. These are starting hyperparameters, not a tuned
replacement for the TD3 recipe. FB's pairwise loss costs quadratic batch memory.

True failures stop bootstrapping, including physical failures this env calls
truncations. Only a pure `max_timesteps_exceeded` truncation bootstraps from its
actual final observation. Replay stores commanded actions and final observations
before reset, with zero previous action at the start of each episode.

## Tasks and evaluation

A shared online replay sample supplies `z = project(sum reward * B(next_obs))`:

- **Juggle:** the configured simulator reward.
- **Touch/contact:** +1 on a step with a paddle–puck collision, using the simulator's
  contact event field rather than noisy position thresholds. Contact and simple
  touching are treated as the same task here.
- **Paddle reach:** a Gaussian position reward centered on the requested table-frame
  `(x, y)`, with width `paddle_goal_radius` (default 5 cm). Default target is
  `(0.70, 0.0)` metres. Success means entering that radius at least once. This is
  position reaching, not orientation or velocity control.

All three are evaluated under the **same juggling spawn distribution and episode
end conditions**, with the puck present. These are reward relabelings for a shared
FB model, not scores on the repo's separate `puck_touch` / no-puck sparse
`paddle_reach_position` benchmarks, which change starts, observations or dynamics.
A reach target need not be appended to observations: it changes the inferred z.
Choose targets inside the simulator's paddle workspace.

Evaluation reports returns, success rates, length, latent norm, reward support in
inference samples, and final distance to the requested paddle target. If there
are no contacts in the inference sample, the touch latent is zero; that indicates
missing experience, not a learned touch skill. Reward-free exploration may need
many interactions to discover useful juggling/contact transitions.

Evaluation uses deterministic actions, fixed held-out seeds and isolated RNG
state. JSONL metrics, resolved args, simulator config, and optional TensorBoard
logs are in the output directory. Periodic and final checkpoints contain:

- Upstream agent model/targets and optimizer state.
- `policy/`: standalone `FBModel.load` compatible inference export.
- `reward_z.pt`: juggling latent.
- `inference_samples.pt`: observations and reward features for re-inferring tasks.
- `run.json`: step/update counts, arguments and simulator configuration.

Checkpoints do **not** contain the full replay or environment/RNG state; exact
training resume is not implemented. The upstream agent loader can restore
weights/optimizers for future warm-start workflows.

Evaluate a saved policy, optionally selecting a different paddle position:

```bash
.venv/bin/python -m scripts.fb.fb_training \
  --eval-checkpoint runs/fb/realplant_seed0/final \
  --device cuda --eval-episodes 10 --paddle-goal-x 0.75 --paddle-goal-y 0.10
```

The saved simulator configuration is used for checkpoint evaluation. Architecture
and training options are not applied to the loaded policy.

## Tests

```bash
.venv/bin/python -m pytest scripts/fb/tests -q
.venv/bin/python -m scripts.fb.fb_training \
  --args-file configs/fb/fb_online_realplant.yaml \
  --output-dir runs/fb/smoke --total-timesteps 1000 \
  --learning-starts 128 --batch-size 64 --buffer-size 2000 \
  --eval-every-steps 500 --eval-episodes 2 --checkpoint-every-steps 500
```

Integration tests perform real CPU/CUDA FB updates on this simulator, wrap the
replay buffer, check finite metrics and changed weights, reload both checkpoint
formats, and reproduce evaluation. Unit checks cover termination semantics,
last-action observations, latency queue reset, reward relabeling, and RNG isolation.
Short smoke runs validate execution, not convergence or useful skill quality.

## W&B and performance GIFs

The preset enables online W&B logging to `air-hockey-fb` using your configured
W&B account/server. Override `--wandb-entity` / `--wandb-project` /
`--wandb-name` as needed. Use `--no-wandb` for local-only runs. The URL and run
ID are saved in `wandb_run.json`; training errors are reported as failed runs.
Install `wandb` in addition to the dependencies above.

Evaluation runs after warmup (5,000 steps), every 10,000 steps, and at the end.
Ten fixed-seed episodes per task provide scalar metrics; the first two episodes
per task become 160-pixel-wide, 20 fps GIFs, with task, step, reward and return
labels. Reaching videos show a green goal circle. GIFs appear under W&B's
`videos/*` panels and locally in:

- `gifs/index.html`: refreshing gallery of all recorded evaluations.
- `gifs/step_000010000/{juggle,touch,reach}_ep00.gif`: historical clips.
- `gifs/latest/`: stable filenames for the newest clips.

`--gif-episodes 0` disables recording; values above `eval_episodes` record all
available evaluation episodes. Checkpoints are saved every 25,000 steps. For
checkpoint evaluation, `--gif-episodes 2` writes a separate `evaluation_gifs/`
gallery inside that checkpoint. GIF capture uses the evaluation trajectories
and does not alter training RNG state.
