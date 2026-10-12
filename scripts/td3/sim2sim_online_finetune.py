"""Sim2sim test bed for the sim-to-online fine-tuning recipe (Box2D only, no robot).

What this is
------------
The recipe in ``scripts/td3/td3_online_real_finetune.py`` (As et al., "What
Matters for Simulation-to-Online Reinforcement Learning on Real Robots",
arXiv:2602.20220) fine-tunes a sim-trained TD3 policy online. This script runs
the same recipe sim-to-sim so it can be iterated on without the robot:

* **Source**: the 5 hist4 juggle policies trained on the sysid-v2 sim with the
  real-robot workspace (``runs/td3/sysid_v2_real_workspace/hist4_seed<S>/
  juggle_sysid_v2_hist4/checkpoint_1975000/training_state.pth``, sim config
  ``configs/new_juggle/sysid_v2_hist4.yaml``).
* **Targets**: ``--n-envs`` (5) Box2D envs whose physics are sampled uniformly
  inside the ±25 % DR box around the sysid-v2 values of
  ``configs/new_juggle/sysid_v2_hist2.yaml``, on ``SAMPLED_PARAMS``:
  paddle_density, puck_damping and gravity. Everything else (wall / paddle-puck
  restitution, PID gains, hist_len 4, workspace, noise, occlusion, delay, task)
  stays at the source config. Each env is an
  independent in-distribution draw; ``--env-seed`` only seeds the random
  generator so that ``--train`` and the fine-tuning phase use the same 5 envs
  (written to ``<root>/envs.json``; another seed gives another 5 envs).

Per target env it runs three things:

1. ``--train``: a fresh hist4 TD3 policy per seed (5 seeds × 5 envs, 2M steps,
   the recipe of ``configs/td3/sysid_v2_hist/juggle_sysid_v2_hist4.yaml``) with
   ``scripts/td3/td3_training.py``: the in-env baseline / ceiling. Runs are
   named ``sim2sim_hist4_env<E>_seed<S>`` under ``<root>/train/env<E>/``;
   ``tensorboard --logdir <root>/train`` shows them all.
2. Zero-shot: the 5 source policies evaluated in the target env.
3. Online fine-tuning of each source policy in the target env for
   ``--num-online-episodes`` (100) episodes with the recipe, using the exact
   learner code and hyperparameters of the real-robot script
   (``configs/td3/td3_online_real_finetune/juggle_hist4.yaml``: in-place load
   of actor / critics / targets / Adam states, actor LR 1e-5, critic LR 1e-3,
   one online buffer with no sim data, K = round(5·T) critic updates after
   each episode, one actor update per 20 critic updates, Polyak every critic
   update). Default: no warm start (the real script's
   ``--no-warmup-no-sim-data``), so learning starts after the first episode;
   ``--warm-start-episodes N`` restores a warm start (counted inside the 100).
   Only the collector is new: a Box2D episode loop instead of the robot
   orchestrator. Nothing here talks to the robot: the real script is only
   imported for its learner functions (its robot collector is never called),
   and the worker refuses to run unless the env is the Box2D simulator.
   ``tensorboard --logdir <root>/online`` shows the same per-episode curves as
   the real script (``online_finetune/return``, ``critic_loss``,
   ``actor_loss``, ``online_finetune_episode/*``, ``online_finetune_updates/*``,
   ``online_finetune/warm_start_return``) plus a deterministic eval every
   ``--online-eval-every`` rounds (``online_finetune_eval/return``), and the
   real script's critic error ε = Q_φ − Q_MC on each new episode before it is
   trained on (``online_finetune_critic_error/*``: mean / median / |ε| / std,
   histogram, episode × ε heatmap; per-step ε in ``critic_error.jsonl``).
   ``--policy-lrs 1e-5 1e-4 ...`` runs one sweep per actor LR, each in
   ``<root>/online_policy_lr<lr>/``; ``scripts/analysis/aggregate_online_tb.py``
   collapses the seeds to mean ± SE per env for TensorBoard.

``--recycle-data`` is the data-recycling experiment of the paper's Fig. 6: per
(env, seed), ``--recycle-trials`` (4) fine-tuning trials run in sequence, each
from the same source policy with an empty online buffer; trial t ≥ 1 loads the
saved online buffers of trials < t as D0 and samples every batch from
(1 − α) Unif(D0) + α Unif(D_online) (Eq. 16), α rising linearly from
``--recycle-alpha-start`` (0.5) to 1 over ``--recycle-anneal-episodes`` (100)
training rounds. Output in ``<root>/online_recycle_alpha<α0>/trial<t>/env<E>/
hist4_seed<S>/`` (plus ``online_buffer.pt``); ``--recycle-results`` prints the
per-trial table and writes ``figure6_env<E>.png``. ``--envs`` restricts any
fine-tuning mode to some target envs.

``--zero-shot-and-online-fine-tune`` runs 2 and 3 (one job per env × source
seed, in parallel across ``--gpus``) and prints the tables. ``--results``
prints the tables from whatever has finished. Every policy (in-env baseline,
zero-shot source, fine-tuned) is scored with the same deterministic eval:
``--eval-episodes`` episodes on identical episode seeds per env, so the
columns are paired.

Usage (from the repo root)
--------------------------
    # 1. train the in-env baselines (or scripts/td3/run_sim2sim_online_finetune.sh)
    .venv/bin/python -m scripts.td3.sim2sim_online_finetune --train --gpus 0 1

    # 2 + 3. zero-shot eval + online fine-tuning, then the tables
    .venv/bin/python -m scripts.td3.sim2sim_online_finetune --zero-shot-and-online-fine-tune --gpus 0 1

    # tables only (evaluates finished baselines that have no eval yet)
    .venv/bin/python -m scripts.td3.sim2sim_online_finetune --results

    tensorboard --logdir runs/td3/sim2sim_online_finetune

``--dry-run`` prints the sampled envs and the commands without running them.
Re-running ``--train`` skips finished runs (``model.pth`` at the run root);
an unfinished run directory is renamed ``*_incomplete_<time>`` and restarted.
Re-running the fine-tuning skips (env, seed) pairs that already have a
``result.json``; ``--overwrite`` redoes them.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.td3.extras.make_sysid_v2_configs import dr_ranges  # noqa: E402

# Physics parameters drawn per target env (the canonical 3-parameter DR set).
SAMPLED_PARAMS = ["paddle_density", "puck_damping", "gravity"]

DEFAULT_ROOT = "runs/td3/sim2sim_online_finetune"
SYSID_MEANS_CONFIG = "configs/new_juggle/sysid_v2_hist2.yaml"
SOURCE_SIM_CONFIG = "configs/new_juggle/sysid_v2_hist4.yaml"
TRAIN_ARGS_BASE = "configs/td3/sysid_v2_hist/juggle_sysid_v2_hist4.yaml"
FINETUNE_ARGS_FILE = "configs/td3/td3_online_real_finetune/juggle_hist4.yaml"
SOURCE_CHECKPOINT = "runs/td3/sysid_v2_real_workspace/hist4_seed{seed}/juggle_sysid_v2_hist4/checkpoint_1975000/training_state.pth"

EVAL_SEED_BASE = 900_000  # episode seeds of the paired eval: EVAL_SEED_BASE + 1000 * env + episode
RESULT_FILE = "result.json"
BASELINE_EVAL_FILE = "sim2sim_eval.json"


def _repo_path(path: str | os.PathLike) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_yaml(path: str | os.PathLike) -> dict:
    with open(_repo_path(path)) as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, header: str, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for line in header.strip("\n").split("\n"):
            f.write(f"# {line}\n")
        yaml.safe_dump(data, f, sort_keys=False)


# ---------------------------------------------------------------------------
# Target environments
# ---------------------------------------------------------------------------


def sample_envs(root: Path, n_envs: int, env_seed: int, scale_box: Optional[List[float]] = None,
                direct_force: Optional[bool] = None, ranges_from: Optional[str] = None) -> List[dict]:
    """Draw the target envs and write their sim configs (plus ``configs/source_sim.yaml``,
    the source env with the same control mode).

    Each parameter is drawn uniformly in the ±25 % DR box, or with ``scale_box = [lo, hi]``
    in [lo, hi] x its sysid-v2 value (e.g. [2.0, 2.5] = step 6 of the identify_ood_gap
    sweep), or with ``ranges_from`` in the ``random_variable_ranges`` of that sim config (e.g. the
    DR oracle's training box, so targets and oracle share one box). ``direct_force`` sets ``direct_force_control`` in every sim config. Both
    default to what an existing ``envs.json`` recorded. Deterministic in ``env_seed``;
    refuses to silently replace an existing draw made with different settings.
    """
    manifest_path = root / "envs.json"
    old = json.load(open(manifest_path)) if manifest_path.exists() else None
    if scale_box is None and ranges_from is None and old is not None:
        scale_box, ranges_from = old.get("scale_box"), old.get("ranges_from")
    if scale_box is not None and ranges_from is not None:
        raise SystemExit("--scale-box and --ranges-from are mutually exclusive")
    if direct_force is None:
        direct_force = bool(old.get("direct_force", False)) if old is not None else False
    means = _load_yaml(SYSID_MEANS_CONFIG)["air_hockey"]["simulator_params"]
    if ranges_from is not None:
        box = _load_yaml(ranges_from)["air_hockey"]["random_variable_ranges"]
        ranges = {n: sorted(float(v) for v in box[n]) for n in SAMPLED_PARAMS}
        box_text = f"in the random_variable_ranges of {ranges_from}"
    elif scale_box is None:
        ranges = dr_ranges(means, SAMPLED_PARAMS)
        box_text = f"in the ±25 % box around {SYSID_MEANS_CONFIG}"
    else:
        lo, hi = (float(v) for v in scale_box)
        ranges = {n: sorted([round(float(means[n]) * lo, 6), round(float(means[n]) * hi, 6)]) for n in SAMPLED_PARAMS}
        box_text = f"in [{lo:g}, {hi:g}] x their {SYSID_MEANS_CONFIG} values"
    rng = np.random.default_rng(env_seed)
    envs = []
    for e in range(1, n_envs + 1):
        params = {n: round(float(rng.uniform(*ranges[n])), 6) for n in SAMPLED_PARAMS}
        envs.append({"env": e, "params": params, "sim_config": str(root / "configs" / f"env{e}_sim.yaml")})
    source_sim_config = str(root / "configs" / "source_sim.yaml")
    manifest = {"env_seed": env_seed, "n_envs": n_envs, "means_from": SYSID_MEANS_CONFIG, "base_config": SOURCE_SIM_CONFIG,
                "scale_box": scale_box, "ranges_from": ranges_from, "direct_force": direct_force, "source_sim_config": source_sim_config,
                "dr_params": SAMPLED_PARAMS, "ranges": ranges, "means": {n: means[n] for n in SAMPLED_PARAMS}, "envs": envs}
    if old is not None and (old.get("envs") != envs or bool(old.get("direct_force", False)) != direct_force):
        raise SystemExit(f"{manifest_path} holds a different env draw (env_seed {old.get('env_seed')}, n_envs "
                         f"{old.get('n_envs')}, scale_box {old.get('scale_box')}, ranges_from {old.get('ranges_from')}, direct_force {old.get('direct_force', False)}). "
                         "Use the same --env-seed / --n-envs / --scale-box / --ranges-from / --direct-force or another --root.")
    root.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    base = _load_yaml(SOURCE_SIM_CONFIG)
    control = "direct force control (no filter / PID)" if direct_force else "PID control"
    for env in envs:
        cfg = copy.deepcopy(base)
        cfg["air_hockey"]["simulator_params"].update(env["params"])
        cfg["air_hockey"]["simulator_params"]["direct_force_control"] = direct_force
        _write_yaml(Path(env["sim_config"]),
                    f"sim2sim target env{env['env']} (env_seed {env_seed}): {SOURCE_SIM_CONFIG} with "
                    f"{', '.join(SAMPLED_PARAMS)} drawn uniformly {box_text}, {control}. "
                    "Generated by scripts/td3/sim2sim_online_finetune.py; do not edit.", cfg)
    cfg = copy.deepcopy(base)
    cfg["air_hockey"]["simulator_params"]["direct_force_control"] = direct_force
    _write_yaml(Path(source_sim_config), f"sim2sim source env: {SOURCE_SIM_CONFIG} with {control}. "
                "Generated by scripts/td3/sim2sim_online_finetune.py; do not edit.", cfg)
    return envs


def print_envs(manifest_envs: List[dict], ranges: Dict[str, List[float]], means: Dict[str, float]) -> None:
    names = list(SAMPLED_PARAMS)
    short = {"paddle_density": "paddle_dens", "puck_damping": "puck_damp", "side_wall_restitution": "side_wall",
             "end_wall_restitution": "end_wall", "puck_restitution": "puck_rest"}
    header = ["env"] + [short.get(n, n) for n in names]
    rows = [["sysid v2"] + [f"{means[n]:g}" for n in names]]
    rows += [[f"env{e['env']}"] + [f"{e['params'][n]:.4g}" for n in names] for e in manifest_envs]
    rows += [["range"] + [f"{ranges[n][0]:.4g}..{ranges[n][1]:.4g}" for n in names]]
    _print_table("Target envs (sampled physics)", header, rows)


# ---------------------------------------------------------------------------
# Job scheduling (training runs and fine-tuning workers)
# ---------------------------------------------------------------------------


def run_jobs(jobs: List[dict], gpus: List[int], jobs_per_gpu: int, poll_s: float = 5.0) -> None:
    """Run ``jobs`` (dicts with name / cmd / log) with at most ``jobs_per_gpu``
    at a time on each GPU; ``{gpu}`` in a command is replaced by the GPU id."""
    pending = list(jobs)
    running: List[tuple] = []
    slots = {g: jobs_per_gpu for g in gpus}
    failed = []
    while pending or running:
        for item in list(running):
            proc, job, gpu, t0, log = item
            if proc.poll() is None:
                continue
            log.close()
            running.remove(item)
            slots[gpu] += 1
            status = "ok" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
            if proc.returncode != 0:
                failed.append(job["name"])
            print(f"[done ] {job['name']} gpu{gpu} {time.time() - t0:.0f}s {status} (log {job['log']})", flush=True)
        while pending and any(slots[g] > 0 for g in gpus):
            gpu = max(gpus, key=lambda g: slots[g])  # round-robin: the GPU with the most free slots
            job = pending.pop(0)
            cmd = [str(c).replace("{gpu}", str(gpu)) for c in job["cmd"]]
            Path(job["log"]).parent.mkdir(parents=True, exist_ok=True)
            log = open(job["log"], "w")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT, env=env)
            running.append((proc, job, gpu, time.time(), log))
            slots[gpu] -= 1
            print(f"[start] {job['name']} gpu{gpu} ({len(pending)} queued)", flush=True)
        time.sleep(poll_s)
    if failed:
        print(f"{len(failed)} job(s) failed: {', '.join(failed)}", flush=True)


# ---------------------------------------------------------------------------
# 1. In-env baselines (td3_training.py)
# ---------------------------------------------------------------------------


def run_name(env: int, seed: int) -> str:
    return f"sim2sim_hist4_env{env}_seed{seed}"


def train_run_dir(root: Path, env: int, seed: int) -> Path:
    return root / "train" / f"env{env}" / run_name(env, seed)


def build_train_jobs(root: Path, envs: List[dict], seeds: List[int], total_timesteps: int,
                     checkpoint_interval: int) -> List[dict]:
    base_args = _load_yaml(TRAIN_ARGS_BASE)
    jobs = []
    for env in envs:
        e = env["env"]
        args = dict(base_args)
        args.update(config=env["sim_config"], total_timesteps=int(total_timesteps),
                    checkpoint_interval=int(checkpoint_interval), run_name=f"sim2sim_hist4_env{e}")
        args_path = root / "configs" / f"env{e}_train.yaml"
        _write_yaml(args_path, f"TD3 args of the sim2sim env{e} baseline = {TRAIN_ARGS_BASE} with this env's sim config, "
                               f"{total_timesteps} steps, checkpoint every {checkpoint_interval}. Seed / run name / "
                               "device / log dir come from the command line. Generated; do not edit.", args)
        for s in seeds:
            run_dir = train_run_dir(root, e, s)
            if (run_dir / "model.pth").exists():
                print(f"[skip ] {run_dir.name}: finished (model.pth exists)")
                continue
            if run_dir.exists():
                moved = run_dir.with_name(f"{run_dir.name}_incomplete_{time.strftime('%Y%m%d-%H%M%S')}")
                run_dir.rename(moved)
                print(f"[move ] unfinished {run_dir.name} -> {moved.name} (restarting it)")
            jobs.append({
                "name": run_dir.name,
                "log": str(run_dir) + ".stdout.log",
                "cmd": [sys.executable, "-u", "-m", "scripts.td3.td3_training", "--args-file", str(args_path),
                        "--config", env["sim_config"], "--seed", str(s), "--run-name", run_dir.name,
                        "--device", "cuda:{gpu}", "--log-parent-dir", str(run_dir)],
            })
    return jobs


# ---------------------------------------------------------------------------
# Paired deterministic evaluation
# ---------------------------------------------------------------------------


def _make_env(sim_config: str, seed: int):
    from airhockey import AirHockeyEnv

    cfg = copy.deepcopy(_load_yaml(sim_config)["air_hockey"])
    cfg["seed"] = int(seed)
    return AirHockeyEnv(cfg)


def _juggles_from_obs(obs_seq: List[np.ndarray]) -> int:
    """Juggle count of one episode from the newest paddle / puck frames of the
    history obs (paddle [12:14], puck [27:29], puck occluded flag [29]); the
    same counter the real-robot pipeline uses."""
    from scripts.td3.helper.juggle_counter import count_juggles_from_arrays

    obs = np.asarray(obs_seq)
    return int(count_juggles_from_arrays(obs[:, 12:14], obs[:, 27:29], obs[:, 29]).n_juggles)


def load_actor(path: str, arch: dict):
    """Deterministic actor from a model.pth / training_state.pth (CPU, eval mode)."""
    import torch

    from scripts.td3.deterministic_agent import DeterministicAgent
    from scripts.td3.eval_utils import unwrap_eval_state_dict
    from scripts.td3.helper.real_td3_runtime import build_policy_env_view

    state = unwrap_eval_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    # Input width of the first linear layer = policy obs dim (30 obs + 2 last action).
    obs_dim = int(next(v for k, v in state.items() if k.startswith("actor.") and v.ndim == 2).shape[1])
    actor = DeterministicAgent(build_policy_env_view(obs_dim, 2), action_scale=1.0, action_bias=0.0,
                               hidden_layer_size=int(arch["agent_hidden_layer_size"]),
                               num_hidden_layers=int(arch["agent_num_hidden_layers"]))
    actor.load_state_dict(state)
    return actor.eval()


def evaluate_actor(actor, sim_config: str, env_index: int, n_episodes: int, use_last_action: bool = True) -> dict:
    """Deterministic rollouts on fixed episode seeds (identical for every policy in an env)."""
    import torch

    env = _make_env(sim_config, seed=EVAL_SEED_BASE + 1000 * env_index)
    returns, lengths, juggles, contacts = [], [], [], []
    try:
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=EVAL_SEED_BASE + 1000 * env_index + ep)
            last = np.zeros(2, np.float32)
            ret, n, hits, obs_seq = 0.0, 0, 0, [obs]
            while True:
                x = np.concatenate([obs, last]) if use_last_action else obs
                with torch.no_grad():
                    action = actor.get_action(torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)).numpy()[0]
                obs, reward, term, trunc, info = env.step(action)
                ret += float(reward); n += 1; hits += int(info.get("paddle_puck_collision_count", 0))
                obs_seq.append(obs); last = action.astype(np.float32)
                if term or trunc:
                    break
            returns.append(ret); lengths.append(n); contacts.append(hits); juggles.append(_juggles_from_obs(obs_seq))
    finally:
        env.close()
    return {"n_episodes": n_episodes, "return_mean": float(np.mean(returns)),
            "return_sem": float(np.std(returns) / np.sqrt(len(returns))), "length_mean": float(np.mean(lengths)),
            "juggles_mean": float(np.mean(juggles)), "contacts_mean": float(np.mean(contacts)),
            "returns": [float(r) for r in returns]}


# ---------------------------------------------------------------------------
# 2 + 3. Zero-shot eval and online fine-tuning (one worker per env × source seed)
# ---------------------------------------------------------------------------


def online_root(root: Path, policy_lr: Optional[float] = None) -> Path:
    """<root>/online for the args-file actor LR, <root>/online_policy_lr<lr> for an override."""
    return root / ("online" if policy_lr is None else f"online_policy_lr{policy_lr:g}")


def online_dir(root: Path, env: int, seed: int, policy_lr: Optional[float] = None) -> Path:
    return online_root(root, policy_lr) / f"env{env}" / f"hist4_seed{seed}"


def recycle_root(root: Path, alpha_start: float) -> Path:
    """<root>/online_recycle_alpha<α0>: the data-recycling experiment (Fig. 6 of As et al.)."""
    return root / f"online_recycle_alpha{alpha_start:g}"


def recycle_dir(root: Path, alpha_start: float, trial: int, env: int, seed: int) -> Path:
    return recycle_root(root, alpha_start) / f"trial{trial}" / f"env{env}" / f"hist4_seed{seed}"


RECYCLE_BUFFER_FILE = "online_buffer.pt"
_BUFFER_KEYS = ("observations", "next_observations", "actions", "prev_actions", "rewards", "dones")


def build_recycle_jobs(root: Path, envs: List[dict], seeds: List[int], cli) -> List[dict]:
    """One job per (env, seed); each runs the ``--recycle-trials`` trials in sequence
    (trial t needs the online buffers of trials < t)."""
    jobs = []
    for env, s in itertools.product(envs, seeds):
        last = recycle_dir(root, cli.recycle_alpha_start, cli.recycle_trials - 1, env["env"], s)
        if (last / RESULT_FILE).exists() and not cli.overwrite:
            print(f"[skip ] env{env['env']} seed{s}: all {cli.recycle_trials} trials done (--overwrite to redo)")
            continue
        cmd = [sys.executable, "-u", "-m", "scripts.td3.sim2sim_online_finetune", "--worker", "--recycle",
               "--root", str(root), "--env", str(env["env"]), "--seed", str(s), "--learner-device", "cuda:{gpu}",
               "--num-online-episodes", str(cli.num_online_episodes), "--eval-episodes", str(cli.eval_episodes),
               "--online-eval-every", str(cli.online_eval_every), "--online-eval-episodes", str(cli.online_eval_episodes),
               "--min-episode-steps", str(cli.min_episode_steps), "--source-checkpoint", cli.source_checkpoint,
               "--warm-start-episodes", str(cli.warm_start_episodes),
               "--recycle-trials", str(cli.recycle_trials), "--recycle-alpha-start", repr(cli.recycle_alpha_start),
               "--recycle-anneal-episodes", str(cli.recycle_anneal_episodes)]
        if cli.policy_lr is not None:
            cmd += ["--policy-lr", repr(cli.policy_lr)]
        if cli.overwrite:
            cmd.append("--overwrite")
        out = recycle_root(root, cli.recycle_alpha_start) / f"env{env['env']}_seed{s}.log"
        jobs.append({"name": f"recycle_env{env['env']}_seed{s}", "cmd": cmd, "log": str(out)})
    return jobs


def recycle_alpha(rounds_trained: int, alpha_start: float, anneal_episodes: int) -> float:
    """α of the training round after ``rounds_trained`` earlier rounds: linear from
    ``alpha_start`` (first round) to 1 (round ``anneal_episodes``, and after)."""
    if anneal_episodes <= 1:
        return 1.0
    return float(alpha_start + (1.0 - alpha_start) * min(1.0, rounds_trained / (anneal_episodes - 1)))


class _MixedReplaySnapshot:
    """(1 − α) Unif(D0) + α Unif(D_online), per sample (As et al. Eq. 16).

    D0 = the online buffers of this (env, seed)'s earlier trials, on the learner
    device once per trial; the online part is copied each round like
    ``_DeviceReplaySnapshot`` (the learner's default sampler)."""

    def __init__(self, online, d0: Dict[str, "torch.Tensor"], alpha: float):
        import torch

        self.online, self.d0, self.alpha = online, d0, float(alpha)
        self.d0_size = int(d0["rewards"].shape[0])
        self.device = online.device
        self._torch = torch

    def sample(self, batch_size: int) -> Dict[str, "torch.Tensor"]:
        torch = self._torch
        n = int(batch_size)
        use_online = torch.rand(n, device=self.device) < self.alpha
        on_idx = torch.randint(0, self.online.size, (n,), device=self.device)
        d0_idx = torch.randint(0, self.d0_size, (n,), device=self.device)
        out = {}
        for key, value in self.online.tensors.items():
            mask = use_online.view(-1, *([1] * (value.dim() - 1)))
            out[key] = torch.where(mask, value[on_idx], self.d0[key][d0_idx])
        return out


def _load_d0(root: Path, cli, device: str) -> tuple:
    """Concatenate the saved online buffers of trials 0 .. cli.trial − 1 (same env, seed)."""
    import torch

    parts, sources = [], []
    for t in range(cli.trial):
        path = recycle_dir(root, cli.recycle_alpha_start, t, cli.env, cli.seed) / RECYCLE_BUFFER_FILE
        if not path.exists():
            raise SystemExit(f"trial {cli.trial} needs the online buffer of trial {t}: {path} is missing")
        parts.append(torch.load(path, map_location="cpu", weights_only=False))
        sources.append(str(path))
    if not parts:
        return None, sources
    return {k: torch.cat([p[k] for p in parts]).to(device) for k in _BUFFER_KEYS}, sources


def build_finetune_jobs(root: Path, envs: List[dict], seeds: List[int], cli) -> List[dict]:
    jobs = []
    for lr, env, s in itertools.product(cli.policy_lrs or [None], envs, seeds):
        out = online_dir(root, env["env"], s, lr)
        tag = "" if lr is None else f" policy_lr {lr:g}"
        if (out / RESULT_FILE).exists() and not cli.overwrite:
            print(f"[skip ] env{env['env']} seed{s}{tag}: {out / RESULT_FILE} exists (--overwrite to redo)")
            continue
        cmd = [sys.executable, "-u", "-m", "scripts.td3.sim2sim_online_finetune", "--worker",
               "--root", str(root), "--env", str(env["env"]), "--seed", str(s), "--learner-device", "cuda:{gpu}",
               "--num-online-episodes", str(cli.num_online_episodes), "--eval-episodes", str(cli.eval_episodes),
               "--online-eval-every", str(cli.online_eval_every), "--online-eval-episodes", str(cli.online_eval_episodes),
               "--min-episode-steps", str(cli.min_episode_steps), "--source-checkpoint", cli.source_checkpoint]
        cmd += ["--warm-start-episodes", str(cli.warm_start_episodes)]
        if lr is not None:
            cmd += ["--policy-lr", repr(lr)]
        if cli.overwrite:
            cmd.append("--overwrite")
        name = f"finetune_env{env['env']}_seed{s}" + ("" if lr is None else f"_lr{lr:g}")
        jobs.append({"name": name, "cmd": cmd, "log": str(out) + ".log"})
    return jobs


def _save_finetuned(out: Path, state, source_checkpoint: dict, episodes_trained: int) -> None:
    """model.pth + a training_state.pth with the keys td3_online_real_finetune's
    --checkpoint expects (actor, actor_target, critics, critic targets, both Adam states, args)."""
    import torch

    payload = {"actor": state.actor.state_dict(), "actor_target": state.actor_target.state_dict(),
               "q_optimizer": state.q_optimizer.state_dict(), "actor_optimizer": state.actor_optimizer.state_dict(),
               "args": source_checkpoint.get("args"), "online_episodes_trained": int(episodes_trained)}
    for i, (q, qt) in enumerate(zip(state.qfs, state.qfs_target), start=1):
        payload[f"qf{i}"] = q.state_dict()
        payload[f"qf{i}_target"] = qt.state_dict()
    out.mkdir(parents=True, exist_ok=True)
    torch.save(state.actor.state_dict(), out / "model.pth")
    torch.save(payload, out / "training_state.pth")


def run_worker(cli) -> None:
    """Zero-shot eval + online fine-tuning of one source seed in one target env."""
    import torch

    from scripts.td3 import td3_online_real_finetune as ft
    from scripts.td3.helper.shared_replay import SharedTD3Replay
    from scripts.td3.helper.td3_episode_collection import EpisodeTrajectory

    torch.set_num_threads(1)
    root = _repo_path(cli.root)
    manifest = json.load(open(root / "envs.json"))
    env_spec = next(e for e in manifest["envs"] if e["env"] == cli.env)
    trial = getattr(cli, "trial", None)
    out = (online_dir(root, cli.env, cli.seed, cli.policy_lr) if trial is None
           else recycle_dir(root, cli.recycle_alpha_start, trial, cli.env, cli.seed))
    if (out / ft._PROGRESS_FILE).exists():
        if not cli.overwrite and trial is None:  # an unfinished recycling trial is simply restarted
            raise SystemExit(f"{out} already has an online run; pass --overwrite to redo it.")
        moved = out.with_name(f"{out.name}_old_{time.strftime('%Y%m%d-%H%M%S')}")
        out.rename(moved)
        print(f"[worker] moved the previous run to {moved}")
    out.mkdir(parents=True, exist_ok=True)

    # Recipe hyperparameters: exactly the real-robot args file; only the learner
    # device, the episode budget and the warm-start switch come from here. The
    # args file's robot-side fields are replaced by sim ones so nothing can
    # point at the robot: config = this env's Box2D config, outputs under `out`.
    defaults, _, _ = ft._build_finetune_args_file_defaults(str(_repo_path(FINETUNE_ARGS_FILE)))
    args = ft.OnlineFinetuneArgs(**defaults)
    args.checkpoint = str(_repo_path(cli.source_checkpoint.format(seed=cli.seed)))
    args.learner_device = cli.learner_device
    args.num_online_episodes = int(cli.num_online_episodes)
    args.min_episode_steps = int(cli.min_episode_steps)
    args.warm_start_episodes = int(cli.warm_start_episodes)
    if cli.policy_lr is not None:
        args.policy_lr = float(cli.policy_lr)  # actor LR ablation; applied to the loaded Adam state
    args.no_warmup_no_sim_data = args.warm_start_episodes == 0
    args.config = env_spec["sim_config"]
    args.data_root_dir = str(out)
    args.enable_periodic_checkpointing = False      # one final checkpoint below instead
    args.checkpoint_every_online_episodes = 0
    args.checkpoint_root_dir = str(out)
    args.experiment_name = f"sim2sim_env{cli.env}"
    args.seed = int(cli.seed)
    ft._validate_args(args)

    checkpoint = ft._load_checkpoint(args.checkpoint)
    train_args, _ = ft._resolve_train_args(args, checkpoint)
    arch = {"agent_hidden_layer_size": train_args.agent_hidden_layer_size,
            "agent_num_hidden_layers": train_args.agent_num_hidden_layers}
    use_last = bool(train_args.use_last_action_in_policy_state)
    src_hist = ft._hist_len_of((checkpoint.get("args") or {}).get("config"))
    env_hist = _load_yaml(env_spec["sim_config"])["air_hockey"]["simulator_params"]["hist_len"]
    if src_hist is not None and int(src_hist) != int(env_hist):
        raise SystemExit(f"hist_len mismatch: source policy {src_hist}, target env {env_hist}")

    sim_config = env_spec["sim_config"]
    print(f"[worker] env{cli.env} {env_spec['params']}\n[worker] source {args.checkpoint}", flush=True)

    # 2. zero-shot
    t0 = time.time()
    zero_shot = evaluate_actor(load_actor(args.checkpoint, arch), sim_config, cli.env, cli.eval_episodes, use_last)
    print(f"[worker] zero-shot return {zero_shot['return_mean']:.1f} ± {zero_shot['return_sem']:.1f} "
          f"({time.time() - t0:.0f}s)", flush=True)

    # 3. online fine-tuning: real-script learner, Box2D collector.
    env = _make_env(sim_config, seed=10_000 * cli.env + cli.seed)
    if env.simulator_name != "box2d":
        raise SystemExit(f"sim2sim worker expects the Box2D simulator, got {env.simulator_name!r} from {sim_config}")
    obs_dim = int(np.prod(env.observation_space.shape)); act_dim = int(np.prod(env.action_space.shape))
    action_low = np.asarray(env.action_space.low, np.float32); action_high = np.asarray(env.action_space.high, np.float32)
    replay = SharedTD3Replay(success_capacity=int(args.online_buffer_size), failure_capacity=1,
                             obs_shape=(obs_dim,), action_shape=(act_dim,))
    state = ft._build_learner_from_checkpoint(args=args, train_args=train_args, checkpoint=checkpoint, obs_dim=obs_dim,
                                              act_dim=act_dim, action_low_np=action_low, action_high_np=action_high,
                                              tb_log_dir=str(out / "learner_tb"))
    progress = ft._OnlineProgress(seed_dir=out, experiment=args.experiment_name, hist_len=int(env_hist), sim_seed=cli.seed)
    stats: Dict[str, object] = {}
    # pending_episode hands each stored episode to the learner step, which logs the
    # critic error ε = Q_φ − Q_MC on it before training (online_finetune_critic_error/*).
    pending_episode: Dict[str, object] = {}
    push = ft._make_online_replay_push(stats, args, pending_episode)
    make_buffer, d0_sources, d0_size = None, [], 0
    if trial is not None:
        d0, d0_sources = _load_d0(root, cli, args.learner_device)
        d0_size = 0 if d0 is None else int(d0["rewards"].shape[0])
        print(f"[worker] recycling trial {trial}: D0 = {d0_size} transitions from {len(d0_sources)} earlier trial(s); "
              f"α {cli.recycle_alpha_start:g} -> 1 over {cli.recycle_anneal_episodes} rounds", flush=True)
        if d0 is not None:
            def make_buffer(rb, device):
                rounds = progress.episodes_trained
                alpha = recycle_alpha(rounds, cli.recycle_alpha_start, cli.recycle_anneal_episodes)
                progress.writer.add_scalar("online_finetune_recycle/alpha", alpha, rounds + 1)
                progress.writer.add_scalar("online_finetune_recycle/d0_size", float(d0_size), rounds + 1)
                return _MixedReplaySnapshot(ft._DeviceReplaySnapshot(rb, device), d0, alpha)
    learner_step = ft._make_online_learner_step(progress, pending_episode, make_buffer=make_buffer)
    recent_returns: deque = deque(maxlen=500)

    collector = load_actor(args.checkpoint, arch)  # CPU copy; synced after every training round
    rng = torch.Generator().manual_seed(1_000_003 * cli.env + cli.seed)
    online_evals = []

    def _online_eval(rounds: int) -> None:
        if cli.online_eval_every <= 0:
            return
        r = evaluate_actor(collector, sim_config, cli.env, cli.online_eval_episodes, use_last)
        progress.writer.add_scalar("online_finetune_eval/return", r["return_mean"], rounds)
        progress.writer.add_scalar("online_finetune_eval/juggles", r["juggles_mean"], rounds)
        progress.writer.add_scalar("online_finetune_eval/length", r["length_mean"], rounds)
        progress.writer.flush()
        online_evals.append({"rounds_trained": rounds, **{k: r[k] for k in ("return_mean", "return_sem", "juggles_mean", "length_mean")}})

    _online_eval(0)
    episode_id = 0
    attempts = 0
    t_start = time.time()
    while progress.launch_kept_episodes < args.num_online_episodes:
        attempts += 1
        obs, _ = env.reset(seed=500_000 + 10_000 * cli.env + 100 * cli.seed + attempts)
        last = torch.zeros(act_dim)
        traj = EpisodeTrajectory.empty()
        obs_seq, hits = [obs], 0
        while True:
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            x = torch.cat([obs_t, last]) if use_last else obs_t
            with torch.no_grad():
                action = collector.get_action(x.unsqueeze(0))[0]
                if args.exploration_noise > 0:
                    action = action + torch.randn(action.shape, generator=rng) * float(args.exploration_noise)
                action = torch.clamp(action, torch.as_tensor(action_low), torch.as_tensor(action_high))
            next_obs, reward, term, trunc, info = env.step(action.numpy())
            hits += int(info.get("paddle_puck_collision_count", 0))
            # Same transition semantics as td3_training: done = termination only
            # (time-limit truncation still bootstraps); prev_action = the last
            # action the policy saw (zeros at the episode start).
            traj.append_step(obs=obs_t, next_obs=torch.as_tensor(next_obs, dtype=torch.float32), action=action,
                             reward=torch.tensor(float(reward)), done=torch.tensor(float(term)), prev_action=last)
            obs_seq.append(next_obs)
            obs, last = next_obs, action.clone()
            if term or trunc:
                break
        if len(traj.observations) < int(args.min_episode_steps):
            print(f"[online] discarded attempt ({len(traj.observations)} < {args.min_episode_steps} steps)", flush=True)
            continue
        episode_id += 1
        stats.update(last_episode_id=float(episode_id), last_episode_juggles=float(_juggles_from_obs(obs_seq)),
                     last_episode_contacts=float(hits), last_episode_estop_flag=0.0)
        push(replay, traj, recent_returns, 0.0)
        actor_changed = learner_step(args, train_args, replay, stats, state)
        report = stats.pop("online_episode_report", {}) or {}
        if actor_changed:
            collector.load_state_dict({k: v.detach().cpu() for k, v in state.actor.state_dict().items()})
        kind = report.get("kind", "?")
        detail = (f"round {report['round']} K={report['K']} q_loss {report['q_loss']:.4f} actor_loss {report['actor_loss']:.4f}"
                  if kind == "trained" else f"{kind}")
        print(f"[online] ep {progress.launch_kept_episodes}/{args.num_online_episodes} return {traj.episode_return:6.1f} "
              f"len {len(traj.observations):3d} | {detail} | buffer {replay.len(ft._ONLINE_PARTITION)}", flush=True)
        if kind == "trained" and cli.online_eval_every > 0 and progress.episodes_trained % cli.online_eval_every == 0:
            _online_eval(progress.episodes_trained)
    if not online_evals or online_evals[-1]["rounds_trained"] != progress.episodes_trained:
        _online_eval(progress.episodes_trained)
    online_wall = time.time() - t_start
    env.close()
    progress.close()
    state.writer.close()

    _save_finetuned(out / "final", state, checkpoint, progress.episodes_trained)
    if trial is not None:  # this trial's D_online becomes part of D0 for the later trials
        partition = replay.state_dict()[ft._ONLINE_PARTITION]
        torch.save({k: partition[k] for k in _BUFFER_KEYS}, out / RECYCLE_BUFFER_FILE)
    finetuned = evaluate_actor(collector, sim_config, cli.env, cli.eval_episodes, use_last)
    rows = [json.loads(line) for line in open(out / ft._PROGRESS_FILE)]
    trained_returns = [r["episode_return"] for r in rows if r.get("trained") or r.get("warm_start") is False]
    result = {
        "env": cli.env, "source_seed": cli.seed, "source_checkpoint": args.checkpoint, "env_params": env_spec["params"],
        "recipe": {"args_file": FINETUNE_ARGS_FILE, "warm_start_episodes": args.warm_start_episodes,
                   "utd_ratio": args.utd_ratio, "actor_update_every": args.actor_update_every,
                   "policy_lr": args.policy_lr, "q_lr": args.q_lr, "target_network_frequency": args.target_network_frequency,
                   "exploration_noise": args.exploration_noise, "num_online_episodes": args.num_online_episodes,
                   "min_episode_steps": args.min_episode_steps},
        "recycle": None if trial is None else {
            "trial": trial, "alpha_start": cli.recycle_alpha_start, "anneal_episodes": cli.recycle_anneal_episodes,
            "d0_transitions": d0_size, "d0_sources": d0_sources,
            "online_transitions": replay.len(ft._ONLINE_PARTITION)},
        "zero_shot": zero_shot, "finetuned": finetuned,
        "online": {"episodes_kept": progress.launch_kept_episodes, "attempts": attempts,
                   "rounds_trained": progress.episodes_trained, "critic_updates": int(state.total_updates),
                   "actor_updates": int(state.total_actor_updates), "wall_s": online_wall,
                   "last10_return_mean": float(np.mean(trained_returns[-10:])) if trained_returns else math.nan,
                   "periodic_eval": online_evals},
    }
    with open(out / RESULT_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[worker] done: zero-shot {zero_shot['return_mean']:.1f} -> fine-tuned {finetuned['return_mean']:.1f} "
          f"after {progress.episodes_trained} rounds ({online_wall:.0f}s online)", flush=True)


def run_recycle_worker(cli) -> None:
    """Trials 0 .. N−1 of one (env, seed) in sequence. Every trial starts from the same
    source policy with an empty D_online; trial t samples (1 − α) Unif(D0) + α Unif(D_online)
    with D0 = the online buffers of trials < t and α annealed ``--recycle-alpha-start`` -> 1.
    Finished trials (``result.json``) are skipped unless ``--overwrite``."""
    root = _repo_path(cli.root)
    for t in range(cli.recycle_trials):
        out = recycle_dir(root, cli.recycle_alpha_start, t, cli.env, cli.seed)
        if (out / RESULT_FILE).exists() and not cli.overwrite:
            print(f"[recycle] trial {t}: {out / RESULT_FILE} exists, skipping", flush=True)
            continue
        print(f"\n[recycle] ===== env{cli.env} seed{cli.seed} trial {t}/{cli.recycle_trials - 1} =====", flush=True)
        trial_cli = copy.copy(cli)
        trial_cli.trial = t
        run_worker(trial_cli)


def show_recycle_results(root: Path, envs: List[dict], seeds: List[int], cli) -> None:
    """Per-trial table + ``figure6_env<E>.png`` (periodic deterministic eval vs training rounds, mean ± SE over seeds)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rroot = recycle_root(root, cli.recycle_alpha_start)
    for env in envs:
        e = env["env"]
        rows, curves = [], {}
        for t in range(cli.recycle_trials):
            res = [json.load(open(f)) for s in seeds
                   if (f := recycle_dir(root, cli.recycle_alpha_start, t, e, s) / RESULT_FILE).exists()]
            if not res:
                rows.append([f"trial{t}", "-", "-", "-", "-", "-", f"0/{len(seeds)}"])
                continue
            rows.append([f"trial{t}", f"{np.mean([r['recycle']['d0_transitions'] for r in res]):.0f}",
                         _mean_std([r["zero_shot"]["return_mean"] for r in res]),
                         _mean_std([r["finetuned"]["return_mean"] for r in res]),
                         _mean_std([r["finetuned"]["return_mean"] - r["zero_shot"]["return_mean"] for r in res]),
                         _mean_std([r["online"]["last10_return_mean"] for r in res]), f"{len(res)}/{len(seeds)}"])
            by_round: Dict[int, list] = {}
            for r in res:
                for ev in r["online"]["periodic_eval"]:
                    by_round.setdefault(ev["rounds_trained"], []).append(ev["return_mean"])
            xs = sorted(k for k, v in by_round.items() if len(v) == len(res))
            curves[t] = (np.array(xs), np.array([np.mean(by_round[x]) for x in xs]),
                         np.array([np.std(by_round[x], ddof=1) / np.sqrt(len(res)) if len(res) > 1 else 0.0
                                   for x in xs]), len(res))
        _print_table(f"Data recycling, env{e} (α {cli.recycle_alpha_start:g} -> 1 over {cli.recycle_anneal_episodes} rounds, "
                     f"{rroot.name}/): mean ± std over seeds ({cli.eval_episodes} deterministic eval episodes per policy)",
                     ["trial", "D0 transitions", "zero-shot", "fine-tuned", "Δ ft − zs", "online last-10 ret",
                      "seeds done"], rows)
        if not curves:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for t, (x, m, se, n) in curves.items():
            line, = ax.plot(x, m, label=f"trial {t} (D0 = trials < {t})" if t else "trial 0 (online only)")
            ax.fill_between(x, m - se, m + se, color=line.get_color(), alpha=0.2)
        ax.set_xlabel("online episodes trained")
        ax.set_ylabel(f"eval return ({cli.online_eval_episodes} det. episodes)")
        ax.set_title(f"Data recycling, env{e}: mean ± SE over seeds")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = rroot / f"figure6_env{e}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"figure: {path}")
    trials = " ".join(str(rroot / f"trial{t}") for t in range(cli.recycle_trials))
    print(f"\nTensorBoard (seeds -> mean ± SE, Custom Scalars tab):\n  .venv/bin/python scripts/analysis/aggregate_online_tb.py "
          f"{trials}\n  .venv/bin/tensorboard --logdir {rroot / 'online_agg'}")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def _print_table(title: str, header: List[str], rows: List[List[str]]) -> None:
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = "+".join("-" * (w + 2) for w in widths)
    print(f"\n{title}")
    print(" | ".join(str(h).ljust(w) for h, w in zip(header, widths)))
    print(line)
    for r in rows:
        print(" | ".join(str(c).rjust(w) if i else str(c).ljust(w) for i, (c, w) in enumerate(zip(r, widths))))


def _peak_train_return(run_dir: Path, window: int = 50) -> float:
    """Best rolling mean (``window`` episodes) of the training return, from TensorBoard."""
    from tensorboard.backend.event_processing import event_accumulator

    ea = event_accumulator.EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    if "charts/episodic_return" not in ea.Tags()["scalars"]:
        return math.nan
    values = np.array([e.value for e in ea.Scalars("charts/episodic_return")])
    if len(values) < window:
        return float(values.mean()) if len(values) else math.nan
    return float(np.convolve(values, np.ones(window) / window, mode="valid").max())


def _baseline_eval_task(task) -> tuple:
    run_dir, sim_config, env_index, n_eps = task
    import torch

    torch.set_num_threads(1)
    run_dir = Path(run_dir)
    arch = _load_yaml(run_dir / "args.yaml")
    result = evaluate_actor(load_actor(str(run_dir / "model.pth"), arch), sim_config, env_index, n_eps,
                            bool(arch.get("use_last_action_in_policy_state", True)))
    result["peak_train_return_rolling50"] = _peak_train_return(run_dir)
    result["eval_episode_seed_base"] = EVAL_SEED_BASE + 1000 * env_index
    with open(run_dir / BASELINE_EVAL_FILE, "w") as f:
        json.dump(result, f, indent=2)
    return str(run_dir), result


def _source_nominal_task(task) -> tuple:
    seed, source_checkpoint, source_sim_config, n_eps = task
    import torch

    torch.set_num_threads(1)
    path = str(_repo_path(source_checkpoint.format(seed=seed)))
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return seed, evaluate_actor(load_actor(path, ckpt["args"]), source_sim_config, 0, n_eps, True)


def _mean_std(values: List[float]) -> str:
    values = [v for v in values if v is not None and not math.isnan(v)]
    if not values:
        return "-"
    return f"{np.mean(values):.1f} ± {np.std(values):.1f}" if len(values) > 1 else f"{values[0]:.1f}"


def show_results(root: Path, seeds: List[int], cli) -> None:
    from multiprocessing import Pool

    manifest = json.load(open(root / "envs.json"))
    envs = manifest["envs"]
    print_envs(envs, manifest["ranges"], manifest["means"])

    # Baselines: evaluate finished runs that have no (current) eval yet.
    tasks = []
    for env in envs:
        for s in seeds:
            run_dir = train_run_dir(root, env["env"], s)
            ev = run_dir / BASELINE_EVAL_FILE
            fresh = ev.exists() and json.load(open(ev)).get("n_episodes") == cli.eval_episodes
            if (run_dir / "model.pth").exists() and not fresh:
                tasks.append((str(run_dir), env["sim_config"], env["env"], cli.eval_episodes))
    source_cache = root / "source_nominal_eval.json"
    source_nominal = json.load(open(source_cache)) if source_cache.exists() else {}
    if source_nominal.get("n_episodes") != cli.eval_episodes:
        source_nominal = {}
    source_sim_config = manifest.get("source_sim_config", SOURCE_SIM_CONFIG)
    nominal_tasks = [] if source_nominal or cli.source_checkpoint is None else [
        (s, cli.source_checkpoint, source_sim_config, cli.eval_episodes) for s in seeds]
    if tasks or nominal_tasks:
        print(f"\nEvaluating {len(tasks)} finished baseline run(s) and {len(nominal_tasks)} source policies on the "
              f"source env ({cli.eval_episodes} episodes each)...", flush=True)
        with Pool(processes=min(len(tasks) + len(nominal_tasks), max(1, (os.cpu_count() or 2) // 2))) as pool:
            results = pool.map(_baseline_eval_task, tasks) if tasks else []
            nominal = pool.map(_source_nominal_task, nominal_tasks) if nominal_tasks else []
        if nominal:
            source_nominal = {"n_episodes": cli.eval_episodes, "per_seed": {str(s): r for s, r in nominal}}
            with open(source_cache, "w") as f:
                json.dump(source_nominal, f, indent=2)
        del results

    for lr in cli.policy_lrs or [None]:
        _results_table(root, envs, seeds, cli, source_nominal, lr)
    print(f"\nTensorBoard: tensorboard --logdir {root}  (train/ = in-env baselines, online*/ = fine-tuning curves)")


def _results_table(root: Path, envs: List[dict], seeds: List[int], cli, source_nominal: dict,
                   policy_lr: Optional[float]) -> None:
    header = ["env", "trained in env (final)", "trained: peak train ret", "zero-shot", "fine-tuned",
              "Δ ft − zs", "online last-10 ret", "trained / zs / ft done"]
    rows, detail_rows = [], []
    if source_nominal:
        per = source_nominal["per_seed"]
        rows.append(["source env", _mean_std([per[k]["return_mean"] for k in per]), "-", "(same policies)", "-", "-", "-",
                     f"-/{len(per)}/-"])
    for env in envs:
        e = env["env"]
        base, peak, zs, fte, last10, delta = [], [], [], [], [], []
        for s in seeds:
            ev = train_run_dir(root, e, s) / BASELINE_EVAL_FILE
            b = json.load(open(ev)) if ev.exists() else None
            rf = online_dir(root, e, s, policy_lr) / RESULT_FILE
            r = json.load(open(rf)) if rf.exists() else None
            if b:
                base.append(b["return_mean"]); peak.append(b.get("peak_train_return_rolling50", math.nan))
            if r:
                zs.append(r["zero_shot"]["return_mean"]); fte.append(r["finetuned"]["return_mean"])
                last10.append(r["online"]["last10_return_mean"])
                delta.append(r["finetuned"]["return_mean"] - r["zero_shot"]["return_mean"])
            detail_rows.append([f"env{e}", f"seed{s}",
                                f"{b['return_mean']:.1f}" if b else "-",
                                f"{b['juggles_mean']:.2f}" if b else "-",
                                f"{r['zero_shot']['return_mean']:.1f}" if r else "-",
                                f"{r['zero_shot']['juggles_mean']:.2f}" if r else "-",
                                f"{r['finetuned']['return_mean']:.1f}" if r else "-",
                                f"{r['finetuned']['juggles_mean']:.2f}" if r else "-",
                                f"{r['online']['last10_return_mean']:.1f}" if r else "-"])
        rows.append([f"env{e}", _mean_std(base), _mean_std(peak), _mean_std(zs), _mean_std(fte), _mean_std(delta),
                     _mean_std(last10), f"{len(base)}/{len(zs)}/{len(fte)}"])
    lr_tag = "args-file actor LR" if policy_lr is None else f"actor LR {policy_lr:g}"
    _print_table(f"Juggle return per target env [{lr_tag}, {online_root(root, policy_lr).name}/]: mean ± std over seeds "
                 f"({cli.eval_episodes} deterministic eval episodes per policy, same episode seeds in every column)",
                 header, rows)
    print("  trained in env = new hist4 policy trained from scratch in that env (final model); peak train ret = best "
          "50-episode rolling training return (with exploration noise)\n  zero-shot = source hist4 policies (trained on "
          "the sysid-v2 env); fine-tuned = the same policies after the online recipe\n  online last-10 ret = mean "
          "return of the last 10 online episodes (with exploration noise); source env row = source policies on their "
          "own env")
    if cli.per_seed:
        _print_table("Per seed (return / juggles per episode)",
                     ["env", "seed", "trained", "jug", "zero-shot", "jug", "fine-tuned", "jug", "online last-10"],
                     detail_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true", help="1. train hist4 policies from scratch in every target env")
    mode.add_argument("--zero-shot-and-online-fine-tune", action="store_true",
                      help="2 + 3. zero-shot eval and online fine-tuning of the source policies, then the tables")
    mode.add_argument("--results", action="store_true", help="print the tables (evaluates finished baselines first)")
    mode.add_argument("--recycle-data", action="store_true",
                      help="data-recycling experiment (As et al. Fig. 6): per (env, seed), --recycle-trials fine-tuning "
                           "trials from the same source policy, trial t mixing the online buffers of trials < t (D0) "
                           "with its own (α annealed to 1), in <root>/online_recycle_alpha<α0>/trial<t>/; then the tables")
    mode.add_argument("--recycle-results", action="store_true", help="tables + figure of the data-recycling experiment")
    mode.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--root", default=DEFAULT_ROOT, help=f"experiment folder (default {DEFAULT_ROOT})")
    ap.add_argument("--n-envs", type=int, default=5)
    ap.add_argument("--env-seed", type=int, default=0, help="seed of the env draw (fixed per experiment)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="training seeds of the baselines = source policy seeds")
    ap.add_argument("--source-checkpoint", default=None,
                    help="source training_state.pth, {seed} filled in (default: the PID-trained sysid-v2 hist4 policies "
                         f"{SOURCE_CHECKPOINT}; required with --direct-force, whose envs those policies were not trained for)")
    ap.add_argument("--scale-box", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                    help="draw each of paddle_density / puck_damping / gravity uniformly in [LO, HI] x its sysid-v2 value "
                         "instead of the ±25 %% box (identify_ood_gap step k = [1.25 + 0.125k, 1.75 + 0.125k], so step 6 = "
                         "2.0 2.5). Recorded in <root>/envs.json; later calls on the same --root reuse it.")
    ap.add_argument("--ranges-from", default=None, metavar="SIM_CONFIG",
                    help="draw the target envs uniformly in this sim config's air_hockey.random_variable_ranges "
                         "(e.g. the DR oracle's training box). Recorded in <root>/envs.json like --scale-box.")
    ap.add_argument("--direct-force", action="store_true", default=None,
                    help="direct force control in every sim config (action = paddle force, no filter / PID; "
                         "the identify_ood_gap oracle setup). Recorded in <root>/envs.json like --scale-box.")
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--jobs-per-gpu", type=int, default=None,
                    help="concurrent jobs per GPU (default 13 = all 25 at once on 2 GPUs; one Quadro RTX 6000 saturates at ~1000 training steps/s in total from ~5 jobs up, so more jobs per GPU does not finish sooner)")
    ap.add_argument("--total-timesteps", type=int, default=2_000_000)
    ap.add_argument("--checkpoint-interval", type=int, default=0,
                    help="baseline intermediate checkpoint spacing (each stores a ~35 MB eval GIF); 0 = only the "
                         "final checkpoint_<total_timesteps>/")
    ap.add_argument("--num-online-episodes", type=int, default=100,
                    help="online episodes per fine-tuning run (warm-start episodes, if any, included, as in the real script)")
    ap.add_argument("--warm-start-episodes", type=int, default=0,
                    help="episodes of the unchanged source policy stored before the first update; 0 (default) = "
                         "no warm start, learning starts after the first episode from an empty buffer")
    ap.add_argument("--min-episode-steps", type=int, default=1,
                    help="drop shorter episodes (the real args file uses 20 to filter bad robot episodes; "
                         "every sim episode is valid, so the default keeps all)")
    ap.add_argument("--eval-episodes", type=int, default=30, help="deterministic eval episodes per policy")
    ap.add_argument("--online-eval-every", type=int, default=10,
                    help="deterministic eval every N training rounds during fine-tuning (0 = off)")
    ap.add_argument("--online-eval-episodes", type=int, default=10)
    ap.add_argument("--policy-lrs", type=float, nargs="+", default=None,
                    help="actor LR ablation: one fine-tuning sweep per value, each in <root>/online_policy_lr<lr>/ "
                         "(default: the args file's policy_lr, in <root>/online/)")
    ap.add_argument("--envs", type=int, nargs="+", default=None,
                    help="only these target envs (1-based) for fine-tuning / recycling (default: all)")
    ap.add_argument("--recycle-trials", type=int, default=4, help="trials per (env, seed) in --recycle-data")
    ap.add_argument("--recycle-alpha-start", type=float, default=0.5,
                    help="α (online share of each batch) at the first training round of trials ≥ 1")
    ap.add_argument("--recycle-anneal-episodes", type=int, default=100,
                    help="training rounds over which α rises linearly to 1 (default = the 100-episode budget)")
    ap.add_argument("--per-seed", action="store_true", help="also print the per-seed table")
    ap.add_argument("--overwrite", action="store_true", help="redo fine-tuning runs that already have results")
    ap.add_argument("--dry-run", action="store_true", help="print envs and commands, run nothing")
    ap.add_argument("--env", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--learner-device", default="cuda:0", help=argparse.SUPPRESS)
    ap.add_argument("--policy-lr", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--recycle", action="store_true", help=argparse.SUPPRESS)  # worker: run the recycling trials
    cli = ap.parse_args()

    if cli.worker:
        run_recycle_worker(cli) if cli.recycle else run_worker(cli)
        return

    root = _repo_path(cli.root)
    envs = sample_envs(root, cli.n_envs, cli.env_seed, cli.scale_box, cli.direct_force, cli.ranges_from)
    manifest = json.load(open(root / "envs.json"))
    if cli.source_checkpoint is None and not manifest.get("direct_force", False):
        cli.source_checkpoint = SOURCE_CHECKPOINT
    print_envs(envs, manifest["ranges"], manifest["means"])
    print(f"\nExperiment folder: {root}")
    run_envs = envs if cli.envs is None else [e for e in envs if e["env"] in cli.envs]

    if cli.recycle_data or cli.recycle_results:
        if cli.recycle_data:
            if cli.source_checkpoint is None:
                raise SystemExit("--source-checkpoint is required for a --direct-force experiment.")
            jobs = build_recycle_jobs(root, run_envs, cli.seeds, cli)
            per_gpu = cli.jobs_per_gpu or 13
            print(f"{len(jobs)} recycling job(s) ({cli.recycle_trials} trials each, sequential), {per_gpu} per GPU on "
                  f"GPUs {cli.gpus}; output {recycle_root(root, cli.recycle_alpha_start)}", flush=True)
            if cli.dry_run:
                for j in jobs:
                    print(" ".join(j["cmd"]))
                return
            run_jobs(jobs, cli.gpus, per_gpu, poll_s=15.0)
        show_recycle_results(root, run_envs, cli.seeds, cli)
        return

    if cli.train:
        jobs = build_train_jobs(root, envs, cli.seeds, cli.total_timesteps, cli.checkpoint_interval)
        per_gpu = cli.jobs_per_gpu or 13
        print(f"{len(jobs)} training run(s), {per_gpu} per GPU on GPUs {cli.gpus}; "
              f"TensorBoard: tensorboard --logdir {root / 'train'}", flush=True)
        if cli.dry_run:
            for j in jobs:
                print(" ".join(j["cmd"]))
            return
        run_jobs(jobs, cli.gpus, per_gpu, poll_s=30.0)
        show_results(root, cli.seeds, cli)
    elif cli.zero_shot_and_online_fine_tune:
        if cli.source_checkpoint is None:
            raise SystemExit("--source-checkpoint is required for a --direct-force experiment (the default source "
                             "policies were trained with PID control).")
        missing = [s for s in cli.seeds if not _repo_path(cli.source_checkpoint.format(seed=s)).exists()]
        if missing:
            raise SystemExit(f"source checkpoints missing for seeds {missing}: {cli.source_checkpoint}")
        jobs = build_finetune_jobs(root, run_envs, cli.seeds, cli)
        per_gpu = cli.jobs_per_gpu or 13
        print(f"{len(jobs)} fine-tuning job(s), {per_gpu} per GPU on GPUs {cli.gpus}; "
              f"TensorBoard: tensorboard --logdir {root}/online*", flush=True)
        if cli.dry_run:
            for j in jobs:
                print(" ".join(j["cmd"]))
            return
        run_jobs(jobs, cli.gpus, per_gpu)
        show_results(root, cli.seeds, cli)
    else:
        show_results(root, cli.seeds, cli)


if __name__ == "__main__":
    main()
