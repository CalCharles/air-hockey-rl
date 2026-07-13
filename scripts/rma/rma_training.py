"""
RMA training for air-hockey (hora-style two-phase protocol).

Phase 1 — PPO jointly trains the privileged object-prop encoder μ and base
policy π (obs ⊕ z → action), with z = tanh(μ(priv_info)).

Phase 2 — freeze μ and π, train the adaptation module φ so
||φ(proprio_hist) − μ(priv_info)||² is minimized while rolling out π(obs ⊕ hat{z}).

Both phases run from a single call to this module (or via rma_training_dr.py).

Launch:
  python -m scripts.rma.rma_training --args-file configs/rma/rma_paramrand_pm25.yaml
  python -m scripts.rma.rma_training_dr --args-file configs/rma/rma_paramrand_pm25.yaml

Cluster submit (patches scripts/rma/vista_template.slurm and sbatches):
  python -m scripts.rma.rma_training_dr \\
    --args-file configs/rma/rma_paramrand_pm25.yaml \\
    --sbatch --sbatch-run-name rma_paramrand_pm25 --sbatch-time 12:00:00
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import torch
import tyro
import yaml

from scripts.rma.env_wrapper import RMAVecEnv
from scripts.rma.misc import set_seed
from scripts.rma.padapt import ProprioAdapt
from scripts.rma.ppo import PPO
from scripts.rma.wandb_utils import wandb_finish


# Match the TD3 trainer's wandb project settings.
WANDB_ENTITY = "rpp689-the-university-of-texas-at-austin"
WANDB_PROJECT = "meta-rl-air-hockey"


@dataclass
class Args:
    """RMA training args (hora-style PPO + proprio adaptation)."""

    # --- Paths / run ---
    config: str = "configs/new_juggle/zeroshot_ablations/sim_paramrand_pm25.yaml"
    args_file: Optional[str] = None
    log_parent_dir: str = "runs/rma/"
    run_name: str = "default_run_name"
    device: str = "cuda:0"
    seed: int = 0

    # --- Env ---
    num_envs: int = 8
    # Context length T for φ (HistoryBuffer of paddle/puck pos+valid; no actions).
    prop_hist_len: int = 30
    # Policy input is always raw air-hockey obs ⊕ z. We deliberately do NOT use
    # TD3's use_last_action_in_policy_state.

    # --- Network ---
    actor_units: List[int] = field(default_factory=lambda: [512, 256, 128])
    # priv MLP ends at latent z dim; default 8 matches hora embed size.
    priv_mlp_units: List[int] = field(default_factory=lambda: [128, 64, 8])

    # --- Privileged / RMA flags ---
    priv_info: bool = True
    # priv_info_dim is inferred from air_hockey.random_variables unless set.
    priv_info_dim: Optional[int] = None

    # --- PPO (phase 1) ---
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    gamma: float = 0.99
    tau: float = 0.95  # GAE λ
    e_clip: float = 0.2
    clip_value: bool = True
    entropy_coef: float = 0.0
    critic_coef: float = 4.0
    bounds_loss_coef: float = 0.0001
    truncate_grads: bool = True
    grad_norm: float = 1.0
    value_bootstrap: bool = True
    normalize_advantage: bool = True
    normalize_input: bool = True
    normalize_value: bool = True
    reward_scale: float = 1.0
    horizon_length: int = 128
    minibatch_size: int = 512
    mini_epochs: int = 5
    kl_threshold: float = 0.02
    save_frequency: int = 50
    save_best_after: int = 0
    # Phase-1 agent-steps budget (env-steps ≈ agent_steps with vector envs).
    max_agent_steps: int = 2_000_000

    # --- Phase 2 (adaptation) ---
    # Set 0 to skip phase 2. Phase 2 runs automatically after phase 1.
    adaptation_max_agent_steps: int = 500_000
    adaptation_lr: float = 3e-4
    adaptation_save_interval: int = 50_000
    # Skip phase 1 and load a prior stage1 checkpoint (strict=False into phase 2).
    phase1_checkpoint: Optional[str] = None
    skip_phase1: bool = False

    # --- Eval (consumed by rma_training_dr) ---
    eval_param_seed: Optional[int] = None
    eval_n_envs: int = 5
    eval_eps_per_env: int = 4
    checkpoint_eval_episodes: int = 4

    # --- sbatch (used only when --sbatch is passed; see _submit_sbatch_job) ---
    sbatch_run_name: Optional[str] = None  # e.g. "rma_paramrand_pm25_seed0"
    sbatch_partition: str = "gh"
    sbatch_time: str = "12:00:00"

    # Optional hook: evaluate_agent callable monkey-patched by rma_training_dr.
    # Not a CLI field — set at runtime.
    _evaluate_agent: Optional[object] = None


_SLURM_TEMPLATE_PATH = "scripts/rma/vista_template.slurm"


def _submit_sbatch_job():
    """Patch the RMA slurm template and submit via sbatch (mirrors TD3 helper)."""
    raw_argv = sys.argv[1:]
    forward_argv = [a for a in raw_argv if a != "--sbatch"]

    def _pop_flag(argv, flag, default=None):
        argv = list(argv)
        if flag in argv:
            idx = argv.index(flag)
            value = argv[idx + 1] if idx + 1 < len(argv) else default
            argv.pop(idx)
            if idx < len(argv):
                argv.pop(idx)
            return argv, value
        return argv, default

    forward_argv, sbatch_run_name = _pop_flag(forward_argv, "--sbatch-run-name")
    forward_argv, sbatch_partition = _pop_flag(
        forward_argv, "--sbatch-partition", default="gh"
    )
    forward_argv, sbatch_time = _pop_flag(
        forward_argv, "--sbatch-time", default="12:00:00"
    )

    if sbatch_run_name is None:
        if "--run-name" in forward_argv:
            idx = forward_argv.index("--run-name")
            sbatch_run_name = forward_argv[idx + 1]
        else:
            sbatch_run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    log_parent_dir = f"runs/rma/{sbatch_run_name}"
    results_dir = f"results/{sbatch_run_name}"

    if "--log-parent-dir" in forward_argv:
        idx = forward_argv.index("--log-parent-dir")
        forward_argv[idx + 1] = log_parent_dir
    else:
        forward_argv += ["--log-parent-dir", log_parent_dir]

    if "--run-name" in forward_argv:
        idx = forward_argv.index("--run-name")
        forward_argv[idx + 1] = sbatch_run_name
    else:
        forward_argv += ["--run-name", sbatch_run_name]

    # Always launch the DR wrapper (multi-env eval hooks), matching TD3.
    python_cmd = f"python -m scripts.rma.rma_training_dr {shlex.join(forward_argv)}"

    try:
        with open(_SLURM_TEMPLATE_PATH, "r") as f:
            template = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Slurm template not found at '{_SLURM_TEMPLATE_PATH}'. "
            "Set _SLURM_TEMPLATE_PATH in rma_training.py."
        )

    patched_lines = []
    time_patched = False
    partition_patched = False
    for line in template.splitlines():
        if line.startswith("#SBATCH --job-name="):
            line = f"#SBATCH --job-name={sbatch_run_name}"
        elif line.startswith("#SBATCH --partition="):
            line = f"#SBATCH --partition={sbatch_partition}"
            partition_patched = True
        elif line.startswith("#SBATCH --time="):
            line = f"#SBATCH --time={sbatch_time}"
            time_patched = True
        patched_lines.append(line)

    last_sbatch_idx = max(i for i, l in enumerate(patched_lines) if l.startswith("#SBATCH"))
    if not time_patched:
        patched_lines.insert(last_sbatch_idx + 1, f"#SBATCH --time={sbatch_time}")
    if not partition_patched:
        patched_lines.insert(
            last_sbatch_idx + 1, f"#SBATCH --partition={sbatch_partition}"
        )

    slurm_script = "\n".join(patched_lines).rstrip() + f"\n{python_cmd}\n"

    os.makedirs("slurm_jobs", exist_ok=True)
    tmp_path = "slurm_jobs/temp_submission.slurm"
    with open(tmp_path, "w") as f:
        f.write(slurm_script)

    print("=" * 60)
    print(f"run_name     : {sbatch_run_name}")
    print(f"partition    : {sbatch_partition}")
    print(f"time         : {sbatch_time}")
    print(f"logs (slurm) : slurm_jobs/job_<jid>/")
    print(f"runs (tb)    : {log_parent_dir}")
    print(f"results      : {results_dir}")
    print(f"cmd          : {python_cmd}")
    print("=" * 60)

    result = subprocess.run(["sbatch", tmp_path], capture_output=True, text=True)
    print(result.stdout.strip())
    if result.stderr.strip():
        print("stderr:", result.stderr.strip())

    if "Submitted batch job" in result.stdout:
        jid = result.stdout.strip().split()[-1]
        job_dir = f"slurm_jobs/job_{jid}"
        os.makedirs(job_dir, exist_ok=True)
        with open(f"{job_dir}/submission.slurm", "w") as f:
            f.write(slurm_script)
        with open(f"{job_dir}/submitted_argv.txt", "w") as f:
            f.write(" ".join(sys.argv) + "\n")
        with open(f"{job_dir}/job_info.json", "w") as f:
            json.dump(
                {
                    "job_id": jid,
                    "run_name": sbatch_run_name,
                    "partition": sbatch_partition,
                    "time": sbatch_time,
                    "log_parent_dir": log_parent_dir,
                    "results_dir": results_dir,
                    "python_cmd": python_cmd,
                },
                f,
                indent=2,
            )
        print(f"Job artifacts saved to {job_dir}/")
    else:
        raise RuntimeError(
            f"sbatch submission failed:\n{result.stdout}\n{result.stderr}"
        )


def _load_air_hockey_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    if "air_hockey" not in cfg:
        raise KeyError(f"Config {path} missing top-level 'air_hockey' key.")
    return cfg


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(os.getcwd(), path))


def _build_trainer_cfg(args: Args, priv_info_dim: int, proprio_adapt: bool) -> SimpleNamespace:
    # Context entry = 6-d paddle/puck pos+valid (HistoryBuffer.HISTORY_ENTRY_DIM).
    from scripts.rma.history_buffer import HISTORY_ENTRY_DIM

    return SimpleNamespace(
        actor_units=list(args.actor_units),
        priv_mlp_units=list(args.priv_mlp_units),
        priv_info=bool(args.priv_info),
        proprio_adapt=bool(proprio_adapt),
        priv_info_dim=int(priv_info_dim),
        proprio_hist_input_dim=int(HISTORY_ENTRY_DIM),
        num_actors=int(args.num_envs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        gamma=float(args.gamma),
        tau=float(args.tau),
        e_clip=float(args.e_clip),
        clip_value=bool(args.clip_value),
        entropy_coef=float(args.entropy_coef),
        critic_coef=float(args.critic_coef),
        bounds_loss_coef=float(args.bounds_loss_coef),
        truncate_grads=bool(args.truncate_grads),
        grad_norm=float(args.grad_norm),
        value_bootstrap=bool(args.value_bootstrap),
        normalize_advantage=bool(args.normalize_advantage),
        normalize_input=bool(args.normalize_input),
        normalize_value=bool(args.normalize_value),
        reward_scale=float(args.reward_scale),
        horizon_length=int(args.horizon_length),
        minibatch_size=int(args.minibatch_size),
        mini_epochs=int(args.mini_epochs),
        kl_threshold=float(args.kl_threshold),
        save_frequency=int(args.save_frequency),
        save_best_after=int(args.save_best_after),
        max_agent_steps=int(args.max_agent_steps),
        adaptation_max_agent_steps=int(args.adaptation_max_agent_steps),
        adaptation_lr=float(args.adaptation_lr),
        adaptation_save_interval=int(getattr(args, "adaptation_save_interval", 50_000)),
        wandb_step_offset=int(getattr(args, "wandb_step_offset", 0)),
    )


def _make_output_dir(args: Args) -> str:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(
        args.log_parent_dir, f"{args.run_name}_{stamp}_seed_{args.seed}"
    )
    if os.path.exists(run_dir):
        base = run_dir
        i = 1
        while os.path.exists(run_dir):
            run_dir = f"{base}r{i}"
            i += 1
        print(f"[rma_training] Log directory exists. Using alternate: {run_dir}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _init_wandb(args: Args, air_hockey_cfg: dict, run_dir: str):
    """Initialize wandb the same way as scripts/td3/td3_training.py."""
    import wandb

    # Drop non-serializable private fields from the logged args.
    cli_args = {
        k: getattr(args, k)
        for k in args.__dataclass_fields__
        if not k.startswith("_")
    }
    full_trackable_config = {"yaml_config": air_hockey_cfg, "cli_args": cli_args, "run_dir": run_dir}
    return wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=args.run_name,
        name=f"{args.run_name}_seed_{args.seed}",
        config=full_trackable_config,
        dir=run_dir,
    )


def _save_run_config(run_dir: str, args: Args, air_hockey_cfg: dict) -> None:
    out = {
        "args": {k: getattr(args, k) for k in args.__dataclass_fields__ if not k.startswith("_")},
        "air_hockey_config": air_hockey_cfg,
    }
    # Convert non-YAML-friendly types.
    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    with open(os.path.join(run_dir, "rma_config.yaml"), "w") as f:
        yaml.dump(_sanitize(out), f, default_flow_style=False)


def parse_args() -> Args:
    """Load optional YAML via --args-file, then apply tyro CLI overrides."""
    args_file = None
    if "--args-file" in sys.argv:
        idx = sys.argv.index("--args-file")
        if idx + 1 < len(sys.argv):
            args_file = sys.argv[idx + 1]

    defaults = Args()
    if args_file is not None:
        with open(args_file, "r") as f:
            file_args = yaml.load(f, Loader=yaml.FullLoader) or {}
        # Drop unknown keys quietly for forward-compat with older TD3 YAMLs.
        known = set(Args.__dataclass_fields__.keys())
        filtered = {k: v for k, v in file_args.items() if k in known and not k.startswith("_")}
        for k, v in filtered.items():
            setattr(defaults, k, v)
        defaults.args_file = args_file

    # Strip --args-file so tyro doesn't complain about an unknown flag.
    argv = []
    skip_next = False
    for i, tok in enumerate(sys.argv[1:]):
        if skip_next:
            skip_next = False
            continue
        if tok == "--args-file":
            skip_next = True
            continue
        argv.append(tok)
    return tyro.cli(Args, default=defaults, args=argv)


def _entrypoint(args: Optional[Args] = None) -> str:
    """Run phase 1 then phase 2. Returns the run output directory."""
    # Submit path must run before tyro/parse_args (sbatch flags are not Args fields).
    if args is None and "--sbatch" in sys.argv:
        _submit_sbatch_job()
        return ""

    if args is None:
        args = parse_args()

    set_seed(int(args.seed))
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[rma_training] CUDA unavailable ({device}); falling back to cpu.")
        device = "cpu"
        args.device = device

    air_hockey_cfg = _load_air_hockey_config(_resolve_path(args.config))
    air_hockey_params = air_hockey_cfg["air_hockey"]
    if not air_hockey_params.get("domain_random", False):
        print(
            "[rma_training] WARNING: domain_random is False in the sim config; "
            "RMA typically requires per-reset physics randomization."
        )

    priv_keys = air_hockey_params.get("random_variables") or []
    if not priv_keys:
        raise ValueError("RMA requires air_hockey.random_variables to be non-empty.")
    priv_info_dim = int(args.priv_info_dim) if args.priv_info_dim is not None else len(priv_keys)

    # PPO batch constraint: horizon_length * num_envs % minibatch_size == 0
    batch_size = int(args.horizon_length) * int(args.num_envs)
    if batch_size % int(args.minibatch_size) != 0:
        raise ValueError(
            f"horizon_length * num_envs ({batch_size}) must be divisible by "
            f"minibatch_size ({args.minibatch_size})."
        )

    run_dir = _make_output_dir(args)
    _save_run_config(run_dir, args, air_hockey_cfg)
    print(f"[rma_training] output → {run_dir}")

    wandb_run = None
    try:
        wandb_run = _init_wandb(args, air_hockey_cfg, run_dir)
        print(f"[rma_training] wandb run → {getattr(wandb_run, 'url', wandb_run)}")
    except Exception as exc:
        print(f"[rma_training] wandb.init failed (continuing without wandb): {exc}")
    
    # TODO: why do we need RMAVecEnv
    env = RMAVecEnv(
        air_hockey_params=air_hockey_params,
        num_envs=int(args.num_envs),
        device=device,
        prop_hist_len=int(args.prop_hist_len),
        seed=int(args.seed),
    )
    assert env.priv_info_dim == priv_info_dim, (
        f"env priv dim {env.priv_info_dim} != expected {priv_info_dim}"
    )

    phase1_ckpt = args.phase1_checkpoint
    phase1_steps = 0
    try:
        # -------------------- Phase 1: μ + π via PPO --------------------
        if not args.skip_phase1:
            print("[rma_training] === Phase 1: PPO (privileged encoder μ + base policy π) ===")
            phase1_cfg = _build_trainer_cfg(args, priv_info_dim, proprio_adapt=False)
            ppo = PPO(env, run_dir, phase1_cfg, device)
            if args.phase1_checkpoint:
                print(f"[rma_training] Resuming phase 1 from {args.phase1_checkpoint}")
                ppo.restore_train(args.phase1_checkpoint)
            ppo.train()
            phase1_steps = int(ppo.agent_steps)
            # phase1_ckpt = os.path.join(ppo.nn_dir, "best.pth")
            # if not os.path.exists(phase1_ckpt):
            phase1_ckpt = os.path.join(ppo.nn_dir, "last.pth")
            print(f"[rma_training] Phase 1 complete. Checkpoint: {phase1_ckpt}")
        else:
            if not phase1_ckpt:
                raise ValueError("skip_phase1=True requires phase1_checkpoint.")
            print(f"[rma_training] Skipping phase 1; using {phase1_ckpt}")

        # -------------------- Phase 2: φ via supervised L2 --------------------
        if int(args.adaptation_max_agent_steps) > 0:
            print("[rma_training] === Phase 2: ProprioAdapt (freeze μ/π, train φ) ===")
            phase2_cfg = _build_trainer_cfg(args, priv_info_dim, proprio_adapt=True)
            phase2_cfg.wandb_step_offset = phase1_steps
            padapt = ProprioAdapt(env, run_dir, phase2_cfg, device)
            padapt.load_phase1_checkpoint(phase1_ckpt)
            padapt.train()
            phase2_best = os.path.join(padapt.nn_dir, "model_best.ckpt")
            print(f"[rma_training] Phase 2 complete. Best checkpoint: {phase2_best}")
        else:
            print("[rma_training] adaptation_max_agent_steps=0 → skipping phase 2.")
    finally:
        env.close()
        wandb_finish()

    return run_dir


if __name__ == "__main__":
    _entrypoint()
