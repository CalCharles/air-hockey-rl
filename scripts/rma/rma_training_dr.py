"""
RMA training entrypoint with multi-environment evaluation across a fixed
seed-sampled set of dynamics-parameter dicts.

Wraps `rma_training` (phase 1 PPO → phase 2 ProprioAdapt in one call) and
attaches an eval callback so periodic checkpoints are scored across
`eval_n_envs` fixed parameter overlays (sampled once with `eval_param_seed`).

Multi-env eval semantics match the former td3_training_dr wrapper:
- Overrides sampled once from air_hockey.random_variable_ranges.
- Per-env + aggregate stats → <ckpt_dir>/multi_env_eval.json
- eval_envs.json written once under log_parent_dir / run dir.

Launch:
  python -m scripts.rma.rma_training_dr \\
    --args-file configs/rma/rma_paramrand_pm25.yaml
"""

from __future__ import annotations

import copy
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from scripts.rma import rma_training
from scripts.rma.evaluate import evaluate_agent, rollout_returns


# Module-level state populated at startup from the args YAML.
_EVAL_PARAM_SEED: Optional[int] = None
_EVAL_N_ENVS: int = 1
_EVAL_EPS_PER_ENV: int = 4
_EVAL_ENV_OVERRIDES: Optional[List[Dict[str, float]]] = None
_LOG_PARENT_DIR: Optional[str] = None
_EVAL_CALL_COUNT: int = 0
_RUN_ARGS: Optional[rma_training.Args] = None
_AIR_HOCKEY_PARAMS: Optional[Dict[str, Any]] = None


def _sample_eval_env_overrides(
    seed: int,
    n_envs: int,
    random_variable_ranges: dict,
    random_variables: list,
) -> List[Dict[str, float]]:
    rng = np.random.RandomState(int(seed))
    overrides: List[Dict[str, float]] = [{} for _ in range(n_envs)]
    for var in random_variables:
        if var not in random_variable_ranges:
            raise KeyError(
                f"random_variables lists '{var}' but random_variable_ranges "
                f"has no entry (keys: {list(random_variable_ranges.keys())})"
            )
        low, high = random_variable_ranges[var]
        for env_idx in range(n_envs):
            overrides[env_idx][var] = float(rng.uniform(float(low), float(high)))
    return overrides


def _apply_overrides(air_hockey_params: dict, overrides: dict) -> dict:
    cfg = copy.deepcopy(air_hockey_params)
    sim_params = cfg.setdefault("simulator_params", {})
    for var, value in overrides.items():
        sim_params[var] = value
    cfg["domain_random"] = False
    return cfg


def _ensure_eval_overrides(air_hockey_params: dict) -> List[Dict[str, float]]:
    global _EVAL_ENV_OVERRIDES
    if _EVAL_ENV_OVERRIDES is not None:
        return _EVAL_ENV_OVERRIDES
    if _EVAL_PARAM_SEED is None:
        return [{}]
    random_variables = list(air_hockey_params.get("random_variables", []))
    random_variable_ranges = dict(air_hockey_params.get("random_variable_ranges", {}))
    if not random_variables or not random_variable_ranges:
        raise ValueError(
            "rma_training_dr requires random_variables and random_variable_ranges "
            "in the air_hockey config when eval_param_seed is set."
        )
    _EVAL_ENV_OVERRIDES = _sample_eval_env_overrides(
        seed=_EVAL_PARAM_SEED,
        n_envs=_EVAL_N_ENVS,
        random_variable_ranges=random_variable_ranges,
        random_variables=random_variables,
    )
    dump_dir = _LOG_PARENT_DIR
    if dump_dir is not None:
        os.makedirs(dump_dir, exist_ok=True)
        with open(os.path.join(dump_dir, "eval_envs.json"), "w") as f:
            json.dump(
                {
                    "eval_param_seed": _EVAL_PARAM_SEED,
                    "n_envs": _EVAL_N_ENVS,
                    "eps_per_env": _EVAL_EPS_PER_ENV,
                    "random_variables": random_variables,
                    "random_variable_ranges": {
                        k: list(v) for k, v in random_variable_ranges.items()
                    },
                    "overrides": _EVAL_ENV_OVERRIDES,
                },
                f,
                indent=2,
            )
        print(
            f"[rma_training_dr] Sampled {_EVAL_N_ENVS} eval envs "
            f"(seed={_EVAL_PARAM_SEED}) → {os.path.join(dump_dir, 'eval_envs.json')}"
        )
    return _EVAL_ENV_OVERRIDES


def multi_env_evaluate(
    checkpoint_path: str,
    save_dir: str,
    air_hockey_params: dict,
    stage: str,
    actor_units,
    priv_mlp_units,
    prop_hist_len: int,
    device: str,
    n_gifs: int = 1,
):
    """Score a checkpoint across the fixed eval-env overrides."""
    global _EVAL_CALL_COUNT
    overrides = _ensure_eval_overrides(air_hockey_params)
    _EVAL_CALL_COUNT += 1
    os.makedirs(save_dir, exist_ok=True)

    per_env: List[Dict[str, Any]] = []
    for env_idx, override in enumerate(overrides):
        eval_cfg = _apply_overrides(air_hockey_params, override)
        eval_cfg["seed"] = int(
            ((_EVAL_PARAM_SEED or 0) * 100000) + (env_idx * 1000) + _EVAL_CALL_COUNT
        )
        if env_idx == 0:
            try:
                evaluate_agent(
                    checkpoint_path=checkpoint_path,
                    save_dir=save_dir,
                    air_hockey_params=eval_cfg,
                    actor_units=actor_units,
                    priv_mlp_units=priv_mlp_units,
                    prop_hist_len=prop_hist_len,
                    n_eps=_EVAL_EPS_PER_ENV,
                    n_gifs=n_gifs,
                    device=device,
                    force_phase=stage,
                    write_metrics=False,
                )
            except Exception as exc:
                print(f"[rma_training_dr] env0 GIF eval failed (continuing): {exc}")

        try:
            stats = rollout_returns(
                air_hockey_params=eval_cfg,
                checkpoint_path=checkpoint_path,
                n_eps=_EVAL_EPS_PER_ENV,
                actor_units=list(actor_units),
                priv_mlp_units=list(priv_mlp_units),
                prop_hist_len=prop_hist_len,
                device=device,
                force_phase=stage,
            )
            stats["env_idx"] = env_idx
            stats["override"] = override
            per_env.append(stats)
        except Exception as exc:
            print(f"[rma_training_dr] env{env_idx} rollout failed (continuing): {exc}")
            per_env.append({"env_idx": env_idx, "override": override, "error": str(exc)})

    valid = [r for r in per_env if "error" not in r]
    aggregate = {
        "n_envs_used": len(valid),
        "eps_per_env": _EVAL_EPS_PER_ENV,
        "mean_return_across_envs": (
            float(np.mean([r["mean_return"] for r in valid])) if valid else float("nan")
        ),
        "mean_success_across_envs": (
            float(np.mean([r["mean_success_rate"] for r in valid])) if valid else float("nan")
        ),
        "mean_ep_length_across_envs": (
            float(np.mean([r["mean_episode_length"] for r in valid])) if valid else float("nan")
        ),
        "per_env_mean_return": [r.get("mean_return", float("nan")) for r in per_env],
        "per_env_mean_success": [r.get("mean_success_rate", float("nan")) for r in per_env],
        "stage": stage,
        "checkpoint": checkpoint_path,
    }
    out_path = os.path.join(save_dir, "multi_env_eval.json")
    with open(out_path, "w") as f:
        json.dump({"aggregate": aggregate, "per_env": per_env}, f, indent=2)

    per_env_str = ", ".join(f"{x:.1f}" for x in aggregate["per_env_mean_return"])
    print(
        f"[rma_training_dr] Multi-env eval ({stage}, n_envs={len(overrides)}, "
        f"eps/env={_EVAL_EPS_PER_ENV}): mean_return={aggregate['mean_return_across_envs']:.2f}, "
        f"per_env_returns=[{per_env_str}]"
    )
    return aggregate


def _make_eval_callback(args: rma_training.Args, air_hockey_params: dict, run_dir: str):
    def _cb(checkpoint_path: str, stage: str):
        if _EVAL_PARAM_SEED is None:
            return
        # Phase-1 privileged eval is optional; default to scoring phase-2 only
        # (deploy-relevant). Set RMA_EVAL_PHASE1=1 to also score stage1.
        if stage == "phase1" and os.environ.get("RMA_EVAL_PHASE1", "0") != "1":
            return
        ckpt_tag = os.path.splitext(os.path.basename(checkpoint_path))[0]
        save_dir = os.path.join(run_dir, f"eval_{stage}_{ckpt_tag}")
        multi_env_evaluate(
            checkpoint_path=checkpoint_path,
            save_dir=save_dir,
            air_hockey_params=air_hockey_params,
            stage=stage,
            actor_units=args.actor_units,
            priv_mlp_units=args.priv_mlp_units,
            prop_hist_len=args.prop_hist_len,
            device=args.device if not args.device.startswith("cuda") else "cpu",
            n_gifs=1 if stage == "phase2" else 0,
        )

    return _cb


def _entrypoint_dr():
    global _EVAL_PARAM_SEED, _EVAL_N_ENVS, _EVAL_EPS_PER_ENV, _LOG_PARENT_DIR
    global _RUN_ARGS, _AIR_HOCKEY_PARAMS

    args = rma_training.parse_args()
    _RUN_ARGS = args
    _EVAL_PARAM_SEED = args.eval_param_seed
    _EVAL_N_ENVS = int(args.eval_n_envs)
    _EVAL_EPS_PER_ENV = int(args.eval_eps_per_env)
    _LOG_PARENT_DIR = args.log_parent_dir

    print(
        f"[rma_training_dr] eval_param_seed={_EVAL_PARAM_SEED}, "
        f"eval_n_envs={_EVAL_N_ENVS}, eval_eps_per_env={_EVAL_EPS_PER_ENV}, "
        f"log_parent_dir={_LOG_PARENT_DIR}"
    )

    # Load sim config so we can attach the eval callback with air_hockey params.
    with open(rma_training._resolve_path(args.config), "r") as f:
        air_cfg = yaml.load(f, Loader=yaml.FullLoader)
    air_hockey_params = air_cfg["air_hockey"]
    _AIR_HOCKEY_PARAMS = air_hockey_params

    # Pre-create run dir so eval_envs.json can live next to training outputs.
    # rma_training._entrypoint also creates a stamped run dir; we pass the
    # callback via cfg inside a patched _build_trainer_cfg.
    original_build = rma_training._build_trainer_cfg
    run_dir_box: Dict[str, str] = {}

    def _build_with_callback(args_inner, priv_info_dim, proprio_adapt):
        cfg = original_build(args_inner, priv_info_dim, proprio_adapt)
        # run_dir is not known yet at first build; callback closes over box.
        def _lazy_cb(checkpoint_path, stage):
            rd = run_dir_box.get("run_dir")
            if rd is None:
                return
            _make_eval_callback(args_inner, air_hockey_params, rd)(
                checkpoint_path=checkpoint_path, stage=stage
            )

        cfg.eval_callback = _lazy_cb
        return cfg

    original_make_output = rma_training._make_output_dir

    def _make_output_tracked(args_inner):
        rd = original_make_output(args_inner)
        run_dir_box["run_dir"] = rd
        global _LOG_PARENT_DIR
        _LOG_PARENT_DIR = rd
        if _EVAL_PARAM_SEED is not None:
            _ensure_eval_overrides(air_hockey_params)
        return rd

    rma_training._build_trainer_cfg = _build_with_callback
    rma_training._make_output_dir = _make_output_tracked

    try:
        run_dir = rma_training._entrypoint(args)
    finally:
        rma_training._build_trainer_cfg = original_build
        rma_training._make_output_dir = original_make_output

    # Final multi-env eval on stage2 best (or stage1 best if phase 2 skipped).
    stage2_best = os.path.join(run_dir, "stage2_nn", "model_best.ckpt")
    stage1_best = os.path.join(run_dir, "stage1_nn", "best.pth")
    if os.path.exists(stage2_best):
        multi_env_evaluate(
            checkpoint_path=stage2_best,
            save_dir=os.path.join(run_dir, "eval_final_phase2"),
            air_hockey_params=air_hockey_params,
            stage="phase2",
            actor_units=args.actor_units,
            priv_mlp_units=args.priv_mlp_units,
            prop_hist_len=args.prop_hist_len,
            device=args.device if not str(args.device).startswith("cuda") else "cpu",
            n_gifs=1,
        )
    elif os.path.exists(stage1_best):
        multi_env_evaluate(
            checkpoint_path=stage1_best,
            save_dir=os.path.join(run_dir, "eval_final_phase1"),
            air_hockey_params=air_hockey_params,
            stage="phase1",
            actor_units=args.actor_units,
            priv_mlp_units=args.priv_mlp_units,
            prop_hist_len=args.prop_hist_len,
            device=args.device if not str(args.device).startswith("cuda") else "cpu",
            n_gifs=1,
        )

    print(f"[rma_training_dr] done → {run_dir}")
    return run_dir


if __name__ == "__main__":
    _entrypoint_dr()
