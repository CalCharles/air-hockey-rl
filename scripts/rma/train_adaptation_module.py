"""RMA phase 2: adaptation module phi(x_{t-H:t-1}, a_{t-H:t-1}) -> z_hat_t.

Supervised regression of the latent z_t = mu(e_t) (mu and pi are frozen
phase-1 outputs) from the recent state/action history, trained on
**on-policy** data: the base policy is rolled out in the randomized sim
*with the current adaptation module's estimate* z_hat_t (not with the
privileged z_t), so phi sees exactly the state distribution the deployed
policy induces — the iterative scheme of the paper (Sec. 3.2).

Loop (``n_iterations`` times):
  1. collect ``steps_per_iteration`` env steps with pi(x, a_prev, phi(history));
  2. add the episodes to a rolling dataset (``dataset_max_steps``; a fixed
     fraction of episodes is held out for validation);
  3. ``epochs_per_iteration`` epochs of MSE regression on the train split;
  4. validation metrics: latent MSE (overall / per dim / by steps-since-reset
     bucket), per-dim R^2, and physics-parameter recovery (a ridge probe
     z_hat -> e fitted on train, R^2 per parameter on validation);
  5. every ``eval_every_iterations``: policy return of the *adapted* policy
     on the fixed DR eval-env set (ID), plus the OOD set if the sim config
     has ``random_variable_ranges_OOD``.

The final evaluation runs adapted / privileged (oracle upper bound) /
nominal (z = mu(0), i.e. RMA without adaptation) and optionally a plain DR
TD3 actor (``baseline_model_path``) on the same env seeds -> ``phase2_summary.json``.

Run:
    .venv/bin/python -m scripts.rma.train_adaptation_module --args-file configs/rma/rma_juggle_dr_phase2.yaml \
        --phase1-dir runs/rma/<phase1 run>
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tyro
import yaml
from torch.utils.tensorboard import SummaryWriter

from airhockey import AirHockeyEnv
from scripts.rma.bundle import adaptation_meta, load_phase1, save_adaptation_module
from scripts.rma.env_wrapper import EnvParamNormalizer, raw_env_params
from scripts.rma.evaluate import (
    AdaptedRMAAgent,
    EvalAgent,
    NominalLatentAgent,
    PlainTD3Agent,
    PrivilegedRMAAgent,
    evaluate_multi_env,
    make_eval_env,
)
from scripts.rma.history import AdaptationDataset
from scripts.rma.networks import AdaptationModule, RMAActor, step_feature_dim, step_state_features

T_BUCKETS = ((0, 10), (10, 25), (25, 50), (50, 10**9))


@dataclass
class Phase2Args:
    """Phase-2 (adaptation module) args. Can be given as YAML via --args-file."""

    # Phase-1 run dir / checkpoint dir / model.pth (needs rma_meta.json next to it).
    phase1_dir: str = ""
    args_file: str | None = None
    # Sim config for the on-policy rollouts; default = the phase-1 config (DR on).
    config: str | None = None
    log_parent_dir: str | None = None
    run_name: str = "rma_phase2"
    device: str = "cuda:0"
    seed: int = 0
    torch_num_threads: int = 1

    # --- adaptation module (paper: H=50, 32-dim embed, conv 32ch k=8/5/5 s=4/1/1) ---
    history_len: int = 50
    step_features: str = "latest_frame"  # "latest_frame" (6 + act dims) | "full_obs" (30 + act dims)
    embed_dim: int = 32
    conv_channels: int = 32

    # --- data collection ---
    n_iterations: int = 20
    steps_per_iteration: int = 20000
    dataset_max_steps: int = 400000
    holdout_episode_fraction: float = 0.1
    # Gaussian action noise on the phase-2 rollouts (TD3's actor is deterministic;
    # 0 = exactly the deployed policy, the paper's on-policy setting).
    rollout_action_noise: float = 0.0
    # Iterations whose rollouts use the privileged z (off-policy warm start; paper: 0).
    privileged_warmup_iterations: int = 0

    # --- regression ---
    epochs_per_iteration: int = 4
    batch_size: int = 256
    lr: float = 5.0e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    # --- evaluation ---
    eval_every_iterations: int = 5
    eval_param_seed: int | None = None  # default: phase-1 args
    eval_n_envs: int | None = None
    eval_eps_per_env: int | None = None
    final_eval_eps_per_env: int = 10
    eval_ood: bool = True
    eval_ood_n_envs: int = 5
    baseline_model_path: str | None = None  # plain DR TD3 actor for the paired comparison
    baseline_hidden_layer_size: int = 64
    baseline_num_hidden_layers: int = 2
    gifs: bool = True


# ---------------------------------------------------------------- collection
def collect_steps(
    env,
    agent: EvalAgent,
    normalizer: EnvParamNormalizer,
    encoder,
    n_steps: int,
    step_features: str,
    dataset: AdaptationDataset,
    holdout: AdaptationDataset,
    holdout_fraction: float,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Roll `agent` for >= n_steps (whole episodes) and append episodes to the datasets."""
    act_dim = int(np.prod(env.action_space.shape))
    steps_done, episodes, returns, lengths, successes = 0, 0, [], [], []
    online_sq_err: List[float] = []
    while steps_done < n_steps:
        obs, _ = env.reset()
        obs = np.asarray(obs, dtype=np.float32)
        e_norm = normalizer.normalize(raw_env_params(env, normalizer.random_variables))
        with torch.no_grad():
            z = encoder(torch.as_tensor(e_norm, dtype=torch.float32).reshape(1, -1)).numpy().reshape(-1)
        agent.reset(obs, e_norm)
        last_action = np.zeros(act_dim, dtype=np.float32)
        feats, acts = [], []
        done, cum_rew, steps, info = False, 0.0, 0, {}
        while not done:
            feats.append(step_state_features(torch.as_tensor(obs), step_features).numpy())
            action = agent.act(obs, last_action)
            acts.append(np.asarray(action, dtype=np.float32).reshape(-1))
            obs, rew, term, trunc, info = env.step(action)
            obs = np.asarray(obs, dtype=np.float32)
            rew = float(np.asarray(rew, dtype=np.float64).reshape(-1)[0])
            cum_rew += rew
            steps += 1
            done = bool(term or trunc)
            last_action = acts[-1]
        extras = agent.episode_extras()
        if "latent_mse" in extras and np.isfinite(extras["latent_mse"]):
            online_sq_err.append(extras["latent_mse"])
        target = holdout if rng.random() < holdout_fraction else dataset
        target.add_episode(np.stack(feats), np.stack(acts), z, e_norm, cum_rew)
        steps_done += steps
        episodes += 1
        returns.append(cum_rew)
        lengths.append(steps)
        successes.append(int(bool(info.get("success", False))) if info else 0)
    return {
        "collect/steps": float(steps_done),
        "collect/episodes": float(episodes),
        "collect/mean_return": float(np.mean(returns)),
        "collect/mean_length": float(np.mean(lengths)),
        "collect/success_rate": float(np.mean(successes)),
        "collect/online_latent_mse": float(np.mean(online_sq_err)) if online_sq_err else float("nan"),
    }


# ---------------------------------------------------------------- regression
def train_epochs(module: AdaptationModule, optimizer, dataset: AdaptationDataset, epochs: int, batch_size: int, grad_clip: float, device, rng: np.random.Generator) -> float:
    module.train()
    idx_all = dataset.all_indices()
    losses = []
    for _ in range(int(epochs)):
        perm = rng.permutation(idx_all)
        for start in range(0, len(perm), int(batch_size)):
            batch_idx = perm[start : start + int(batch_size)]
            x, y = dataset.batch(batch_idx, device)
            pred = module(x)
            loss = torch.nn.functional.mse_loss(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(module.parameters(), grad_clip)
            optimizer.step()
            losses.append(float(loss.item()))
    module.eval()
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def predict_all(module: AdaptationModule, dataset: AdaptationDataset, device, batch_size: int = 4096) -> np.ndarray:
    module.eval()
    idx_all = dataset.all_indices()
    preds = []
    for start in range(0, len(idx_all), batch_size):
        x = torch.as_tensor(dataset.windows(idx_all[start : start + batch_size]), dtype=torch.float32, device=device)
        preds.append(module(x).cpu().numpy())
    return np.concatenate(preds, axis=0)


def _r2(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-column R^2 (nan when the target column is constant)."""
    ss_res = ((target - pred) ** 2).sum(axis=0)
    var = target.var(axis=0) * target.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(var > 1e-12, 1.0 - ss_res / var, np.nan)


def _ridge_fit(x: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    xb = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    a = xb.T @ xb + lam * np.eye(xb.shape[1], dtype=xb.dtype)
    return np.linalg.solve(a, xb.T @ y)


def _ridge_predict(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    xb = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    return xb @ w


def latent_fit_metrics(module: AdaptationModule, train: AdaptationDataset, holdout: AdaptationDataset, device, random_variables: List[str]) -> Dict[str, Any]:
    """Validation metrics of phi on held-out episodes (+ the ridge parameter probe)."""
    out: Dict[str, Any] = {}
    pred_tr = predict_all(module, train, device)
    z_tr = train.targets(train.all_indices())
    out["fit/train_mse"] = float(((pred_tr - z_tr) ** 2).mean())
    if holdout.num_steps == 0:
        return out
    idx = holdout.all_indices()
    pred = predict_all(module, holdout, device)
    z = holdout.targets(idx)
    e = holdout.env_params(idx)
    t_in_ep = holdout.steps_in_episode(idx)
    sq = (pred - z) ** 2
    out["fit/val_mse"] = float(sq.mean())
    out["fit/val_target_var"] = float(z.var(axis=0).mean())
    r2 = _r2(pred, z)
    out["fit/val_r2_mean"] = float(np.nanmean(r2))
    out["fit/val_r2_per_dim"] = [float(v) for v in r2]
    out["fit/val_mse_per_dim"] = [float(v) for v in sq.mean(axis=0)]
    for lo, hi in T_BUCKETS:
        m = (t_in_ep >= lo) & (t_in_ep < hi)
        key = f"fit/val_mse_t{lo}_{hi if hi < 10**9 else 'end'}"
        out[key] = float(sq[m].mean()) if m.any() else float("nan")
    # Physics-parameter recovery through the latent: linear probe z_hat -> e.
    e_tr = train.env_params(train.all_indices())
    w = _ridge_fit(pred_tr.astype(np.float64), e_tr.astype(np.float64))
    e_hat = _ridge_predict(w, pred.astype(np.float64))
    r2_e = _r2(e_hat, e.astype(np.float64))
    out["probe/val_r2_mean"] = float(np.nanmean(r2_e))
    for var, v in zip(random_variables, r2_e):
        out[f"probe/val_r2_{var}"] = float(v)
    # Same probe on the *true* z (how much of e is linearly decodable from mu's z at all).
    w_true = _ridge_fit(z_tr.astype(np.float64), e_tr.astype(np.float64))
    r2_e_true = _r2(_ridge_predict(w_true, z.astype(np.float64)), e.astype(np.float64))
    out["probe/val_r2_mean_from_true_z"] = float(np.nanmean(r2_e_true))
    out["fit/val_steps"] = float(holdout.num_steps)
    out["fit/val_episodes"] = float(holdout.num_episodes)
    out["fit/train_steps"] = float(train.num_steps)
    out["fit/train_episodes"] = float(train.num_episodes)
    return out


# -------------------------------------------------------------------- main
def _entrypoint() -> None:
    temp = tyro.cli(Phase2Args)
    if temp.args_file is not None:
        with open(temp.args_file) as f:
            defaults = Phase2Args(**yaml.load(f, Loader=yaml.FullLoader))
    else:
        defaults = Phase2Args()
    args = tyro.cli(Phase2Args, default=defaults)
    if not args.phase1_dir:
        raise ValueError("--phase1-dir is required (RMA phase-1 run / checkpoint dir)")
    torch.set_num_threads(max(1, int(args.torch_num_threads)))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    bundle = load_phase1(args.phase1_dir)
    actor: RMAActor = bundle["actor"]
    normalizer: EnvParamNormalizer = bundle["normalizer"]
    phase1_args = bundle["args"] or {}
    if args.config is not None:
        with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        config = bundle["config"]
    if config is None:
        raise FileNotFoundError("no sim config: pass --config or keep config.yaml next to the phase-1 checkpoint")
    air_hockey_cfg = config["air_hockey"]
    if not air_hockey_cfg.get("domain_random", False):
        print("[rma] WARNING: phase-2 rollouts run in a sim without domain randomization.")
    eval_param_seed = args.eval_param_seed if args.eval_param_seed is not None else phase1_args.get("eval_param_seed", 12345)
    eval_n_envs = args.eval_n_envs if args.eval_n_envs is not None else int(phase1_args.get("eval_n_envs", 5))
    eval_eps_per_env = args.eval_eps_per_env if args.eval_eps_per_env is not None else int(phase1_args.get("eval_eps_per_env", 4))
    has_ood = args.eval_ood and bool(air_hockey_cfg.get("random_variable_ranges_OOD"))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_parent_dir = args.log_parent_dir or os.path.join(bundle["dir"], f"{args.run_name}_{timestamp}")
    if os.path.exists(log_parent_dir):
        base, i = log_parent_dir, 1
        while os.path.exists(log_parent_dir):
            log_parent_dir = f"{base}r{i}"
            i += 1
    os.makedirs(log_parent_dir, exist_ok=True)
    writer = SummaryWriter(log_parent_dir)
    with open(os.path.join(log_parent_dir, "args.yaml"), "w") as f:
        yaml.dump(vars(args), f)
    with open(os.path.join(log_parent_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    feature_dim = step_feature_dim(actor.obs_dim, actor.act_dim, args.step_features)
    conv_kernels, conv_strides = (8, 5, 5), (4, 1, 1)
    module = AdaptationModule(
        feature_dim=feature_dim,
        history_len=args.history_len,
        latent_dim=actor.latent_dim,
        embed_dim=args.embed_dim,
        conv_channels=args.conv_channels,
        conv_kernels=conv_kernels,
        conv_strides=conv_strides,
    ).to(device)
    module_cpu = AdaptationModule(
        feature_dim=feature_dim, history_len=args.history_len, latent_dim=actor.latent_dim, embed_dim=args.embed_dim,
        conv_channels=args.conv_channels, conv_kernels=conv_kernels, conv_strides=conv_strides,
    )
    module_meta = adaptation_meta(module, args.step_features, args.embed_dim, args.conv_channels, conv_kernels, conv_strides)
    optimizer = torch.optim.Adam(module.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_params = sum(p.numel() for p in module.parameters())
    print(
        f"[rma] phase 2: phase1={bundle['model_path']} history_len={args.history_len} step_features={args.step_features} "
        f"feature_dim={feature_dim} latent_dim={actor.latent_dim} conv_out={module.conv_out_dim} params={n_params} device={device}"
    )

    def sync_cpu_module() -> AdaptationModule:
        module_cpu.load_state_dict({k: v.detach().cpu() for k, v in module.state_dict().items()})
        return module_cpu.eval()

    dataset = AdaptationDataset(args.history_len, feature_dim, actor.latent_dim, actor.env_param_dim, max_steps=args.dataset_max_steps)
    holdout = AdaptationDataset(args.history_len, feature_dim, actor.latent_dim, actor.env_param_dim, max_steps=int(args.dataset_max_steps * args.holdout_episode_fraction) + 1)

    env_cfg = dict(air_hockey_cfg)
    env_cfg["seed"] = int(args.seed)
    env = make_eval_env(env_cfg)      # FlatGoalEnv for goal-conditioned tasks (flat [observation, goal] obs)
    encoder = actor.encoder.eval()

    def adapted_factory(noise: float = 0.0) -> AdaptedRMAAgent:
        return AdaptedRMAAgent(actor, sync_cpu_module(), args.step_features, action_noise=noise, rng=np.random.default_rng(int(rng.integers(1 << 31))))

    def eval_agents(final: bool) -> Dict[str, Any]:
        agents: Dict[str, Any] = {"adapted": lambda: adapted_factory(0.0)}
        if final:
            agents["privileged"] = lambda: PrivilegedRMAAgent(actor)
            agents["nominal"] = lambda: NominalLatentAgent(actor)
            if args.baseline_model_path:
                agents["td3_dr"] = lambda: PlainTD3Agent(
                    args.baseline_model_path, actor.obs_dim, actor.act_dim, actor.use_last_action, args.baseline_hidden_layer_size, args.baseline_num_hidden_layers
                )
        return agents

    def run_eval(tag: str, eps_per_env: int, final: bool, call_index: int) -> Dict[str, Any]:
        save_dir = os.path.join(log_parent_dir, tag)
        agents = eval_agents(final)
        gif_agents = tuple(agents.keys()) if args.gifs else ()
        out = {"id": evaluate_multi_env(air_hockey_cfg, agents, normalizer, eval_param_seed=eval_param_seed, n_envs=eval_n_envs, eps_per_env=eps_per_env,
                                          call_index=call_index, save_dir=save_dir, primary="adapted", gif_agents=gif_agents, json_name="multi_env_eval.json")}
        if has_ood:
            out["ood"] = evaluate_multi_env(air_hockey_cfg, agents, normalizer, eval_param_seed=eval_param_seed, n_envs=args.eval_ood_n_envs, eps_per_env=eps_per_env,
                                            call_index=call_index, save_dir=save_dir, primary="adapted", gif_agents=(), ranges_key="random_variable_ranges_OOD",
                                            json_name="multi_env_eval_ood.json")
        return out

    metrics_path = os.path.join(log_parent_dir, "phase2_metrics.jsonl")
    history_rows: List[Dict[str, Any]] = []
    t_start = time.time()
    for it in range(int(args.n_iterations)):
        t0 = time.time()
        use_privileged = it < int(args.privileged_warmup_iterations)
        agent: EvalAgent = PrivilegedRMAAgent(actor) if use_privileged else adapted_factory(args.rollout_action_noise)
        collect = collect_steps(env, agent, normalizer, encoder, args.steps_per_iteration, args.step_features, dataset, holdout, args.holdout_episode_fraction, rng)
        train_loss = train_epochs(module, optimizer, dataset, args.epochs_per_iteration, args.batch_size, args.grad_clip, device, rng)
        fit = latent_fit_metrics(module, dataset, holdout, device, normalizer.random_variables)
        row: Dict[str, Any] = {"iteration": it + 1, "rollout_policy": "privileged" if use_privileged else "adapted", "train/loss": train_loss, "wall_s": time.time() - t0}
        row.update(collect)
        row.update(fit)
        do_eval = (it + 1) % max(1, int(args.eval_every_iterations)) == 0 and (it + 1) < int(args.n_iterations)
        if do_eval:
            ev = run_eval(f"eval_iter{it + 1:03d}", eval_eps_per_env, final=False, call_index=it + 1)
            row["eval/id_adapted_return"] = ev["id"]["aggregate"]["mean_return_across_envs"]
            row["eval/id_adapted_latent_mse"] = ev["id"]["aggregate"].get("latent_mse", float("nan"))
            if "ood" in ev:
                row["eval/ood_adapted_return"] = ev["ood"]["aggregate"]["mean_return_across_envs"]
        for k, v in row.items():
            if isinstance(v, (int, float)) and k not in ("iteration",):
                writer.add_scalar(k, v, it + 1)
        history_rows.append(row)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"[rma] iter {it + 1}/{args.n_iterations} ({'priv' if use_privileged else 'adapted'} rollouts): "
            f"collect_return={collect['collect/mean_return']:.1f} online_mse={collect['collect/online_latent_mse']:.4f} "
            f"train_loss={train_loss:.4f} val_mse={fit.get('fit/val_mse', float('nan')):.4f} val_r2={fit.get('fit/val_r2_mean', float('nan')):.3f} "
            f"probe_r2={fit.get('probe/val_r2_mean', float('nan')):.3f} data={dataset.num_steps}+{holdout.num_steps} steps "
            f"[{time.time() - t0:.0f}s]",
            flush=True,
        )
        save_adaptation_module(log_parent_dir, sync_cpu_module(), module_meta)
    env.close()

    # ---------------------------------------------------------------- final
    final_module_path = save_adaptation_module(log_parent_dir, sync_cpu_module(), module_meta)
    final_eval = run_eval("eval_final", args.final_eval_eps_per_env, final=True, call_index=int(args.n_iterations) + 1)
    summary: Dict[str, Any] = {
        "phase1": bundle["model_path"],
        "adaptation_module": final_module_path,
        "n_iterations": int(args.n_iterations),
        "total_env_steps": int(sum(r["collect/steps"] for r in history_rows)),
        "wall_s": time.time() - t_start,
        "latent_fit_final": {k: v for k, v in history_rows[-1].items() if k.startswith("fit/") or k.startswith("probe/") or k.startswith("collect/")},
        "latent_fit_history": [{k: r.get(k) for k in ("iteration", "fit/val_mse", "fit/val_r2_mean", "probe/val_r2_mean", "collect/online_latent_mse", "collect/mean_return")} for r in history_rows],
        "policy_eval": {},
    }
    for split, res in final_eval.items():
        summary["policy_eval"][split] = {
            name: {
                "mean_return": r["aggregate"]["mean_return_across_envs"],
                "return_sem": r["aggregate"]["return_sem_across_episodes"],
                "mean_success": r["aggregate"]["mean_success_across_envs"],
                "mean_ep_length": r["aggregate"]["mean_ep_length_across_envs"],
                "per_env_mean_return": r["aggregate"]["per_env_mean_return"],
                **({"latent_mse": r["aggregate"]["latent_mse"]} if "latent_mse" in r["aggregate"] else {}),
            }
            for name, r in res["agents"].items()
        }
    with open(os.path.join(log_parent_dir, "phase2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[rma] phase 2 done. Policy eval (mean return across envs):")
    for split, agents in summary["policy_eval"].items():
        print(f"  {split}: " + ", ".join(f"{n}={v['mean_return']:.1f}±{v['return_sem']:.1f}" for n, v in agents.items()))
    writer.close()


if __name__ == "__main__":
    _entrypoint()
