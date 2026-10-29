"""Paired long-horizon juggling evaluation, including timing and motion metrics.

Run with ``python -m scripts.td3.juggle_survival --run-dir <checkpoint>``.
Evaluation uses only the actor's deployment observations; true simulator state
is used exclusively for reporting motion, never for choosing an action.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter, deque
from pathlib import Path
from types import SimpleNamespace

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import yaml

from airhockey import AirHockeyEnv
from airhockey.renderers import AirHockeyRenderer
from scripts.td3.eval_utils import (
    augment_policy_observation,
    build_policy_env_view,
    load_policy_for_evaluation,
)
from scripts.td3.helper.juggle_counter import count_juggles_from_arrays


def diagnostic_body_positions(simulator):
    """Read physical body positions without sampling sensors or advancing RNG.

    Do not call get_current_state here: it draws observation noise and updates
    occlusion/interpolation state. These values are for saved diagnostics only.
    """
    paddle = simulator.paddles["paddle_ego"].position
    puck = next(iter(simulator.pucks.values())).position
    return (np.asarray([-paddle[1], paddle[0]], dtype=np.float64),
            np.asarray([-puck[1], puck[0]], dtype=np.float64))


def summarize_episodes(rows, horizon):
    """Treat capped episodes as censored, and actual terminations as failures."""
    n = len(rows)
    if n == 0:
        raise ValueError("At least one evaluation episode is required")
    lengths = np.asarray([row["steps"] for row in rows])
    failures = sum(row["failed"] for row in rows)
    summary = {
        "episodes": n,
        "horizon": horizon,
        "failures": failures,
        "horizon_survival_fraction": sum(row["reached_horizon"] for row in rows) / n,
        "sustained_juggling_fraction": sum(
            row["reached_horizon"] and row.get("juggles", row["contacts"]) >= max(3, horizon // 200)
            for row in rows
        ) / n,
        "mean_juggles": float(np.mean([row.get("juggles", 0) for row in rows])),
        "mean_valid_juggling_steps": float(np.mean([
            row["steps"] if row.get("juggles", 0) >= max(2, row["steps"] // 200) else 0
            for row in rows
        ])),
        "restricted_mean_steps": float(lengths.mean()),
        "median_observed_steps": float(np.median(lengths)),
        "min_steps": int(lengths.min()),
        "mean_contacts": float(np.mean([row["contacts"] for row in rows])),
        "mean_return": float(np.mean([row["return"] for row in rows])),
        "end_reasons": dict(Counter(reason for row in rows for reason in row["end_reasons"])),
        # Independent episodes, not independent timesteps. One-sided exact
        # binomial upper bound applies only when no failures were observed.
        "zero_failure_probability_upper_95": 1 - 0.05 ** (1 / n) if failures == 0 else None,
        "survival_at_steps": {
            str(t): sum(row["steps"] >= t and not (
                row["failed"] and row["steps"] == t
            ) for row in rows) / n
            for t in (250, 1000, 2000, 10000) if t <= horizon
        },
        "worst_peak_paddle_speed_m_s": max(row["peak_paddle_speed_m_s"] for row in rows),
        "worst_peak_paddle_acceleration_m_s2": max(
            row["peak_paddle_acceleration_m_s2"] for row in rows
        ),
        "mean_action_delta_rms": float(np.mean([row["action_delta_rms"] for row in rows])),
        "mean_target_delta_rms_m": float(np.mean([row["target_delta_rms_m"] for row in rows])),
    }
    if all("true_state_juggles" in row for row in rows):
        summary["mean_true_state_juggles"] = float(np.mean([row["true_state_juggles"] for row in rows]))
        summary["true_state_sustained_juggling_fraction"] = sum(
            row["reached_horizon"] and row["true_state_juggles"] >= max(3, horizon // 200)
            for row in rows
        ) / n
    return summary


def clean_sim_config(cfg):
    """A nominal-physics benchmark without sensor corruption; retain control limits."""
    clean = copy.deepcopy(cfg)
    clean["domain_random"] = False
    clean["simulator_params"].update(
        puck_noise=False, enable_random_occlusions=False, occlusion_map_path=None,
        enable_observation_delay=False, enable_action_delay=False,
        enable_puck_delay_interpolation=False, puck_obs_lag_seconds=0.,
        puck_obs_lag_frame_period=0.,
    )
    return clean


def evaluation_scenarios(cfg, scenario_set="transfer"):
    if scenario_set == "training":
        return {"training": copy.deepcopy(cfg)}
    if scenario_set == "clean":
        return {"clean": clean_sim_config(cfg)}
    nominal = copy.deepcopy(cfg)
    nominal["domain_random"] = False
    if scenario_set == "nominal":
        return {"nominal": nominal}
    if scenario_set != "transfer":
        raise ValueError(f"Unknown scenario set: {scenario_set}")
    scenarios = {"nominal": nominal, "randomized": copy.deepcopy(cfg)}
    # Fixed camera latency probes expose dependence on fresh, synchronous puck
    # observations. Keep physical parameters nominal for an interpretable probe.
    for lag in (0.05, 0.10):
        timing = copy.deepcopy(nominal)
        timing["simulator_params"].update(
            puck_obs_lag_seconds=lag, puck_obs_lag_frame_period=1 / 30
        )
        scenarios[f"camera_{round(lag * 1000)}ms"] = timing
    return scenarios


@torch.inference_mode()
def evaluate_checkpoint(run_dir, *, horizon=10000, episodes=32, seed_base=10000,
                        gif_steps=400, out_dir=None, config_path=None, scenarios=None,
                        scenario_set="transfer"):
    if horizon < 1 or episodes < 1 or gif_steps < 0:
        raise ValueError("horizon/episodes must be positive and gif_steps nonnegative")
    torch.set_num_threads(1)
    run_dir = Path(run_dir)
    out = Path(out_dir) if out_dir else run_dir / "survival_eval"
    out.mkdir(parents=True, exist_ok=True)
    args = yaml.safe_load((run_dir / "args.yaml").read_text())
    cfg = yaml.safe_load(Path(config_path or run_dir / "config.yaml").read_text())["air_hockey"]
    if cfg.get("simulator") != "box2d":
        raise ValueError("This survival evaluator supports Box2D simulation only")
    history_agent = None
    probe = AirHockeyEnv(copy.deepcopy(cfg))
    try:
        view = build_policy_env_view(SimpleNamespace(
            single_observation_space=probe.observation_space,
            single_action_space=probe.action_space,
        ), bool(args.get("use_last_action_in_policy_state", False)))
        if (run_dir / "rma_meta.json").exists():
            from scripts.rma.bundle import load_phase1
            from scripts.rma.evaluate import HistoryAgent
            bundle = load_phase1(str(run_dir))
            if bundle["mode"] != "history":
                raise ValueError("Survival evaluation requires a deployable history actor, not a privileged actor")
            history_agent = HistoryAgent(bundle["actor"])
            policy = None
        else:
            policy = load_policy_for_evaluation(
                str(run_dir / "model.pth"), view, 1,
                int(args["agent_hidden_layer_size"]), int(args["agent_num_hidden_layers"]),
            )
    finally:
        probe.close()

    (out / "evaluation_config.yaml").write_text(yaml.safe_dump({"air_hockey": cfg}, sort_keys=False))
    (out / "evaluation_args.yaml").write_text(yaml.safe_dump(args, sort_keys=False))
    result = {"run_dir": str(run_dir), "seed_base": seed_base,
              "horizon": horizon, "episodes_per_scenario": episodes, "scenarios": {}}
    for name, scenario in (scenarios or evaluation_scenarios(cfg, scenario_set)).items():
        scenario = copy.deepcopy(scenario)
        if scenario.get("simulator") != "box2d":
            raise ValueError("Evaluation scenarios must use Box2D simulation")
        scenario["max_timesteps"] = horizon + 1
        rows = []
        frames = deque(maxlen=gif_steps)
        initial_frames = []
        for episode in range(episodes):
            seed = seed_base + episode
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            params = copy.deepcopy(scenario)
            params.update(seed=seed, max_timesteps=horizon + 1)
            params["simulator_params"]["seed"] = seed
            env = AirHockeyEnv(params)
            try:
                obs, _ = env.reset(seed=seed)
                if history_agent is not None:
                    history_agent.reset(obs)
                last_action = torch.zeros((1, 2))
                previous_velocity = np.asarray(env.current_state["paddles"]["paddle_ego"]["velocity"])
                renderer = AirHockeyRenderer(env, orientation="vertical") if episode == 0 and gif_steps else None
                ret, contacts, speed_max, accel_max = 0.0, 0, 0.0, 0.0
                delta_sq, target_delta_sq, failed, unexpected_truncation = 0.0, 0.0, False, False
                info = {}
                trajectory = {key: [] for key in ("observation", "next_observation", "action", "reward", "terminated", "truncated",
                                                  "true_paddle_xy", "true_puck_xy")}
                for step in range(1, horizon + 1):
                    previous_obs = np.asarray(obs, dtype=np.float32).copy()
                    state = augment_policy_observation(
                        torch.as_tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0),
                        last_action, bool(args.get("use_last_action_in_policy_state", False)),
                    )
                    action = (
                        history_agent.act(np.asarray(obs), last_action.numpy().reshape(2))
                        if history_agent is not None else policy(state).cpu().numpy().reshape(2)
                    )
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                    delta = action - last_action.numpy().reshape(2)
                    delta_sq += float(np.mean(delta ** 2))
                    target_delta_sq += float(np.mean((delta * [env.action_x_ratio, env.action_y_ratio]) ** 2))
                    obs, reward, term, trunc, info = env.step(action)
                    # Read-only telemetry: never used to construct policy state.
                    true_paddle, true_puck = diagnostic_body_positions(env.simulator)
                    trajectory["true_paddle_xy"].append(true_paddle)
                    trajectory["true_puck_xy"].append(true_puck)
                    trajectory["observation"].append(previous_obs)
                    trajectory["next_observation"].append(np.asarray(obs, dtype=np.float32).copy())
                    trajectory["action"].append(action.copy())
                    trajectory["reward"].append(float(np.asarray(reward).reshape(-1)[0]))
                    trajectory["terminated"].append(bool(term))
                    trajectory["truncated"].append(bool(trunc))
                    ret += float(np.asarray(reward).reshape(-1)[0])
                    contacts += int(info.get("paddle_puck_collision_count", 0))
                    velocity = np.asarray(env.current_state["paddles"]["paddle_ego"]["velocity"])
                    speed_max = max(speed_max, float(np.linalg.norm(velocity)))
                    accel_max = max(accel_max, float(np.linalg.norm(velocity - previous_velocity) / env.simulator.time_per_step))
                    previous_velocity = velocity.copy()
                    last_action = torch.from_numpy(action.copy()).reshape(1, 2)
                    if renderer is not None:
                        renderer.set_action(action)
                        frame = cv2.cvtColor(renderer.get_frame(), cv2.COLOR_BGR2RGB)
                        frame = cv2.resize(frame, (160, round(frame.shape[0] * 160 / frame.shape[1])))
                        for y, label in enumerate((f"Step {step} / {horizon}", f"Reward {float(reward):.2f}",
                                                   f"Return {ret:.1f}", f"Contacts {contacts}")):
                            cv2.putText(frame, label, (3, 14 + y * 14), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.32, (0, 0, 0), 1)
                        frames.append(frame)
                        if len(initial_frames) < min(gif_steps, 200):
                            initial_frames.append(frame)
                    if term or trunc:
                        # Any early end in this extended-budget evaluator is a
                        # failure, including old configs with failure truncations.
                        failed = bool(term or trunc)
                        unexpected_truncation = bool(trunc)
                        break
                observed = np.asarray(trajectory["next_observation"])
                counts = count_juggles_from_arrays(observed[:, 12:14], observed[:, 27:29], observed[:, 29])
                true_counts = count_juggles_from_arrays(
                    np.asarray(trajectory["true_paddle_xy"]), np.asarray(trajectory["true_puck_xy"]), np.zeros(step))
                rows.append({
                    "seed": seed, "steps": step, "return": ret, "contacts": contacts,
                    "failed": failed, "unexpected_truncation": unexpected_truncation,
                    "reached_horizon": step == horizon and not (failed or unexpected_truncation),
                    "end_reasons": info.get("episode_end_reasons", []) if failed or unexpected_truncation else ["evaluation_horizon"],
                    "peak_paddle_speed_m_s": speed_max,
                    "peak_paddle_acceleration_m_s2": accel_max,
                    "action_delta_rms": float(np.sqrt(delta_sq / step)),
                    "target_delta_rms_m": float(np.sqrt(target_delta_sq / step)),
                    "trajectory": f"{name}_seed{seed}.npz",
                    "contact_events": counts.n_contacts,
                    "juggles": counts.n_juggles,
                    "true_state_contact_events": true_counts.n_contacts,
                    "true_state_juggles": true_counts.n_juggles,
                })
                np.savez_compressed(out / f"{name}_seed{seed}.npz",
                                    **{key: np.asarray(value) for key, value in trajectory.items()})
            finally:
                env.close()
        summary = summarize_episodes(rows, horizon)
        result["scenarios"][name] = {"config": scenario, "summary": summary, "episodes": rows}
        if frames:
            imageio.mimsave(out / f"{name}.gif", list(frames), format="GIF", loop=0, duration=50)
            imageio.mimsave(out / f"{name}_start.gif", initial_frames, format="GIF", loop=0, duration=50)
        (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"{run_dir.name} {name}: mean survival {summary['restricted_mean_steps']:.1f}, "
              f"survived {summary['horizon_survival_fraction']:.1%}, contacts {summary['mean_contacts']:.1f}", flush=True)

    lines = [f"# Juggling survival: {run_dir}", "", f"{episodes} paired episodes/scenario, cap {horizon} steps, seeds {seed_base} onward.",
             "Capped rollouts are censored; they do not demonstrate infinite survival. Motion metrics are simulator diagnostics, not hardware safety certification.", "",
             "| Scenario | Mean observed steps | Survived cap | Mean contacts | True-state juggles | True-state genuine cap | Peak paddle speed (m/s) | Action delta RMS |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, value in result["scenarios"].items():
        s = value["summary"]
        lines.append(f"| {name} | {s['restricted_mean_steps']:.1f} | {s['horizon_survival_fraction']:.1%} | "
                     f"{s['mean_contacts']:.1f} | {s['mean_true_state_juggles']:.2f} | "
                     f"{s['true_state_sustained_juggling_fraction']:.1%} | "
                     f"{s['worst_peak_paddle_speed_m_s']:.3f} | {s['mean_action_delta_rms']:.4f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument("--config")
    parser.add_argument("--horizon", type=int, default=10000)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed-base", type=int, default=10000)
    parser.add_argument("--gif-steps", type=int, default=400)
    parser.add_argument("--scenario-set", choices=["transfer", "clean", "nominal", "training"], default="transfer")
    cli = parser.parse_args()
    evaluate_checkpoint(cli.run_dir, horizon=cli.horizon, episodes=cli.episodes,
                        seed_base=cli.seed_base, gif_steps=cli.gif_steps,
                        out_dir=cli.out_dir, config_path=cli.config, scenario_set=cli.scenario_set)


if __name__ == "__main__":
    main()
