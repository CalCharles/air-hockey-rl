"""Zero-shot puck-goal grid evaluation; reads frozen checkpoints only."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch
from metamotivo.nn_models import eval_mode

from scripts.fb.fb_training import Args, isolated_rng, make_env, policy_obs
from scripts.fb.models import load_policy


def grid_goals(env, size, radius):
    # Inset goals by puck radius + tolerance, so every goal disk lies on-table.
    margin = env.puck_radius + radius
    xs = np.linspace(env.table_x_top + margin, -margin, size)
    ys = np.linspace(env.table_y_left + margin, env.table_y_right - margin, size)
    return np.array([(x, y) for x in xs for y in ys], dtype=np.float32)


def goal_weights(observations, goal, radius):
    # Saved checkpoints contain observed histories, not true puck positions.
    # Explicitly exclude occlusion placeholders from latent inference.
    distance = torch.linalg.vector_norm(observations[:, 27:29] - observations.new_tensor(goal), dim=1)
    visible = observations[:, 29] < .5
    weights = torch.exp(-.5 * (distance / radius).square()) * visible
    return weights[:, None], (visible & (distance <= radius)).sum().item()


def true_puck_position(env):
    # Public current_state is already noisy/delayed/occluded in this simulator.
    body = next(iter(env.simulator.pucks.values()))
    return np.asarray(env.simulator._box2d_to_base_coords(body.position)).copy()


def post_contact_distances(positions, contacts, goals):
    mask = np.maximum.accumulate(np.asarray(contacts, dtype=bool))
    if not mask.any():
        return np.full(len(goals), np.inf)
    return np.linalg.norm(np.asarray(positions)[mask, None, :] - goals[None, :, :], axis=-1).min(axis=0)


@torch.no_grad()
def evaluate_checkpoint(checkpoint, label, output, size=5, radius=.05, episodes=20, seed=300000):
    run = json.loads((checkpoint / "run.json").read_text())
    args = replace(Args(**run["args"]), device="cpu", tensorboard=False, wandb=False)
    model = load_policy(checkpoint / "policy", device="cpu")
    batch = torch.load(checkpoint / "inference_samples.pt", map_location="cpu", weights_only=True)
    observations = batch["next"]["observation"]
    env = make_env(run["sim_config"], args, seed)
    goals = grid_goals(env, size, radius)
    destination = output / label
    destination.mkdir(parents=True, exist_ok=False)
    matrix = np.zeros((len(goals), len(goals)))
    rows, per_goal = [], []
    gif_ids = {0, size - 1, len(goals)//2, len(goals)-size, len(goals)-1}
    try:
        for goal_id in range(-1, len(goals)):
            if goal_id < 0:
                z = model.reward_inference(observations, batch["next"]["reward"])
                support, ess = None, None
            else:
                weights, support = goal_weights(observations, goals[goal_id], radius)
                ess = float(weights.sum().square() / weights.square().sum().clamp_min(1e-30))
                z = model.reward_inference(observations, weights)
            distances, contact_episodes = [], []
            for ep in range(episodes):
                episode_seed = seed + ep
                with isolated_rng(episode_seed), eval_mode(model):
                    obs, _ = env.reset(seed=episode_seed)
                    last_action = np.zeros(2, dtype=np.float32)
                    positions, contacts = [], []
                    frames, renderer = [], None
                    if goal_id in gif_ids and ep == 0:
                        from airhockey.renderers import AirHockeyRenderer
                        renderer = AirHockeyRenderer(env, orientation="vertical")
                    while True:
                        state = policy_obs(obs, last_action, args.use_last_action)
                        action = model.act(torch.as_tensor(state[None]), z, mean=True).numpy()[0]
                        obs, _, terminated, truncated, info = env.step(action)
                        last_action = action
                        positions.append(true_puck_position(env))
                        contacts.append(info.get("paddle_puck_collision_count", 0) > 0)
                        if renderer is not None:
                            import cv2
                            frame = renderer.get_frame()
                            gx, gy = renderer.world_xy_to_output_pixel(*goals[goal_id])
                            px, py = renderer.world_xy_to_output_pixel(*positions[-1])
                            cv2.circle(frame, (round(px), round(py)), 5, (255, 0, 0), -1)
                            cv2.circle(frame, (round(gx), round(gy)), round(radius * renderer.ppm), (0, 200, 0), 2)
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame = cv2.resize(frame, (160, round(frame.shape[0] * 160 / frame.shape[1])))
                            d = post_contact_distances(positions, contacts, goals[[goal_id]])[0]
                            cv2.rectangle(frame, (0, 0), (160, 42), (255, 255, 255), -1)
                            cv2.putText(frame, f"goal {goal_id} hit {int(d <= radius)}", (3, 13),
                                        cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 0, 0), 1)
                            cv2.putText(frame, f"R {int(any(contacts) and np.linalg.norm(positions[-1]-goals[goal_id]) <= radius)}", (3, 29),
                                        cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 0, 0), 1)
                            frames.append(frame)
                        if terminated or truncated:
                            break
                    d = post_contact_distances(positions, contacts, goals)
                    distances.append(d)
                    contact_episodes.append(any(contacts))
                    row = {"goal_id": goal_id, "seed": episode_seed, "contact": any(contacts),
                           "length": len(positions), "min_distance_after_contact": [float(v) if np.isfinite(v) else None for v in d]}
                    rows.append(row)
                    if frames:
                        import imageio.v2 as imageio
                        imageio.mimsave(destination / f"goal_{goal_id:02d}_seed{episode_seed}.gif",
                                        frames, format="GIF", duration=50, loop=0)
            distance_array = np.asarray(distances)
            hits = (distance_array <= radius).mean(axis=0)
            if goal_id == -1:
                baseline = hits
            else:
                matrix[goal_id] = hits
                valid_distances = distance_array[:, goal_id]
                finite = valid_distances[np.isfinite(valid_distances)]
                per_goal.append({"goal_id": goal_id, "x": float(goals[goal_id, 0]), "y": float(goals[goal_id, 1]),
                    "success_rate": float(hits[goal_id]), "success_rate_10cm": float((valid_distances <= .10).mean()),
                    "contact_rate": float(np.mean(contact_episodes)), "inference_samples_in_radius": support,
                    "inference_effective_samples": ess, "latent_norm": float(z.norm()),
                    "mean_closest_distance_given_contact": float(finite.mean()) if len(finite) else None,
                    "juggle_baseline_success_rate": float(baseline[goal_id])})
                print(json.dumps({"label": label, "step": run["step"], **per_goal[-1]}), flush=True)
        result = {"label": label, "checkpoint": str(checkpoint), "training_steps": run["step"],
                  "backward_input": run["args"].get("backward_input", "full"), "grid_size": size,
                  "radius": radius, "episodes_per_goal": episodes, "seed": seed, "goals": per_goal,
                  "cross_goal_success": matrix.tolist(), "juggle_baseline_success": baseline.tolist(),
                  "mean_goal_success": float(np.diag(matrix).mean()),
                  "mean_juggle_baseline_success": float(baseline.mean()),
                  "mean_success_10cm": float(np.mean([r["success_rate_10cm"] for r in per_goal])),
                  "mean_contact_rate": float(np.mean([r["contact_rate"] for r in per_goal]))}
        (destination / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        (destination / "episodes.jsonl").write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))
        return result
    finally:
        env.close()


def plot_results(results, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(4*n, 7), squeeze=False, constrained_layout=True)
    for col, result in enumerate(results):
        size = result["grid_size"]
        for row, key, title in [(0, "success_rate", "Goal-conditioned"), (1, "juggle_baseline_success_rate", "Juggling latent control")]:
            data = np.array([r[key] for r in result["goals"]]).reshape(size, size)
            ax = axes[row, col]
            img = ax.imshow(data, vmin=0, vmax=1, cmap="viridis", aspect="auto")
            for i in range(size):
                for j in range(size):
                    ax.text(j, i, f"{data[i,j]:.0%}", ha="center", va="center", color="white" if data[i,j]<.5 else "black")
            xs = [result["goals"][i*size]["x"] for i in range(size)]
            ys = [result["goals"][j]["y"] for j in range(size)]
            ax.set_xticks(range(size), [f"{v:.2f}" for v in ys])
            ax.set_yticks(range(size), [f"{v:.2f}" for v in xs])
            ax.set_xlabel("Table y (m)")
            ax.set_ylabel("Table x (m), top edge at top")
            ax.set_title(f"{result['label']} ({result['training_steps']:,} steps)\n{title}")
    fig.colorbar(img, ax=axes.ravel().tolist(), label="Post-contact success rate")
    fig.savefig(output / "success_heatmaps.png", dpi=170)
    plt.close(fig)
    (output / "summary.json").write_text(json.dumps(results, indent=2))
    import html
    cards = ''.join(f'<figure><img src="{html.escape(str(p.relative_to(output)))}"><figcaption>{html.escape(str(p.relative_to(output)))}</figcaption></figure>' for p in output.glob('*/*.gif'))
    (output / "index.html").write_text('<!doctype html><title>FB puck goal grid</title><style>body{font:16px sans-serif}section{display:flex;flex-wrap:wrap}figure{margin:12px}figure img{width:160px}</style><h1>Puck goal grid</h1><p>Green circle: target; blue dot: true puck position. Success requires paddle contact first. Clips use the first paired seed, without selection for success.</p><img style="max-width:100%" src="success_heatmaps.png"><section>'+cards+'</section>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--radius", type=float, default=.05)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=300000)
    args = parser.parse_args()
    if len(args.checkpoints) != len(args.labels) or len(set(args.labels)) != len(args.labels):
        parser.error("Supply one distinct label per checkpoint")
    if args.grid_size < 2 or args.episodes < 1 or not 0 < args.radius <= .1:
        parser.error("Require grid-size >= 2, episodes >= 1 and 0 < radius <= .1")
    if any(Path(label).name != label or label in (".", "..") for label in args.labels):
        parser.error("Labels must be plain directory names")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    results = [evaluate_checkpoint(Path(checkpoint).absolute(), label, output, args.grid_size,
                                   args.radius, args.episodes, args.seed)
               for checkpoint, label in zip(args.checkpoints, args.labels)]
    plot_results(results, output)


if __name__ == "__main__":
    main()
