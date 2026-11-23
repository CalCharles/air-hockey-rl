"""Online, reward-free Metamotivo FB on the AirHockey simulator.

Run from the repository root with python -m scripts.fb.fb_training.
Rewards are stored only for task inference/evaluation, never passed to FB losses.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time
from typing import Literal

import numpy as np
import torch
import tyro
import yaml
from metamotivo.buffers.buffers import DictBuffer
from metamotivo.fb import FBAgent, FBAgentConfig
from metamotivo.nn_models import eval_mode

from airhockey import AirHockeyEnv
from scripts.rma.action_latency import install_action_latency

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Args:
    config: str = "configs/new_juggle/sysid_v2_hist4_realplant.yaml"
    output_dir: str = "runs/fb/realplant"
    seed: int = 0
    device: str = "cuda"
    total_timesteps: int = 1_000_000
    learning_starts: int = 5_000
    buffer_size: int = 1_000_000
    batch_size: int = 256
    updates_per_step: int = 1
    backward_input: Literal["full", "current"] = "full"
    z_dim: int = 50
    hidden_dim: int = 256
    backward_hidden_dim: int = 256
    learning_rate: float = 1e-4
    discount: float = 0.975
    actor_std: float = 0.2
    norm_obs: bool = True
    train_goal_ratio: float = 0.5
    update_z_every_steps: int = 150
    action_latency_steps: int = 2
    use_last_action: bool = True
    eval_every_steps: int = 25_000
    eval_episodes: int = 10
    inference_samples: int = 300_000
    # Table-frame metres; a position target (the action space has no orientation).
    paddle_goal_x: float = 0.70
    paddle_goal_y: float = 0.0
    paddle_goal_radius: float = 0.05
    checkpoint_every_steps: int = 100_000
    log_every_steps: int = 1_000
    torch_threads: int = 1
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str = "air-hockey-fb"
    wandb_entity: str | None = None
    wandb_name: str | None = None
    gif_episodes: int = 0

    def validate(self):
        if self.backward_input not in ("full", "current"):
            raise ValueError("backward_input must be full or current")
        for key in ("total_timesteps", "buffer_size", "updates_per_step", "z_dim", "hidden_dim",
                    "backward_hidden_dim", "update_z_every_steps", "eval_episodes",
                    "inference_samples", "log_every_steps", "torch_threads"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.batch_size < 2 or self.buffer_size < self.batch_size:
            raise ValueError("FB needs batch_size >= 2 and buffer_size >= batch_size")
        if min(self.learning_starts, self.action_latency_steps, self.eval_every_steps,
               self.checkpoint_every_steps, self.gif_episodes) < 0:
            raise ValueError("Step counts cannot be negative")
        if not 0 < self.discount < 1 or not 0 <= self.train_goal_ratio <= 1:
            raise ValueError("Require 0 < discount < 1 and 0 <= train_goal_ratio <= 1")
        if self.paddle_goal_radius <= 0:
            raise ValueError("paddle_goal_radius must be positive")
        if self.learning_rate <= 0 or self.actor_std < 0:
            raise ValueError("Require positive learning_rate and nonnegative actor_std")


def load_sim_config(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with path.open() as f:
        config = yaml.safe_load(f)
    if config["air_hockey"].get("simulator") != "box2d":
        raise ValueError("This entrypoint supports only the Box2D simulator")
    return config


def make_env(config, args, seed):
    config = copy.deepcopy(config["air_hockey"])
    config["seed"] = seed
    config.setdefault("simulator_params", {})["seed"] = seed
    env = AirHockeyEnv(config)
    if not (np.all(env.action_space.low == -1) and np.all(env.action_space.high == 1)):
        raise ValueError("FB expects normalized [-1, 1] actions")
    env.action_space.seed(seed)
    return install_action_latency(env, args.action_latency_steps)


def policy_obs(obs, last_action, use_last_action=True):
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    return np.concatenate((obs, last_action)).astype(np.float32) if use_last_action else obs.copy()


def transition(obs, action, next_obs, reward, terminated, truncated, info, paddle_position=None):
    # This env also labels physical failures as truncations. Only a pure time
    # limit bootstraps; store the final observation before calling reset().
    timeout = info.get("truncation_reasons") == ["max_timesteps_exceeded"]
    terminal = bool(terminated or (truncated and not timeout))
    return {"observation": obs[None], "action": action[None],
            "next": {"observation": next_obs[None],
                     "terminated": np.array([[terminal]], dtype=bool),
                     "reward": np.array([[reward]], dtype=np.float32),
                     "contact": np.array([[info.get("paddle_puck_collision_count", 0) > 0]], dtype=np.float32),
                     "paddle_position": np.asarray(paddle_position if paddle_position is not None
                                                   else next_obs[12:14], dtype=np.float32)[None]}}


def create_agent(args, obs_dim, action_dim):
    from scripts.fb.models import AgentConfig, AirHockeyFBAgent
    cfg = AgentConfig() if args.backward_input == "current" else FBAgentConfig()
    if args.backward_input == "current":
        cfg.model.backward_input = "current"
    cfg.model.obs_dim, cfg.model.action_dim = obs_dim, action_dim
    cfg.model.device = args.device
    cfg.model.norm_obs = args.norm_obs
    cfg.model.actor_std = args.actor_std
    cfg.model.archi.z_dim = args.z_dim
    cfg.model.archi.f.hidden_dim = args.hidden_dim
    cfg.model.archi.actor.hidden_dim = args.hidden_dim
    cfg.model.archi.b.hidden_dim = args.backward_hidden_dim
    cfg.train.batch_size = args.batch_size
    cfg.train.discount = args.discount
    cfg.train.train_goal_ratio = args.train_goal_ratio
    cfg.train.lr_f = cfg.train.lr_b = cfg.train.lr_actor = args.learning_rate
    cls = AirHockeyFBAgent if args.backward_input == "current" else FBAgent
    return cls(**asdict(cfg))


@contextmanager
def isolated_rng(seed):
    """Evaluation must not change collection, replay sampling, or update RNGs."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng():
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def task_rewards(batch, task, args):
    if task == "juggle":
        return batch["next"]["reward"]
    if task == "touch":
        return batch["next"]["contact"]
    if task == "reach":
        pos = batch["next"]["paddle_position"]
        goal = pos.new_tensor([args.paddle_goal_x, args.paddle_goal_y])
        # Smooth positional reward supports inference even before a sparse hit.
        return torch.exp(-0.5 * ((pos - goal).norm(dim=-1, keepdim=True) / args.paddle_goal_radius)**2)
    raise ValueError(f"Unknown task: {task}")


@torch.no_grad()
def infer_task_z(model, batch, task, args):
    z = model.reward_inference(batch["next"]["observation"], task_rewards(batch, task, args))
    if not torch.isfinite(z).all():
        raise FloatingPointError("Non-finite reward-inferred latent")
    return z


@torch.no_grad()
def evaluate_tasks(model, batch, sim_config, args, gif_dir=None, step=0):
    result = {}
    for task in ("juggle", "touch", "reach"):
        z = infer_task_z(model, batch, task, args)
        values = evaluate(model, z, sim_config, args, task=task, gif_dir=gif_dir, step=step)
        values["eval/reward_support"] = float((task_rewards(batch, task, args) > 0).float().mean())
        prefix = "eval/" if task == "juggle" else f"eval_{task}/"
        result.update({k.replace("eval/", prefix): v for k, v in values.items()})
    if gif_dir is not None:
        from scripts.fb.monitoring import update_gallery
        update_gallery(Path(gif_dir).parent)
    return result


@torch.no_grad()
def evaluate(model, z, sim_config, args, task="juggle", gif_dir=None, step=0):
    returns, lengths, successes, distances = [], [], [], []
    with isolated_rng(args.seed + 100_000), eval_mode(model):
        env = make_env(sim_config, args, args.seed + 100_000)
        try:
            for ep in range(args.eval_episodes):
                obs, _ = env.reset(seed=args.seed + 100_000 + ep)
                last_action = np.zeros(env.action_space.shape, np.float32)
                total, length, success = 0., 0, False
                distance = 0.
                video = None
                if gif_dir is not None and ep < args.gif_episodes:
                    from scripts.fb.monitoring import EpisodeVideo
                    video = EpisodeVideo(env, task, args, step, ep)
                    video.capture(0., 0.)
                while True:
                    state = policy_obs(obs, last_action, args.use_last_action)
                    action = model.act(torch.as_tensor(state[None], device=args.device), z, mean=True)
                    action = action.cpu().numpy()[0]
                    obs, reward, terminated, truncated, info = env.step(action)
                    last_action = action
                    pos = np.asarray(env.current_state["paddles"]["paddle_ego"]["position"])
                    distance = float(np.linalg.norm(pos - [args.paddle_goal_x, args.paddle_goal_y]))
                    if task == "touch":
                        reward = float(info.get("paddle_puck_collision_count", 0) > 0)
                        success |= reward > 0
                    elif task == "reach":
                        reward = float(np.exp(-0.5 * (distance / args.paddle_goal_radius)**2))
                        success |= distance <= args.paddle_goal_radius
                    else:
                        success |= bool(info.get("success", False))
                    total += float(reward)
                    length += 1
                    if video is not None:
                        video.capture(float(reward), total)
                    if terminated or truncated:
                        break
                if video is not None:
                    video.save(gif_dir)
                returns.append(total)
                lengths.append(length)
                successes.append(success)
                distances.append(distance)
        finally:
            env.close()
    return {"eval/return_mean": float(np.mean(returns)), "eval/return_std": float(np.std(returns)),
            "eval/length_mean": float(np.mean(lengths)), "eval/z_norm": float(z.norm()),
            "eval/success_rate": float(np.mean(successes)),
            "eval/final_goal_distance": float(np.mean(distances))}


def save_checkpoint(agent, replay, args, sim_config, folder, step, updates):
    folder.mkdir(parents=True, exist_ok=False)
    agent.save(str(folder))
    # Agent checkpoints include target networks; FBModel.load expects an
    # inference-only state dict, so export a separate policy as well.
    with isolated_rng(args.seed):
        model = type(agent._model)(**asdict(agent._model.cfg))
    model.load_state_dict({k: v for k, v in agent._model.state_dict().items()
                           if not k.startswith("_target_")})
    model.save(str(folder / "policy"))
    with isolated_rng(args.seed + 200_000):
        batch = replay.sample(min(len(replay), args.inference_samples))
        z = infer_task_z(agent._model, batch, "juggle", args)
    torch.save(z.cpu(), folder / "reward_z.pt")
    torch.save({"next": {k: v.cpu() for k, v in batch["next"].items()}}, folder / "inference_samples.pt")
    (folder / "run.json").write_text(json.dumps({"args": asdict(args), "sim_config": sim_config,
                                                "step": step, "updates": updates}, indent=2))
    return batch


def train(args):
    args.validate()
    torch.set_num_threads(args.torch_threads)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device) if ":" in args.device else 0)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    sim_config = load_sim_config(args.config)
    if args.backward_input == "current" and sim_config["air_hockey"].get("obs_type") != "history":
        raise ValueError("Current backward inputs require obs_type: history")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "args.yaml").exists():
        raise FileExistsError(f"Run already exists: {output}; choose a new --output-dir")
    (output / "args.yaml").write_text(yaml.safe_dump(asdict(args)))
    (output / "sim_config.yaml").write_text(yaml.safe_dump(sim_config))
    env = make_env(sim_config, args, args.seed)
    writer, tracker, success = None, None, False
    try:
        if args.wandb:
            from scripts.fb.monitoring import WandbLogger
            tracker = WandbLogger(args, sim_config, output)
        if args.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(str(output / "tb"))
        obs, _ = env.reset(seed=args.seed)
        last_action = np.zeros(env.action_space.shape, np.float32)
        state = policy_obs(obs, last_action, args.use_last_action)
        agent = create_agent(args, state.size, last_action.size)
        replay = DictBuffer(args.buffer_size, device=args.device)
        z, skill_age = None, 0
        updates, episode_length, episode_return = 0, 0, 0.
        start = time.monotonic()
        metrics = {}
        with (output / "metrics.jsonl").open("w") as log:
            def emit(values, step):
                row = {"step": step, "updates": updates, **values}
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                if writer:
                    for k, v in values.items():
                        writer.add_scalar(k, v, step)
                if tracker:
                    tracker.log(values, step)
                print(json.dumps(row), flush=True)

            def run_evaluation(batch, step):
                gif_dir = output / "gifs" / f"step_{step:09d}" if args.gif_episodes else None
                values = evaluate_tasks(agent._model, batch, sim_config, args, gif_dir, step)
                emit(values, step)
                if tracker and gif_dir is not None:
                    tracker.log_gifs(gif_dir, step)

            for step in range(1, args.total_timesteps + 1):
                if step <= args.learning_starts:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad(), eval_mode(agent._model):
                        if z is None or skill_age >= args.update_z_every_steps:
                            z = agent._model.sample_z(1, device=args.device)
                            skill_age = 0
                        action = agent.act(torch.as_tensor(state[None], device=args.device), z, mean=False)
                        action = action.cpu().numpy()[0]
                    skill_age += 1
                next_obs, reward, terminated, truncated, info = env.step(action)
                next_state = policy_obs(next_obs, action, args.use_last_action)
                replay.extend(transition(state, action, next_state, reward, terminated, truncated, info,
                                         env.current_state["paddles"]["paddle_ego"]["position"]))
                state = next_state
                episode_return += float(reward)
                episode_length += 1
                if terminated or truncated:
                    emit({"rollout/return": episode_return, "rollout/length": episode_length}, step)
                    obs, _ = env.reset()
                    state = policy_obs(obs, np.zeros_like(action), args.use_last_action)
                    episode_return, episode_length, z, skill_age = 0., 0, None, 0
                if step >= args.learning_starts and len(replay) >= args.batch_size:
                    for _ in range(args.updates_per_step):
                        metrics = agent.update({"train": replay}, updates)
                        updates += 1
                    if not all(torch.isfinite(v).all() for v in metrics.values()):
                        raise FloatingPointError(f"Non-finite FB metrics at step {step}")
                if step % args.log_every_steps == 0 or step == args.total_timesteps:
                    emit({**{f"train/{k}": float(v.detach().mean()) for k, v in metrics.items()},
                          "replay/size": len(replay), "time/steps_per_second": step / (time.monotonic() - start)}, step)
                if args.eval_every_steps and (step % args.eval_every_steps == 0 or
                                             step == max(args.learning_starts, args.batch_size)):
                    with isolated_rng(args.seed + 200_000):
                        batch = replay.sample(min(len(replay), args.inference_samples))
                        run_evaluation(batch, step)
                if args.checkpoint_every_steps and step % args.checkpoint_every_steps == 0:
                    save_checkpoint(agent, replay, args, sim_config, output / f"step_{step:09d}", step, updates)
            batch = save_checkpoint(agent, replay, args, sim_config, output / "final", step, updates)
            run_evaluation(batch, step)
        success = True
        return output / "final"
    finally:
        env.close()
        if writer:
            writer.close()
        if tracker:
            tracker.finish(success)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--args-file")
    parser.add_argument("--eval-checkpoint")
    known, remaining = parser.parse_known_args()
    if known.eval_checkpoint:
        folder = Path(known.eval_checkpoint)
        run = json.loads((folder / "run.json").read_text())
        args = tyro.cli(Args, default=Args(**run["args"]), args=remaining)
        args.validate()
        torch.set_num_threads(args.torch_threads)
        if args.device.startswith("cuda"):
            torch.cuda.set_device(torch.device(args.device) if ":" in args.device else 0)
        from scripts.fb.models import load_policy
        model = load_policy(folder / "policy", device=args.device)
        batch = torch.load(folder / "inference_samples.pt", map_location=args.device, weights_only=True)
        # Evaluation exports go next to the checkpoint, leaving training GIFs intact.
        gif_dir = folder / "evaluation_gifs" / f"step_{run['step']:09d}" if args.gif_episodes else None
        print(json.dumps(evaluate_tasks(model, batch, run["sim_config"], args, gif_dir, run["step"]), indent=2))
    else:
        defaults = Args()
        if known.args_file:
            with open(known.args_file) as f:
                defaults = Args(**yaml.safe_load(f))
        train(tyro.cli(Args, default=defaults, args=remaining))


if __name__ == "__main__":
    main()
