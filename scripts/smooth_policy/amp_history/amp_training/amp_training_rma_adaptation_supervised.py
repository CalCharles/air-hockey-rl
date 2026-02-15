import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import tyro
import yaml
from torch.utils.tensorboard import SummaryWriter

from airhockey import AirHockeyEnv
from scripts.smooth_policy.agent import Agent
from scripts.smooth_policy.amp_history.amp_training.rma_adaptation import RMAAdaptationModule
from scripts.smooth_policy.encoder import EnvEncoder


def augment_policy_observation(observation, last_action, use_last_action):
    if not use_last_action:
        return observation
    return torch.cat([observation, last_action], dim=-1)


def concat_env_latent_to_policy_obs(policy_obs_base, env_latent):
    return torch.cat([policy_obs_base, env_latent], dim=-1)


def inject_latent_noise(env_latent, noise_std):
    return env_latent + torch.randn_like(env_latent) * noise_std


def get_env_spec_ranges():
    return {
        "paddle_density": (2500 * 0.8, 2500 * 1.2),
        "paddle_damping": (3 * 0.8, 3 * 1.2),
        "puck_density": (250 * 0.8, 250 * 1.2),
        "puck_damping": (0.5 * 0.8, 0.5 * 1.2),
        "force_scaling": (1 * 0.8, 1 * 1.2),
    }


def extract_env_var_vector_from_spec(spec, env_var_dim):
    ranges = get_env_spec_ranges()
    ordered_keys = [
        "paddle_density",
        "paddle_damping",
        "puck_density",
        "puck_damping",
        "force_scaling",
    ]
    normalized = []
    for key in ordered_keys:
        value = float(spec[key])
        low, high = ranges[key]
        mean = 0.5 * (low + high)
        std = (high - low) / np.sqrt(12.0)
        if std <= 1e-8:
            std = 1.0
        normalized.append((value - mean) / std)
    vec = np.zeros(env_var_dim, dtype=np.float32)
    base = np.asarray(normalized, dtype=np.float32)
    copy_len = min(env_var_dim, len(base))
    vec[:copy_len] = base[:copy_len]
    return vec


def parse_env_vars_from_infos(infos, num_envs, env_var_dim, device, fallback_env_vars):
    if not (isinstance(infos, dict) and "rma_env_vars" in infos):
        return fallback_env_vars
    raw = infos["rma_env_vars"]
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        raw = np.stack([np.asarray(x, dtype=np.float32).reshape(-1) for x in raw], axis=0)
    else:
        raw = np.asarray(raw, dtype=np.float32).reshape(-1, env_var_dim)
    if raw.shape[0] == num_envs:
        return torch.as_tensor(raw, dtype=torch.float32, device=device)
    mask = infos.get("_rma_env_vars")
    if mask is not None and raw.shape[0] == int(np.asarray(mask, dtype=bool).sum()):
        out = fallback_env_vars.clone()
        out[torch.as_tensor(np.asarray(mask, dtype=bool), dtype=torch.bool, device=device)] = torch.as_tensor(
            raw, dtype=torch.float32, device=device
        )
        return out
    return fallback_env_vars


class ResetSampledEnvWrapper(gym.Wrapper):
    def __init__(self, env, env_spec_pool, env_var_dim, rng_seed):
        super().__init__(env)
        self.env_spec_pool = env_spec_pool
        self.env_var_dim = env_var_dim
        self.rng = np.random.default_rng(rng_seed)
        self.current_env_var_vec = np.zeros(env_var_dim, dtype=np.float32)
        self.current_env_id = -1

    def _apply_env_spec(self, env_spec):
        self.env.unwrapped.paddle_density = env_spec["paddle_density"]
        self.env.unwrapped.paddle_damping = env_spec["paddle_damping"]
        self.env.unwrapped.puck_density = env_spec["puck_density"]
        self.env.unwrapped.puck_damping = env_spec["puck_damping"]
        self.env.unwrapped.force_scaling = env_spec["force_scaling"]
        self.current_env_var_vec = extract_env_var_vector_from_spec(env_spec, self.env_var_dim)
        self.current_env_id = int(env_spec["env_id"])

    def _sample_and_apply_spec(self):
        idx = int(self.rng.integers(0, len(self.env_spec_pool)))
        self._apply_env_spec(self.env_spec_pool[idx])

    def reset(self, **kwargs):
        self._sample_and_apply_spec()
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        info["rma_env_vars"] = self.current_env_var_vec.copy()
        info["rma_env_id"] = self.current_env_id
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["rma_env_vars"] = self.current_env_var_vec.copy()
        info["rma_env_id"] = self.current_env_id
        return obs, reward, terminated, truncated, info


def make_env(env_id, env_spec_pool, env_var_dim, air_hockey_config, seed):
    def _thunk():
        cfg = dict(air_hockey_config)
        cfg["seed"] = random.randint(0, int(1e8))
        env = AirHockeyEnv(cfg)
        return ResetSampledEnvWrapper(
            env=env,
            env_spec_pool=env_spec_pool,
            env_var_dim=env_var_dim,
            rng_seed=seed + env_id * 131,
        )

    return _thunk


def load_env_spec_pool(env_spec_pool_path):
    if not os.path.exists(env_spec_pool_path):
        raise FileNotFoundError(
            f"env_spec_pool_path '{env_spec_pool_path}' does not exist. "
            "Use stage-1 artifacts that contain env_spec_pool.yaml or env_spec_pool.pt."
        )
    if env_spec_pool_path.endswith(".pt"):
        pool = torch.load(env_spec_pool_path, map_location="cpu")
    else:
        with open(env_spec_pool_path, "r") as f:
            pool = yaml.load(f, Loader=yaml.FullLoader)
    if not isinstance(pool, list) or len(pool) == 0:
        raise ValueError("Loaded env_spec_pool is empty or invalid.")
    return pool


def split_env_pool(env_spec_pool, train_env_count, eval_env_count, seed):
    if train_env_count > len(env_spec_pool):
        raise ValueError(f"train_env_count={train_env_count} exceeds pool size={len(env_spec_pool)}.")
    rng = np.random.default_rng(seed)
    idxs = rng.permutation(len(env_spec_pool))
    train_idxs = idxs[:train_env_count]
    train_specs = [env_spec_pool[int(i)] for i in train_idxs]
    if eval_env_count <= 0:
        return train_specs, train_specs
    eval_idxs = idxs[: min(eval_env_count, len(env_spec_pool))]
    eval_specs = [env_spec_pool[int(i)] for i in eval_idxs]
    return train_specs, eval_specs


def compute_context_lengths(done_flags):
    # done_flags: [T, N] where done[t, n] indicates transition t ended episode.
    t_steps, n_envs = done_flags.shape
    context = torch.zeros_like(done_flags, dtype=torch.long)
    for t in range(t_steps):
        if t == 0:
            context[t] = 1
        else:
            context[t] = torch.where(done_flags[t - 1], torch.ones(n_envs, dtype=torch.long, device=done_flags.device), context[t - 1] + 1)
    return context


def build_window(states, actions, start_idx, end_idx, env_idx, min_context_len):
    # Inclusive [start_idx, end_idx]
    state_seq = states[start_idx : end_idx + 1, env_idx]
    action_seq = actions[start_idx : end_idx + 1, env_idx]
    curr_len = state_seq.shape[0]
    if curr_len < min_context_len:
        pad = min_context_len - curr_len
        state_pad = state_seq[:1].repeat(pad, 1)
        action_pad = action_seq[:1].repeat(pad, 1)
        state_seq = torch.cat([state_pad, state_seq], dim=0)
        action_seq = torch.cat([action_pad, action_seq], dim=0)
    return state_seq, action_seq


def collect_rollout_dataset(envs, agent, env_encoder, args, device):
    num_envs = args.num_envs
    rollout_len = args.rollout_len
    obs_shape = envs.single_observation_space.shape
    act_shape = envs.single_action_space.shape
    action_dim = int(np.prod(act_shape))

    states = torch.zeros((rollout_len, num_envs) + obs_shape, device=device)
    actions = torch.zeros((rollout_len, num_envs) + act_shape, device=device)
    env_vars = torch.zeros((rollout_len, num_envs, args.env_var_dim), device=device)
    done_flags = torch.zeros((rollout_len, num_envs), dtype=torch.bool, device=device)

    next_obs, infos = envs.reset(seed=args.seed + args.rollout_seed_offset)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    current_env_vars = parse_env_vars_from_infos(
        infos=infos,
        num_envs=num_envs,
        env_var_dim=args.env_var_dim,
        device=device,
        fallback_env_vars=torch.zeros((num_envs, args.env_var_dim), dtype=torch.float32, device=device),
    )
    last_action = torch.zeros((num_envs, action_dim), dtype=torch.float32, device=device)

    with torch.no_grad():
        for t in range(rollout_len):
            states[t] = next_obs
            env_vars[t] = current_env_vars
            policy_obs_base = augment_policy_observation(next_obs, last_action, args.use_last_action_in_policy_state)
            latent_clean = env_encoder(current_env_vars)
            latent_noisy = inject_latent_noise(latent_clean, args.latent_noise_std)
            policy_obs = concat_env_latent_to_policy_obs(policy_obs_base, latent_noisy)
            action, _, _, _ = agent.get_action_and_value(policy_obs)
            actions[t] = action

            next_obs_np, _, terminations, truncations, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminations, truncations)
            done_t = torch.as_tensor(done, dtype=torch.bool, device=device)
            done_flags[t] = done_t
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            current_env_vars = parse_env_vars_from_infos(
                infos=infos,
                num_envs=num_envs,
                env_var_dim=args.env_var_dim,
                device=device,
                fallback_env_vars=current_env_vars,
            )
            last_action = action.detach()
            last_action[done_t] = 0.0

    with torch.no_grad():
        target_latents = env_encoder(env_vars.reshape(-1, args.env_var_dim)).reshape(
            rollout_len, num_envs, args.env_latent_dim
        )

    return {
        "states": states,
        "actions": actions,
        "env_vars": env_vars,
        "done_flags": done_flags,
        "target_latents": target_latents,
        "context_lengths": compute_context_lengths(done_flags),
    }


def sample_batch_from_rollout(rollout, batch_size, args, mode):
    states = rollout["states"]
    actions = rollout["actions"]
    targets = rollout["target_latents"]
    context_lengths = rollout["context_lengths"]

    t_steps, n_envs = context_lengths.shape
    min_k = args.prior_min
    max_k = args.prior_max
    min_model_context = max(args.min_model_context_len, 16)

    sampled_states = []
    sampled_actions = []
    sampled_targets = []
    sampled_k = []
    sampled_available = []

    max_tries = batch_size * 30
    tries = 0
    while len(sampled_states) < batch_size and tries < max_tries:
        tries += 1
        t = random.randint(0, t_steps - 1)
        e = random.randint(0, n_envs - 1)
        available = int(context_lengths[t, e].item())
        if mode == "train" and available < min_k:
            continue
        if mode == "train":
            local_max = min(max_k, available)
            k = random.randint(min_k, local_max)
        elif mode == "eval_max":
            k = min(max_k, available)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        start = t - k + 1
        state_seq, action_seq = build_window(
            states=states,
            actions=actions,
            start_idx=start,
            end_idx=t,
            env_idx=e,
            min_context_len=min_model_context,
        )
        sampled_states.append(state_seq)
        sampled_actions.append(action_seq)
        sampled_targets.append(targets[t, e])
        sampled_k.append(k)
        sampled_available.append(available)

    if len(sampled_states) == 0:
        raise RuntimeError(
            "No valid training samples found in rollout. "
            "Increase rollout_len or reduce prior_min."
        )

    return {
        "states": torch.stack(sampled_states, dim=0),
        "actions": torch.stack(sampled_actions, dim=0),
        "targets": torch.stack(sampled_targets, dim=0),
        "k": torch.tensor(sampled_k, dtype=torch.float32, device=targets.device),
        "available": torch.tensor(sampled_available, dtype=torch.float32, device=targets.device),
    }


def latent_metrics(pred, target):
    mse = torch.mean((pred - target) ** 2)
    mae = torch.mean(torch.abs(pred - target))
    rmse = torch.sqrt(mse + 1e-12)
    cos = nn.functional.cosine_similarity(pred, target, dim=-1).mean()
    pred_norm = pred.norm(dim=-1).mean()
    target_norm = target.norm(dim=-1).mean()
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "cosine": cos,
        "pred_norm": pred_norm,
        "target_norm": target_norm,
    }


def tensor_stats(tensor):
    return tensor.mean(), tensor.std(unbiased=False), tensor.min(), tensor.max()


def param_and_grad_norm(module):
    p_norm_sq = 0.0
    g_norm_sq = 0.0
    for p in module.parameters():
        p_norm_sq += float(torch.sum(p.detach() ** 2).item())
        if p.grad is not None:
            g_norm_sq += float(torch.sum(p.grad.detach() ** 2).item())
    return p_norm_sq ** 0.5, g_norm_sq ** 0.5


@dataclass
class Args:
    # Paths and configs.
    config: str = "scripts/smooth_policy/amp_history/configs/new_juggle/pid_noise_config.yaml"
    args_file: str | None = None
    model_path: str = ""
    encoder_path: str = ""
    env_spec_pool_path: str = ""
    log_parent_dir: str | None = None
    run_name: str = "rma_adaptation_supervised"

    # Runtime.
    seed: int = 0
    device: str = "cuda:0"
    num_envs: int = 16

    # Stage-1 architecture compatibility.
    use_last_action_in_policy_state: bool = True
    action_scale: float = 1.0
    agent_hidden_size: int = 512
    env_var_dim: int = 8
    env_latent_dim: int = 8
    env_encoder_hidden_size: int = 64

    # Stage-2 adaptation model sizes.
    adaptation_embed_dim: int = 16
    adaptation_conv_in_channels: int = 8
    adaptation_hidden_size: int = 64

    # Data collection and supervision.
    train_env_count: int = 500
    eval_env_count: int = 100
    rollout_len: int = 200
    prior_min: int = 10
    prior_max: int = 100
    min_model_context_len: int = 16
    latent_noise_std: float = 0.10
    rollout_seed_offset: int = 1000

    # Optimization.
    num_iterations: int = 2000
    train_steps_per_iter: int = 16
    minibatch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    max_grad_norm: float = 1.0

    # Logging/checkpointing.
    eval_interval: int = 10
    checkpoint_interval: int = 50
    print_interval: int = 1


if __name__ == "__main__":
    temp_args = tyro.cli(Args)
    if temp_args.args_file is not None:
        with open(temp_args.args_file, "r") as f:
            file_args_dict = yaml.load(f, Loader=yaml.FullLoader)
        default_args = Args(**file_args_dict)
    else:
        default_args = Args()
    args = tyro.cli(Args, default=default_args)

    if not args.model_path:
        raise ValueError("model_path must be provided.")
    if not args.encoder_path:
        raise ValueError("encoder_path must be provided.")
    if not args.env_spec_pool_path:
        raise ValueError("env_spec_pool_path must be provided.")
    if args.prior_min > args.prior_max:
        raise ValueError("prior_min must be <= prior_max.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    full_env_pool = load_env_spec_pool(args.env_spec_pool_path)
    train_env_specs, eval_env_specs = split_env_pool(
        env_spec_pool=full_env_pool,
        train_env_count=args.train_env_count,
        eval_env_count=args.eval_env_count,
        seed=args.seed,
    )

    train_envs = gym.vector.AsyncVectorEnv(
        [
            make_env(
                env_id=i,
                env_spec_pool=train_env_specs,
                env_var_dim=args.env_var_dim,
                air_hockey_config=config["air_hockey"],
                seed=args.seed,
            )
            for i in range(args.num_envs)
        ]
    )
    eval_envs = gym.vector.AsyncVectorEnv(
        [
            make_env(
                env_id=i,
                env_spec_pool=eval_env_specs,
                env_var_dim=args.env_var_dim,
                air_hockey_config=config["air_hockey"],
                seed=args.seed + 991,
            )
            for i in range(args.num_envs)
        ]
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    task_name = config["air_hockey"].get("task", "task")
    if args.log_parent_dir is None:
        log_parent_dir = f"runs/rma_adaptation/{task_name}/{args.run_name}_{timestamp}"
    else:
        log_parent_dir = args.log_parent_dir
    if os.path.exists(log_parent_dir):
        base = log_parent_dir
        i = 1
        while os.path.exists(log_parent_dir):
            log_parent_dir = f"{base}r{i}"
            i += 1
    os.makedirs(log_parent_dir, exist_ok=True)

    writer = SummaryWriter(log_parent_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])),
    )
    with open(f"{log_parent_dir}/config.yaml", "w") as f:
        yaml.dump(config, f)
    with open(f"{log_parent_dir}/args.yaml", "w") as f:
        yaml.dump(vars(args), f)

    base_policy_obs_dim = int(np.prod(train_envs.single_observation_space.shape))
    action_dim = int(np.prod(train_envs.single_action_space.shape))
    if args.use_last_action_in_policy_state:
        base_policy_obs_dim += action_dim
    policy_obs_dim = base_policy_obs_dim + args.env_latent_dim
    policy_env_view = SimpleNamespace(
        single_observation_space=gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(policy_obs_dim,),
            dtype=np.float32,
        ),
        single_action_space=train_envs.single_action_space,
    )

    device = torch.device(args.device)
    agent = Agent(
        policy_env_view,
        action_scale=args.action_scale,
        action_bias=0.0,
        hidden_size=args.agent_hidden_size,
    ).to(device)
    env_encoder = EnvEncoder(
        env_var_dim=args.env_var_dim,
        latent_dim=args.env_latent_dim,
        hidden_size=args.env_encoder_hidden_size,
    ).to(device)
    adaptation_module = RMAAdaptationModule(
        action_dim=action_dim,
        state_dim=int(np.prod(train_envs.single_observation_space.shape)),
        embed_dim=args.adaptation_embed_dim,
        conv_in_channels=args.adaptation_conv_in_channels,
        latent_dim=args.env_latent_dim,
        hidden_size=args.adaptation_hidden_size,
    ).to(device)

    agent.load_state_dict(torch.load(args.model_path, map_location=device))
    env_encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))
    agent.eval()
    env_encoder.eval()
    for p in agent.parameters():
        p.requires_grad = False
    for p in env_encoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(
        adaptation_module.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=1e-6,
    )

    best_eval_mse = float("inf")
    global_step = 0
    start_time = time.time()

    for iteration in range(1, args.num_iterations + 1):
        adaptation_module.train()
        rollout = collect_rollout_dataset(train_envs, agent, env_encoder, args, device)
        train_loss_sum = 0.0
        train_mae_sum = 0.0
        train_rmse_sum = 0.0
        train_cos_sum = 0.0

        for _ in range(args.train_steps_per_iter):
            batch = sample_batch_from_rollout(
                rollout=rollout,
                batch_size=args.minibatch_size,
                args=args,
                mode="train",
            )
            preds = adaptation_module(batch["actions"], batch["states"])
            metrics = latent_metrics(preds, batch["targets"])
            loss = metrics["mse"]

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(adaptation_module.parameters(), args.max_grad_norm).item()
            optimizer.step()
            global_step += 1

            train_loss_sum += metrics["mse"].item()
            train_mae_sum += metrics["mae"].item()
            train_rmse_sum += metrics["rmse"].item()
            train_cos_sum += metrics["cosine"].item()

            writer.add_scalar("train/loss_mse", metrics["mse"].item(), global_step)
            writer.add_scalar("train/loss_mae", metrics["mae"].item(), global_step)
            writer.add_scalar("train/loss_rmse", metrics["rmse"].item(), global_step)
            writer.add_scalar("train/latent_cosine", metrics["cosine"].item(), global_step)
            writer.add_scalar("train/pred_norm", metrics["pred_norm"].item(), global_step)
            writer.add_scalar("train/target_norm", metrics["target_norm"].item(), global_step)
            writer.add_scalar("train/prior_k_mean", batch["k"].mean().item(), global_step)
            writer.add_scalar("train/prior_available_mean", batch["available"].mean().item(), global_step)
            writer.add_scalar("optim/lr", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("optim/grad_norm", grad_norm, global_step)
            writer.add_scalar(
                "debug/nan_count_pred",
                float(torch.isnan(preds).sum().item()),
                global_step,
            )
            writer.add_scalar(
                "debug/inf_count_pred",
                float(torch.isinf(preds).sum().item()),
                global_step,
            )

            with torch.no_grad():
                inter = adaptation_module(batch["actions"], batch["states"], return_intermediates=True)
                emb_stats = tensor_stats(inter["embedded"])
                proj_stats = tensor_stats(inter["projected"])
                pooled_stats = tensor_stats(inter["pooled"])
                writer.add_scalar("cnn/embedded_mean", emb_stats[0].item(), global_step)
                writer.add_scalar("cnn/embedded_std", emb_stats[1].item(), global_step)
                writer.add_scalar("cnn/projected_mean", proj_stats[0].item(), global_step)
                writer.add_scalar("cnn/projected_std", proj_stats[1].item(), global_step)
                writer.add_scalar("cnn/pooled_mean", pooled_stats[0].item(), global_step)
                writer.add_scalar("cnn/pooled_std", pooled_stats[1].item(), global_step)

            for name, layer in adaptation_module.named_modules():
                if isinstance(layer, nn.Conv1d):
                    w = layer.weight.detach()
                    writer.add_scalar(f"cnn/{name}_weight_norm", float(torch.norm(w).item()), global_step)
                    if layer.weight.grad is not None:
                        writer.add_scalar(
                            f"cnn/{name}_grad_norm",
                            float(torch.norm(layer.weight.grad.detach()).item()),
                            global_step,
                        )

            p_norm, g_norm = param_and_grad_norm(adaptation_module)
            writer.add_scalar("optim/param_norm", p_norm, global_step)
            writer.add_scalar("optim/global_grad_norm_sqroot", g_norm, global_step)

        mean_train_mse = train_loss_sum / args.train_steps_per_iter

        if args.eval_interval > 0 and iteration % args.eval_interval == 0:
            adaptation_module.eval()
            with torch.no_grad():
                eval_rollout = collect_rollout_dataset(eval_envs, agent, env_encoder, args, device)
                eval_batch = sample_batch_from_rollout(
                    rollout=eval_rollout,
                    batch_size=args.minibatch_size * 4,
                    args=args,
                    mode="eval_max",
                )
                eval_preds = adaptation_module(eval_batch["actions"], eval_batch["states"])
                eval_metrics = latent_metrics(eval_preds, eval_batch["targets"])
                writer.add_scalar("eval/loss_mse", eval_metrics["mse"].item(), global_step)
                writer.add_scalar("eval/loss_mae", eval_metrics["mae"].item(), global_step)
                writer.add_scalar("eval/loss_rmse", eval_metrics["rmse"].item(), global_step)
                writer.add_scalar("eval/latent_cosine", eval_metrics["cosine"].item(), global_step)
                writer.add_scalar("eval/context_k_mean", eval_batch["k"].mean().item(), global_step)
                writer.add_scalar("eval/context_available_mean", eval_batch["available"].mean().item(), global_step)

                eval_payload = {
                    "iteration": int(iteration),
                    "global_step": int(global_step),
                    "eval_mse": float(eval_metrics["mse"].item()),
                    "eval_mae": float(eval_metrics["mae"].item()),
                    "eval_rmse": float(eval_metrics["rmse"].item()),
                    "eval_cosine": float(eval_metrics["cosine"].item()),
                    "eval_context_mean": float(eval_batch["k"].mean().item()),
                    "eval_context_max": float(eval_batch["k"].max().item()),
                }
                with open(os.path.join(log_parent_dir, f"eval_iter_{iteration}.yaml"), "w") as f:
                    yaml.dump(eval_payload, f, sort_keys=False)
                print(
                    "[eval] "
                    f"iter={iteration} mse={eval_payload['eval_mse']:.6f} "
                    f"mae={eval_payload['eval_mae']:.6f} rmse={eval_payload['eval_rmse']:.6f} "
                    f"cos={eval_payload['eval_cosine']:.6f} context_mean={eval_payload['eval_context_mean']:.2f}"
                )

                if eval_metrics["mse"].item() < best_eval_mse:
                    best_eval_mse = float(eval_metrics["mse"].item())
                    torch.save(adaptation_module.state_dict(), os.path.join(log_parent_dir, "adaptation_module_best.pth"))

        if args.checkpoint_interval > 0 and iteration % args.checkpoint_interval == 0:
            ckpt_dir = os.path.join(log_parent_dir, f"checkpoint_{iteration}")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(adaptation_module.state_dict(), os.path.join(ckpt_dir, "adaptation_module.pth"))

        if args.print_interval > 0 and iteration % args.print_interval == 0:
            sps = int(global_step / max(1e-6, (time.time() - start_time)))
            print(
                f"[train] iter={iteration} mse={mean_train_mse:.6f} "
                f"mae={train_mae_sum / args.train_steps_per_iter:.6f} "
                f"rmse={train_rmse_sum / args.train_steps_per_iter:.6f} "
                f"cos={train_cos_sum / args.train_steps_per_iter:.6f} sps={sps}"
            )
            writer.add_scalar("charts/sps", sps, global_step)

    torch.save(adaptation_module.state_dict(), os.path.join(log_parent_dir, "adaptation_module.pth"))
    writer.close()
    train_envs.close()
    eval_envs.close()
