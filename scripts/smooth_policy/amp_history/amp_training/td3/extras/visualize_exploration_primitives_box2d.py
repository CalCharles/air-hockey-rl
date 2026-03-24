"""
Render Box2D GIFs that visualize TD3 primitive exploration behaviors.
"""

import os
from dataclasses import dataclass
from types import SimpleNamespace

import cv2
import gymnasium as gym
import imageio
import numpy as np
import torch
import tyro
import yaml

from airhockey import AirHockeyEnv
from airhockey.renderers import AirHockeyRenderer
from scripts.smooth_policy.deterministic_agent import DeterministicAgent
from scripts.smooth_policy.amp_history.amp_training.td3.helper.exploration_selector import (
    PrimitiveExplorationSelector,
)


def extract_current_paddle_position(observation: torch.Tensor) -> torch.Tensor:
    obs_dim = observation.shape[-1]
    if obs_dim >= 30:
        return observation[:, 12:14]
    return observation[:, 0:2]


def extract_current_puck_position(observation: torch.Tensor) -> torch.Tensor:
    obs_dim = observation.shape[-1]
    if obs_dim >= 30:
        return observation[:, 27:29]
    if obs_dim >= 8:
        return observation[:, 4:6]
    if obs_dim >= 4:
        return observation[:, 2:4]
    raise ValueError(f"Observation dim {obs_dim} is too small to extract puck position.")


def extract_current_puck_velocity(observation: torch.Tensor) -> torch.Tensor:
    obs_dim = observation.shape[-1]
    if obs_dim >= 30:
        # History obs stores 5 puck positions at indices [15:30], so estimate velocity
        # using window displacement (last - first) to reduce one-step noise.
        return observation[:, 27:29] - observation[:, 15:17]
    if obs_dim >= 8 and obs_dim < 30:
        return observation[:, 6:8]
    return torch.zeros((observation.shape[0], 2), dtype=observation.dtype, device=observation.device)


def resolve_path_relative_to_dir(path: str, base_dir: str) -> str:
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base_dir, path))


def augment_policy_observation(
    observation: torch.Tensor,
    last_action: torch.Tensor,
    use_last_action: bool,
) -> torch.Tensor:
    if not use_last_action:
        return observation
    return torch.cat([observation, last_action], dim=-1)


def deterministic_actor_action(actor, policy_obs: torch.Tensor) -> torch.Tensor:
    if hasattr(actor, "get_action_mean_and_logstd"):
        action_mean, _ = actor.get_action_mean_and_logstd(policy_obs)
        return torch.tanh(action_mean) * actor.action_scale + actor.action_bias
    if hasattr(actor, "get_action"):
        return actor.get_action(policy_obs)
    raise TypeError(f"Unsupported actor type for deterministic action: {type(actor)}")


def create_warm_start_policy(
    warm_model_path: str,
    warm_args_path: str,
    warm_config_path: str,
    env_action_space: gym.spaces.Box,
    raw_obs_dim: int,
    act_dim: int,
    device: str,
):
    with open(warm_args_path, "r") as f:
        warm_args = yaml.load(f, Loader=yaml.FullLoader)
    with open(warm_config_path, "r") as f:
        warm_config = yaml.load(f, Loader=yaml.FullLoader)

    warm_use_last_action = bool(warm_args.get("use_last_action_in_policy_state", False))
    warm_hidden_layer_size = int(warm_args.get("agent_hidden_layer_size", 64))
    warm_num_hidden_layers = int(warm_args.get("agent_num_hidden_layers", 2))
    warm_action_scale = float(warm_args.get("action_scale", 0.02))
    if warm_config.get("air_hockey", {}).get("use_pid", False):
        warm_action_scale = 1.0

    warm_policy_obs_dim = raw_obs_dim + act_dim if warm_use_last_action else raw_obs_dim
    warm_policy_env_view = SimpleNamespace(
        single_observation_space=gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(warm_policy_obs_dim,),
            dtype=np.float32,
        ),
        single_action_space=env_action_space,
    )
    warm_policy = DeterministicAgent(
        warm_policy_env_view,
        action_scale=warm_action_scale,
        action_bias=0.0,
        hidden_layer_size=warm_hidden_layer_size,
        num_hidden_layers=warm_num_hidden_layers,
    ).to(device)
    warm_state_dict = torch.load(warm_model_path, map_location=device)
    warm_policy.load_state_dict(warm_state_dict)
    warm_policy.eval()
    return warm_policy, warm_use_last_action


@dataclass
class Args:
    config: str = "scripts/smooth_policy/amp_history/configs/new_juggle/pid_noise_constant_upper_half_custom_sim_params.yaml"
    output_dir: str = "runs/td3/primitive_viz"
    primitive: str = "all"  # all|stand_still|same_direction|y_aligned|policy_takeover|target_position_directional|pre_contact_hit_variant
    gifs_per_primitive: int = 2
    max_timesteps: int = 160
    fps: int = 20
    seed: int = 0
    device: str = "cpu"
    base_action_mode: str = "zero"  # zero|random
    random_action_std: float = 0.5

    exploration_primitive_steps: int = 5
    exploration_direction_y_component_weight: float = 1.5
    exploration_same_direction_min_angle_deg: float = -180.0
    exploration_same_direction_max_angle_deg: float = 180.0
    exploration_same_direction_min_magnitude: float = 0.012
    exploration_same_direction_max_magnitude: float = 0.26
    exploration_y_aligned_min_angle_deg: float = 45.0
    exploration_y_aligned_max_angle_deg: float = 135.0
    exploration_y_aligned_min_magnitude: float = 0.012
    exploration_y_aligned_max_magnitude: float = 0.12
    exploration_target_position_directional_min_angle_deg: float = -180.0
    exploration_target_position_directional_max_angle_deg: float = 180.0
    exploration_target_position_directional_min_magnitude: float = 0.2
    exploration_target_position_directional_max_magnitude: float = 0.5

    exploration_target_position_min_distance: float = 0.2
    exploration_target_position_max_distance: float = 0.5
    exploration_target_position_delta_x: float = 0.26
    exploration_target_position_delta_y: float = 0.12
    exploration_target_position_steps: int = 5
    exploration_pre_contact_hit_variant_chance: float = 0.15
    exploration_pre_contact_hit_variant_steps: int = 5
    exploration_pre_contact_hit_variant_distance_threshold: float = 0.2
    exploration_pre_contact_hit_variant_scale_min: float = 0.5
    exploration_pre_contact_hit_variant_scale_max: float = 1.5
    exploration_pre_contact_hit_variant_min_upward_displacement_x: float = 0.12
    exploration_policy_takeover_enabled: bool = True
    exploration_policy_takeover_base_dir: str = (
        "scripts/smooth_policy/amp_history/amp_training/td3/extras/warm_start"
    )
    exploration_policy_takeover_model_path: str = "model2.pth"
    exploration_policy_takeover_args_path: str = "args.yaml"
    exploration_policy_takeover_config_path: str = "config.yaml"
    debug_logging: bool = False
    save_sampled_frames: bool = True
    sampled_frame_interval: int = 5
    sampled_frame_dirname: str = "sampled_frames"


def primitive_names_in_run(which: str) -> list[str]:
    if which == "all":
        return [
            "stand_still",
            "same_direction",
            "y_aligned",
            "policy_takeover",
            "target_position_directional",
            "pre_contact_hit_variant",
        ]
    return [which]


def set_one_hot_primitive(selector: PrimitiveExplorationSelector, primitive_name: str) -> None:
    weights = {
        "stand_still": 0.0,
        "same_direction": 0.0,
        "y_aligned": 0.0,
        "policy_takeover": 0.0,
        "target_position_directional": 0.0,
        "pre_contact_hit_variant": 0.0,
    }
    if primitive_name not in weights:
        raise ValueError(f"Unknown primitive '{primitive_name}'")
    weights[primitive_name] = 1.0
    selector.set_primitive_weights(
        stand_still=weights["stand_still"],
        same_direction=weights["same_direction"],
        y_aligned=weights["y_aligned"],
        policy_takeover=weights["policy_takeover"],
        target_position_directional=weights["target_position_directional"],
        pre_contact_hit_variant=weights["pre_contact_hit_variant"],
    )


def build_proposed_action(args: Args, action_shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    if args.base_action_mode == "zero":
        return np.zeros(action_shape, dtype=np.float32)
    if args.base_action_mode == "random":
        return np.clip(
            rng.normal(0.0, args.random_action_std, size=action_shape),
            -1.0,
            1.0,
        ).astype(np.float32)
    raise ValueError(f"Unsupported base_action_mode '{args.base_action_mode}'")


def render_primitive_gif(
    args: Args,
    env_params: dict,
    primitive_name: str,
    seed: int,
    gif_index: int,
) -> str:
    env = AirHockeyEnv(dict(env_params))
    env.max_timesteps = int(args.max_timesteps)
    renderer = AirHockeyRenderer(env, show_target_position=True, show_acceleration_arrow=False)
    rng = np.random.default_rng(seed + gif_index)

    selector = PrimitiveExplorationSelector(
        num_envs=1,
        chance=1.0,
        takeover_steps=int(args.exploration_primitive_steps),
        device=args.device,
        dtype=torch.float32,
        direction_y_component_weight=float(args.exploration_direction_y_component_weight),
        target_min_distance=float(args.exploration_target_position_min_distance),
        target_max_distance=float(args.exploration_target_position_max_distance),
        target_action_delta_x=float(args.exploration_target_position_delta_x),
        target_action_delta_y=float(args.exploration_target_position_delta_y),
        same_direction_min_angle_deg=float(args.exploration_same_direction_min_angle_deg),
        same_direction_max_angle_deg=float(args.exploration_same_direction_max_angle_deg),
        same_direction_min_magnitude=float(args.exploration_same_direction_min_magnitude),
        same_direction_max_magnitude=float(args.exploration_same_direction_max_magnitude),
        y_aligned_min_angle_deg=float(args.exploration_y_aligned_min_angle_deg),
        y_aligned_max_angle_deg=float(args.exploration_y_aligned_max_angle_deg),
        y_aligned_min_magnitude=float(args.exploration_y_aligned_min_magnitude),
        y_aligned_max_magnitude=float(args.exploration_y_aligned_max_magnitude),
        target_position_directional_min_angle_deg=float(
            args.exploration_target_position_directional_min_angle_deg
        ),
        target_position_directional_max_angle_deg=float(
            args.exploration_target_position_directional_max_angle_deg
        ),
        target_position_directional_min_magnitude=float(
            args.exploration_target_position_directional_min_magnitude
        ),
        target_position_directional_max_magnitude=float(
            args.exploration_target_position_directional_max_magnitude
        ),
        target_takeover_steps=int(args.exploration_target_position_steps),
        pre_contact_hit_variant_chance=float(args.exploration_pre_contact_hit_variant_chance),
        pre_contact_hit_variant_steps=int(args.exploration_pre_contact_hit_variant_steps),
        pre_contact_hit_variant_distance_threshold=float(
            args.exploration_pre_contact_hit_variant_distance_threshold
        ),
        pre_contact_hit_variant_scale_min=float(args.exploration_pre_contact_hit_variant_scale_min),
        pre_contact_hit_variant_scale_max=float(args.exploration_pre_contact_hit_variant_scale_max),
        pre_contact_hit_variant_min_upward_displacement_x=float(
            args.exploration_pre_contact_hit_variant_min_upward_displacement_x
        ),
    )
    set_one_hot_primitive(selector, primitive_name)

    obs, _ = env.reset(seed=seed + gif_index)
    action_low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=args.device).unsqueeze(0)
    action_high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=args.device).unsqueeze(0)
    raw_obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    warm_policy_actor = None
    warm_policy_use_last_action = False
    warm_policy_last_action = torch.zeros((1, act_dim), dtype=torch.float32, device=args.device)
    if args.exploration_policy_takeover_enabled and primitive_name in (
        "policy_takeover",
        "pre_contact_hit_variant",
    ):
        base_dir = args.exploration_policy_takeover_base_dir
        warm_model_path = resolve_path_relative_to_dir(args.exploration_policy_takeover_model_path, base_dir)
        warm_args_path = resolve_path_relative_to_dir(args.exploration_policy_takeover_args_path, base_dir)
        warm_config_path = resolve_path_relative_to_dir(args.exploration_policy_takeover_config_path, base_dir)
        warm_policy_actor, warm_policy_use_last_action = create_warm_start_policy(
            warm_model_path=warm_model_path,
            warm_args_path=warm_args_path,
            warm_config_path=warm_config_path,
            env_action_space=env.action_space,
            raw_obs_dim=raw_obs_dim,
            act_dim=act_dim,
            device=args.device,
        )

    frames = []
    sampled_frame_dir = os.path.join(
        args.output_dir,
        args.sampled_frame_dirname,
        f"{primitive_name}_seed{seed + gif_index}",
    )
    if args.save_sampled_frames:
        os.makedirs(sampled_frame_dir, exist_ok=True)
    pre_contact_active_steps = 0
    pre_contact_uninitialized_steps = 0
    pre_contact_fixed_steps = 0
    pre_contact_repeat_steps = 0
    previous_puck_position = extract_current_puck_position(
        torch.as_tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
    ).clone()

    for step_idx in range(int(args.max_timesteps)):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
        current_paddle_pos = extract_current_paddle_position(obs_tensor)
        current_puck_pos = extract_current_puck_position(obs_tensor)
        current_puck_vel = extract_current_puck_velocity(obs_tensor)
        if torch.all(current_puck_vel == 0):
            current_puck_vel = current_puck_pos - previous_puck_position
        paddle_puck_distance = torch.norm(current_puck_pos - current_paddle_pos, dim=-1)
        y_alignment_sign = torch.sign(current_puck_pos[:, 1] - current_paddle_pos[:, 1])
        if primitive_name == "pre_contact_hit_variant" and warm_policy_actor is not None:
            with torch.no_grad():
                warm_policy_obs = augment_policy_observation(
                    obs_tensor,
                    warm_policy_last_action,
                    warm_policy_use_last_action,
                )
                proposed_actions = deterministic_actor_action(warm_policy_actor, warm_policy_obs)
                proposed_actions = torch.clamp(proposed_actions, action_low, action_high)
        else:
            proposed_np = build_proposed_action(args, env.action_space.shape, rng)
            proposed_actions = torch.as_tensor(proposed_np, dtype=torch.float32, device=args.device).unsqueeze(0)
        selected_action, stats = selector.apply(
            proposed_actions=proposed_actions,
            action_low=action_low,
            action_high=action_high,
            y_alignment_sign=y_alignment_sign,
            current_paddle_position=current_paddle_pos,
            current_puck_position=current_puck_pos,
            current_puck_velocity=current_puck_vel,
            return_stats=True,
        )
        if primitive_name == "pre_contact_hit_variant":
            pre_contact_active_steps += int(stats["pre_contact_hit_variant_applied_count"])
            if stats["pre_contact_hit_variant_applied_count"] > 0:
                mode_id = int(selector.hit_variant_mode[0].item())
                if mode_id < 0:
                    pre_contact_uninitialized_steps += 1
                    if args.debug_logging:
                        print(
                            f"[WARN] seed={seed + gif_index} step={step_idx} "
                            "pre_contact active with uninitialized mode."
                        )
                elif mode_id == 0:
                    pre_contact_fixed_steps += 1
                elif mode_id == 1:
                    pre_contact_repeat_steps += 1
        if warm_policy_actor is not None:
            policy_takeover_mask = selector.active & (
                selector.primitive_id == selector.primitive_ids.POLICY_TAKEOVER
            )
            if torch.any(policy_takeover_mask):
                with torch.no_grad():
                    warm_policy_obs = augment_policy_observation(
                        obs_tensor,
                        warm_policy_last_action,
                        warm_policy_use_last_action,
                    )
                    warm_policy_actions = deterministic_actor_action(warm_policy_actor, warm_policy_obs)
                    warm_policy_actions = torch.clamp(warm_policy_actions, action_low, action_high)
                    selected_action[policy_takeover_mask] = warm_policy_actions[policy_takeover_mask]
        action_np = selected_action.squeeze(0).detach().cpu().numpy()

        next_obs, reward, terminated, truncated, _ = env.step(action_np)
        done = bool(terminated or truncated)
        selector.reset(torch.tensor([done], dtype=torch.bool, device=args.device))
        warm_policy_last_action = selected_action.clone()
        if done:
            warm_policy_last_action.zero_()

        frame = renderer.get_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.putText(
            frame,
            f"primitive={primitive_name} step={step_idx}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            frame,
            f"paddle_a=({action_np[0]:+.2f}, {action_np[1]:+.2f}) rew={float(reward):+.2f}",
            (12, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )
        puck_vel_np = current_puck_vel[0].detach().cpu().numpy()
        cv2.putText(
            frame,
            (
                f"dist={float(paddle_puck_distance[0].item()):.3f} "
                f"puck_v=({float(puck_vel_np[0]):+.3f}, {float(puck_vel_np[1]):+.3f})"
            ),
            (12, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )
        if warm_policy_actor is not None:
            cv2.putText(
                frame,
                "policy_takeover_source=warm_start_model",
                (12, 96),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )
        if stats["target_position_directional_applied_count"] > 0:
            target_pos = selector.target_position[0].detach().cpu().numpy()
            cv2.putText(
                frame,
                f"target=({target_pos[0]:+.2f}, {target_pos[1]:+.2f})",
                (12, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )
        if stats["pre_contact_hit_variant_applied_count"] > 0:
            mode_id = int(selector.hit_variant_mode[0].item())
            mode_name = "fixed_target" if mode_id == 0 else "repeat_delta"
            scale = float(selector.hit_variant_scale[0].item())
            cv2.putText(
                frame,
                f"hit_variant={mode_name} scale={scale:.2f}",
                (12, 144),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )
        if args.save_sampled_frames and step_idx % max(int(args.sampled_frame_interval), 1) == 0:
            sampled_frame_path = os.path.join(sampled_frame_dir, f"step_{step_idx:04d}.png")
            cv2.imwrite(sampled_frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frames.append(frame)
        obs = next_obs
        previous_puck_position = extract_current_puck_position(
            torch.as_tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
        ).clone()
        if done:
            break

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{primitive_name}_seed{seed + gif_index}.gif")
    duration_ms = int(1000 * 1 / max(int(args.fps), 1))
    imageio.mimsave(out_path, frames, format="GIF", loop=0, duration=duration_ms)
    env.close()
    if primitive_name == "pre_contact_hit_variant":
        print(
            f"[pre_contact_stats] seed={seed + gif_index} active_steps={pre_contact_active_steps} "
            f"fixed_steps={pre_contact_fixed_steps} repeat_steps={pre_contact_repeat_steps} "
            f"uninitialized_steps={pre_contact_uninitialized_steps}"
        )
    return out_path


if __name__ == "__main__":
    args = tyro.cli(Args)
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    env_params = dict(config["air_hockey"])
    env_params["simulator"] = "box2d"

    primitive_names = primitive_names_in_run(args.primitive)
    all_outputs = []
    for primitive_name in primitive_names:
        for gif_index in range(int(args.gifs_per_primitive)):
            out = render_primitive_gif(
                args=args,
                env_params=env_params,
                primitive_name=primitive_name,
                seed=int(args.seed),
                gif_index=gif_index,
            )
            all_outputs.append(out)
            print(f"Saved {out}")

    print("Generated GIFs:")
    for out in all_outputs:
        print(f" - {out}")
