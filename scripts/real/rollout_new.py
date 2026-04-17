import os
import sys
from pathlib import Path

# Force the repo root to the front of sys.path BEFORE importing anything else
# so the local `scripts/` package wins over /opt/ros/iron/.../scripts when the
# ROS environment is sourced. The repo root may already be on sys.path (via
# easy-install.pth from `pip install -e .`), but it sits after the script's
# directory and the ROS PYTHONPATH entries, so we have to move it to index 0.
REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT_STR = str(REPO_ROOT)
while _REPO_ROOT_STR in sys.path:
    sys.path.remove(_REPO_ROOT_STR)
sys.path.insert(0, _REPO_ROOT_STR)

import torch
import argparse
import yaml
from types import SimpleNamespace
from abc import ABC, abstractmethod
import gymnasium as gym
from airhockey import AirHockeyEnv
import numpy as np
from airhockey.airhockey_base import get_observation_by_type
from airhockey.sims.real.multiprocessing import NonBlockingConsole

from scripts.smooth_policy.agent import Agent as ResidualAgent
from scripts.smooth_policy.deterministic_agent import DeterministicAgent
from scripts.real.agent import Agent as LegacyMLPAgent

def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        val = value.strip().lower()
        if val in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if val in {"0", "false", "f", "no", "n", "off"}:
            return False
    raise ValueError(f"Cannot coerce value '{value}' to bool.")


def load_model_settings_from_model_folder(model_path):
    """
    Read rollout-relevant settings from config files next to model checkpoint.
    Priority:
    1) args.yaml
    2) config.yaml
    """
    model_dir = Path(model_path).resolve().parent
    candidates = [model_dir / "args.yaml", model_dir / "config.yaml"]

    settings = {
        "use_last_action_in_policy_state": None,
        "agent_hidden_layer_size": None,
        "agent_num_hidden_layers": None,
        "action_scale": None,
        "use_pid": None,
        "sources": {},
    }

    for p in candidates:
        if not p.exists():
            continue
        with open(p, "r") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        if not isinstance(cfg, dict):
            continue

        if settings["use_last_action_in_policy_state"] is None and "use_last_action_in_policy_state" in cfg:
            settings["use_last_action_in_policy_state"] = _coerce_bool(cfg["use_last_action_in_policy_state"])
            settings["sources"]["use_last_action_in_policy_state"] = str(p)

        if settings["agent_hidden_layer_size"] is None:
            if "agent_hidden_layer_size" in cfg:
                settings["agent_hidden_layer_size"] = int(cfg["agent_hidden_layer_size"])
                settings["sources"]["agent_hidden_layer_size"] = str(p)
            elif "agent_hidden_size" in cfg:
                settings["agent_hidden_layer_size"] = int(cfg["agent_hidden_size"])
                settings["sources"]["agent_hidden_layer_size"] = str(p)

        if settings["agent_num_hidden_layers"] is None and "agent_num_hidden_layers" in cfg:
            settings["agent_num_hidden_layers"] = int(cfg["agent_num_hidden_layers"])
            settings["sources"]["agent_num_hidden_layers"] = str(p)

        if settings["action_scale"] is None and "action_scale" in cfg:
            settings["action_scale"] = float(cfg["action_scale"])
            settings["sources"]["action_scale"] = str(p)

        if settings["use_pid"] is None and isinstance(cfg.get("air_hockey"), dict) and "use_pid" in cfg["air_hockey"]:
            settings["use_pid"] = _coerce_bool(cfg["air_hockey"]["use_pid"])
            settings["sources"]["use_pid"] = str(p)

    return settings


def load_arch_settings_from_args_file(args_file_path):
    settings = {
        "use_last_action_in_policy_state": None,
        "agent_hidden_layer_size": None,
        "agent_num_hidden_layers": None,
        "action_scale": None,
        "use_pid": None,
        "source": None,
    }
    if args_file_path is None:
        return settings
    if not os.path.exists(args_file_path):
        raise FileNotFoundError(f"Requested --args-file does not exist: {args_file_path}")

    with open(args_file_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected --args-file to contain a YAML mapping, got: {type(cfg)}")

    settings["source"] = args_file_path
    if "use_last_action_in_policy_state" in cfg:
        settings["use_last_action_in_policy_state"] = _coerce_bool(cfg["use_last_action_in_policy_state"])
    if "agent_hidden_layer_size" in cfg:
        settings["agent_hidden_layer_size"] = int(cfg["agent_hidden_layer_size"])
    elif "agent_hidden_size" in cfg:
        settings["agent_hidden_layer_size"] = int(cfg["agent_hidden_size"])
    if "agent_num_hidden_layers" in cfg:
        settings["agent_num_hidden_layers"] = int(cfg["agent_num_hidden_layers"])
    if "action_scale" in cfg:
        settings["action_scale"] = float(cfg["action_scale"])
    if isinstance(cfg.get("air_hockey"), dict) and "use_pid" in cfg["air_hockey"]:
        settings["use_pid"] = _coerce_bool(cfg["air_hockey"]["use_pid"])
    return settings


def build_policy_env_view(policy_obs_dim, action_dim):
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(policy_obs_dim),),
            dtype=np.float32,
        ),
        single_action_space=gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(action_dim),),
            dtype=np.float32,
        ),
    )


def infer_policy_dims_from_state_dict(state_dict):
    actor_input_dim = None
    preferred_keys = (
        "actor.0.weight",
        "actor.blocks.0.units.0.0.weight",
        "actor.blocks.0.skip_projection.weight",
    )
    for key in preferred_keys:
        tensor = state_dict.get(key)
        if torch.is_tensor(tensor) and tensor.ndim == 2:
            actor_input_dim = int(tensor.shape[1])
            break

    if actor_input_dim is None:
        actor_weight_shapes = []
        for key, value in state_dict.items():
            if key.startswith("actor.") and key.endswith(".weight") and torch.is_tensor(value) and value.ndim == 2:
                actor_weight_shapes.append((int(value.shape[0]), int(value.shape[1])))
        if not actor_weight_shapes:
            raise ValueError("Unable to infer policy observation dimension from checkpoint actor weights.")
        non_square_inputs = [in_dim for out_dim, in_dim in actor_weight_shapes if out_dim != in_dim]
        if non_square_inputs:
            actor_input_dim = max(non_square_inputs)
        else:
            actor_input_dim = min(in_dim for _, in_dim in actor_weight_shapes)

    action_dim = int(state_dict["actor_mean_head.weight"].shape[0])
    return actor_input_dim, action_dim


def unwrap_eval_state_dict(loaded_obj):
    if not isinstance(loaded_obj, dict):
        raise TypeError(f"Expected checkpoint/state_dict to be a dict, got {type(loaded_obj)}")

    candidate = loaded_obj
    if "state_dict" in loaded_obj and isinstance(loaded_obj["state_dict"], dict):
        candidate = loaded_obj["state_dict"]
    elif "actor" in loaded_obj and isinstance(loaded_obj["actor"], dict):
        # TD3-style checkpoints store actor weights under "actor".
        candidate = loaded_obj["actor"]

    tensor_keys = [k for k, v in candidate.items() if isinstance(k, str) and torch.is_tensor(v)]
    if not tensor_keys:
        raise ValueError("Could not find tensor parameters in provided checkpoint/state dict.")
    return candidate


def infer_policy_class_from_state_dict(state_dict):
    keys = set(state_dict.keys())
    has_agent_only_keys = (
        "actor_logstd" in keys
        or "LOG_STD_MIN" in keys
        or "LOG_STD_MAX" in keys
        or "EPS" in keys
        or any(key.startswith("critic.") for key in keys)
        or "critic_head.weight" in keys
    )
    if has_agent_only_keys:
        return "agent"

    has_actor_keys = any(
        key.startswith("actor.") or key.startswith("actor_mean_head.") for key in keys
    )
    if has_actor_keys:
        return "deterministic_agent"

    preview_keys = sorted(list(keys))[:10]
    raise ValueError(
        f"Unable to infer policy type from checkpoint keys. Example keys: {preview_keys}"
    )


def infer_agent_arch_variant_from_state_dict(state_dict):
    keys = set(state_dict.keys())
    if "actor.0.weight" in keys or "actor.2.weight" in keys:
        return "legacy_mlp"
    if any(key.startswith("actor.blocks.") for key in keys):
        return "residual"
    return "unknown"


def infer_num_actor_blocks_from_state_dict(state_dict):
    block_ids = set()
    for key in state_dict.keys():
        if not key.startswith("actor.blocks."):
            continue
        parts = key.split(".")
        if len(parts) < 3:
            continue
        try:
            block_ids.add(int(parts[2]))
        except ValueError:
            continue
    return (max(block_ids) + 1) if block_ids else None


def build_policy(
    policy_type,
    policy_env_view,
    action_scale,
    agent_hidden_layer_size,
    agent_num_hidden_layers,
    agent_arch_variant="residual",
):
    policy_builders = {
        "agent": lambda: (
            LegacyMLPAgent(
                policy_env_view,
                action_scale=action_scale,
                action_bias=0.0,
                hidden_size=agent_hidden_layer_size,
            )
            if agent_arch_variant == "legacy_mlp"
            else ResidualAgent(
                policy_env_view,
                action_scale=action_scale,
                action_bias=0.0,
                hidden_layer_size=agent_hidden_layer_size,
                num_hidden_layers=agent_num_hidden_layers,
            )
        ),
        "deterministic_agent": lambda: DeterministicAgent(
            policy_env_view,
            action_scale=action_scale,
            action_bias=0.0,
            hidden_layer_size=agent_hidden_layer_size,
            num_hidden_layers=agent_num_hidden_layers,
        ),
    }
    if policy_type in policy_builders:
        return policy_builders[policy_type]()
    raise ValueError(f"Unsupported policy_type '{policy_type}'.")


def deterministic_actor_action(actor, policy_obs):
    if hasattr(actor, "get_action"):
        return actor.get_action(policy_obs)
    if callable(actor):
        return actor(policy_obs)
    raise TypeError(f"Unsupported actor type for action inference: {type(actor)}")


class PolicyRunner(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def act(self, policy_obs):
        raise NotImplementedError

    def get_action_scale_tensor(self, reference_tensor):
        scale = getattr(self.model, "action_scale", 1.0)
        if not torch.is_tensor(scale):
            scale = torch.tensor(scale, dtype=reference_tensor.dtype, device=reference_tensor.device)
        else:
            scale = scale.to(dtype=reference_tensor.dtype, device=reference_tensor.device)
        if scale.ndim == 0:
            scale = scale.reshape(1)
        return scale

    def normalize_for_env(self, action_tensor):
        scale = self.get_action_scale_tensor(action_tensor)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        return action_tensor / scale

    def action_scale_for_logging(self):
        if hasattr(self.model, "action_scale"):
            scale = self.model.action_scale
            if torch.is_tensor(scale):
                return float(scale.detach().cpu().item())
            return float(scale)
        return 1.0


class StochasticAgentRunner(PolicyRunner):
    def __init__(self, model, action_mode):
        super().__init__(model)
        self.action_mode = action_mode

    def act(self, policy_obs):
        if self.action_mode == "sample":
            return self.model(policy_obs)
        if self.action_mode == "deterministic":
            return deterministic_actor_action(self.model, policy_obs)
        if self.action_mode == "auto":
            return self.model(policy_obs)
        raise ValueError(f"Unsupported action_mode '{self.action_mode}'.")


class DeterministicAgentRunner(PolicyRunner):
    def __init__(self, model, action_mode):
        super().__init__(model)
        self.action_mode = action_mode

    def act(self, policy_obs):
        if self.action_mode == "sample":
            raise ValueError("--action-mode sample is only valid for policy type 'agent'.")
        if self.action_mode in {"deterministic", "auto"}:
            return deterministic_actor_action(self.model, policy_obs)
        raise ValueError(f"Unsupported action_mode '{self.action_mode}'.")


def build_policy_runner(model, policy_type, action_mode):
    runner_builders = {
        "agent": lambda: StochasticAgentRunner(model=model, action_mode=action_mode),
        "deterministic_agent": lambda: DeterministicAgentRunner(model=model, action_mode=action_mode),
    }
    if policy_type in runner_builders:
        return runner_builders[policy_type]()
    raise ValueError(f"Unsupported policy_type '{policy_type}'.")


def maybe_generate_gifs_for_saved_trajectory(
    simulator,
    auto_gif: bool,
    gif_fps: int,
    gif_max_frames_per_file: int,
):
    if not auto_gif:
        return

    saved_idx = simulator.tidx - 1
    if saved_idx < 0:
        print("No saved trajectory index found for GIF generation.")
        return

    hdf5_path = (Path(simulator.save_path) / f"trajectory_data{saved_idx}.hdf5").resolve()
    if not hdf5_path.exists():
        print(f"Skipping GIF generation; missing file: {hdf5_path}")
        return

    output_dir = hdf5_path.parent / f"{hdf5_path.stem}_gifs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Box2D-style projection GIF (paddle + puck on simulator-style table view).
    # Uses the same renderer/parameters as async_td3_real's generate_episode_gif,
    # but reads from the legacy train_vals schema rollout_new.py writes.
    try:
        from scripts.smooth_policy.visualize_demo.visualize_real_trajectory import (
            RealTrajectoryRenderer,
            create_trajectory_gif,
            extract_paddle_data,
            load_trajectory_data,
        )
        train_vals = load_trajectory_data(hdf5_path)
        paddle_data = extract_paddle_data(train_vals, require_puck=False)
        renderer = RealTrajectoryRenderer(
            table_length=1.9304,
            table_width=0.8636,
            paddle_radius=0.0508,
            puck_radius=0.03175,
            render_size=360,
            robot_x_offset=1.2,
            orientation="vertical",
            paddle_input_frame="table",
        )
        projection_gif_path = output_dir / "trajectory_visualization.gif"
        create_trajectory_gif(
            paddle_data,
            renderer,
            projection_gif_path,
            max_frames=gif_max_frames_per_file if gif_max_frames_per_file > 0 else None,
            subsample=1,
            fps=gif_fps,
        )
        print(f"Generated Box2D-projection GIF: {projection_gif_path}")
    except Exception as exc:
        print(f"Box2D-projection GIF generation failed for {hdf5_path}: {exc}")

    # Ground-truth camera video (MP4) from train_img frames in the same HDF5.
    try:
        from scripts.smooth_policy.amp_history.amp_training.td3.helper.episode_artifacts import (
            generate_episode_camera_video,
        )
        camera_video_path = generate_episode_camera_video(
            episode_hdf5_path=hdf5_path,
            video_root=output_dir.parent,
            fps=gif_fps,
            max_frames=gif_max_frames_per_file if gif_max_frames_per_file > 0 else None,
            subsample=1,
            codec="mp4v",
        )
        print(f"Generated ground-truth camera video: {camera_video_path}")
    except Exception as exc:
        print(f"Camera video generation failed for {hdf5_path}: {exc}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rollout')

    # optional arguments if use-parent-log-dir is False
    parser.add_argument('--model', type=str, default="ex_model/model.pth", help='Path to the model to evaluate.')
    parser.add_argument('--args-file', type=str, default=None, help='Optional args YAML (e.g. TD3/PPO args.yaml) to resolve architecture-related rollout settings.')
    parser.add_argument('--config-path', type=str, default="configs/real_configs/rollout_config.yaml", help='Path to the config file.')
    parser.add_argument('--save-path', type=str, default=None, help='Override trajectory save path (defaults to config value).')
    parser.add_argument('--policy-type', type=str, choices=['auto', 'agent', 'deterministic_agent'], default='auto', help='Policy class to use. auto infers from checkpoint keys.')
    parser.add_argument('--action-mode', type=str, choices=['auto', 'sample', 'deterministic'], default='auto', help='Action selection mode. auto=sample for Agent, deterministic for DeterministicAgent.')
    parser.add_argument('--action-scale', type=float, default=None, help='Override action scale. If omitted, resolves from model config then defaults to 0.2.')
    parser.add_argument('--agent-hidden-size', type=int, default=None, help='Override hidden layer size for policy network.')
    parser.add_argument('--agent-num-hidden-layers', type=int, default=None, help='Override number of hidden residual blocks.')
    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='torch device')
    parser.add_argument('--auto-gif', action='store_true', help='Generate GIF visualization(s) after each saved trajectory.')
    parser.add_argument('--gif-fps', type=int, default=20, help='GIF playback FPS used when --auto-gif is enabled.')
    parser.add_argument('--gif-max-frames-per-file', type=int, default=250, help='Maximum rendered frames per GIF when --auto-gif is enabled.')
    parser.add_argument('--puck-absence-halt-steps', type=int, default=15,
                        help='Halt policy (send zero action) after this many consecutive frames with the puck undetected. '
                             'The policy resumes the instant the puck is detected again. Set <=0 to disable.')
    parser.add_argument('--reset-position', type=str,
                        choices=['extreme_left', 'left', 'middle', 'right', 'extreme_right'],
                        default='middle',
                        help='Initial paddle reset y-position. middle keeps current behavior; left/right are halfway to the workspace edge; extreme_* are at the workspace bounds. '
                             'Can be changed at runtime via number keys 1-5 (extreme_left=1, left=2, middle=3, right=4, extreme_right=5).')

    args = parser.parse_args()
    
    air_hockey_cfg = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)
    air_hockey_params = air_hockey_cfg['air_hockey']
    
    # processing to avoid bugs
    air_hockey_params['n_training_steps'] = air_hockey_cfg['n_training_steps']

    if 'sac' == air_hockey_cfg['algorithm']:
        if 'goal' in air_hockey_cfg['air_hockey']['task']:
            air_hockey_cfg['air_hockey']['return_goal_obs'] = True
        else:
            air_hockey_cfg['air_hockey']['return_goal_obs'] = False
    else:
        air_hockey_cfg['air_hockey']['return_goal_obs'] = False
    air_hockey_params_cp = air_hockey_params.copy()
    air_hockey_params_cp['seed'] = 42
    air_hockey_params_cp['max_timesteps'] = 500 # override behavior (effectively "no cap"; user drives episode boundaries)

    if args.save_path is not None:
        air_hockey_params_cp['simulator_params']['save_path'] = args.save_path
    
    eval_air_hockey_params = air_hockey_params_cp.copy()
    print("trajectory save path:", eval_air_hockey_params['simulator_params']['save_path'])
    
    # Create environment factory function
    def make_eval_env():
        env = AirHockeyEnv(eval_air_hockey_params)
        return env
    
    eval_env = make_eval_env()

    sim = eval_env.simulator
    y_min = float(getattr(sim, 'y_min'))
    y_max = float(getattr(sim, 'y_max'))
    RESET_POSITION_KEYS = {
        '1': 'extreme_left',
        '2': 'left',
        '3': 'middle',
        '4': 'right',
        '5': 'extreme_right',
    }
    reset_y_map = {
        'extreme_left':  y_min,
        'left':          y_min / 2.0,
        'middle':        0.0,
        'right':         y_max / 2.0,
        'extreme_right': y_max,
    }

    def apply_reset_position(name, *, verbose=True):
        new_y = reset_y_map[name]
        sim.reset_pose[0][1] = new_y
        if verbose:
            print(
                f"Reset position '{name}': paddle reset y set to {new_y:.4f} "
                f"(workspace y in [{y_min:.4f}, {y_max:.4f}], x kept at {sim.reset_pose[0][0]:.4f})."
            )
        return new_y

    current_reset_position = args.reset_position
    apply_reset_position(current_reset_position)

    device = torch.device(args.device)
    loaded_obj = torch.load(args.model, map_location=device)
    policy_state_dict = unwrap_eval_state_dict(loaded_obj)
    model_cfg = load_model_settings_from_model_folder(args.model)
    args_file_cfg = load_arch_settings_from_args_file(args.args_file)
    model_obs_dim, model_action_dim = infer_policy_dims_from_state_dict(policy_state_dict)
    checkpoint_hidden_size = int(policy_state_dict["actor_mean_head.weight"].shape[1])
    inferred_actor_blocks = infer_num_actor_blocks_from_state_dict(policy_state_dict)

    if args.policy_type == "auto":
        resolved_policy_type = infer_policy_class_from_state_dict(policy_state_dict)
        print(f"Using inferred policy type: {resolved_policy_type}")
    else:
        resolved_policy_type = args.policy_type
        print(f"Using policy type from CLI: {resolved_policy_type}")

    agent_arch_variant = infer_agent_arch_variant_from_state_dict(policy_state_dict)
    if resolved_policy_type == "agent":
        print(f"Inferred agent architecture variant from checkpoint: {agent_arch_variant}")

    if args.agent_hidden_size is not None:
        agent_hidden_layer_size = int(args.agent_hidden_size)
        print("Using agent_hidden_size from CLI:", agent_hidden_layer_size)
    elif args_file_cfg["agent_hidden_layer_size"] is not None:
        agent_hidden_layer_size = int(args_file_cfg["agent_hidden_layer_size"])
        print(
            "Using agent_hidden_layer_size from args file:",
            agent_hidden_layer_size,
            f"(source: {args_file_cfg['source']})",
        )
    elif model_cfg["agent_hidden_layer_size"] is not None:
        agent_hidden_layer_size = int(model_cfg["agent_hidden_layer_size"])
        print(
            "Using agent_hidden_layer_size from model config:",
            agent_hidden_layer_size,
            f"(source: {model_cfg['sources'].get('agent_hidden_layer_size', 'unknown')})",
        )
    else:
        agent_hidden_layer_size = 128
        print("No agent_hidden_layer_size found; using default:", agent_hidden_layer_size)
    if agent_hidden_layer_size != checkpoint_hidden_size:
        print(
            "Overriding hidden size to match checkpoint:",
            checkpoint_hidden_size,
            f"(requested {agent_hidden_layer_size})",
        )
        agent_hidden_layer_size = checkpoint_hidden_size

    if args.agent_num_hidden_layers is not None:
        agent_num_hidden_layers = int(args.agent_num_hidden_layers)
        print("Using agent_num_hidden_layers from CLI:", agent_num_hidden_layers)
    elif args_file_cfg["agent_num_hidden_layers"] is not None:
        agent_num_hidden_layers = int(args_file_cfg["agent_num_hidden_layers"])
        print(
            "Using agent_num_hidden_layers from args file:",
            agent_num_hidden_layers,
            f"(source: {args_file_cfg['source']})",
        )
    elif model_cfg["agent_num_hidden_layers"] is not None:
        agent_num_hidden_layers = int(model_cfg["agent_num_hidden_layers"])
        print(
            "Using agent_num_hidden_layers from model config:",
            agent_num_hidden_layers,
            f"(source: {model_cfg['sources'].get('agent_num_hidden_layers', 'unknown')})",
        )
    else:
        agent_num_hidden_layers = 2
        print("No agent_num_hidden_layers found; using default:", agent_num_hidden_layers)

    if resolved_policy_type != "agent" or agent_arch_variant != "legacy_mlp":
        if inferred_actor_blocks is not None and agent_num_hidden_layers != inferred_actor_blocks:
            print(
                "Overriding agent_num_hidden_layers to match checkpoint:",
                inferred_actor_blocks,
                f"(requested {agent_num_hidden_layers})",
            )
            agent_num_hidden_layers = inferred_actor_blocks

    if args.action_scale is not None:
        resolved_action_scale = float(args.action_scale)
        print("Using action_scale from CLI:", resolved_action_scale)
    elif args_file_cfg["use_pid"] is True:
        resolved_action_scale = 1.0
        print(
            "Using PID-derived action_scale=1.0 from args file",
            f"(source: {args_file_cfg['source']})",
        )
    elif args_file_cfg["action_scale"] is not None:
        resolved_action_scale = float(args_file_cfg["action_scale"])
        print(
            "Using action_scale from args file:",
            resolved_action_scale,
            f"(source: {args_file_cfg['source']})",
        )
    elif model_cfg["use_pid"] is True:
        resolved_action_scale = 1.0
        print(
            "Using PID-derived action_scale=1.0 from model config",
            f"(source: {model_cfg['sources'].get('use_pid', 'unknown')})",
        )
    elif model_cfg["action_scale"] is not None:
        resolved_action_scale = float(model_cfg["action_scale"])
        print(
            "Using action_scale from model config:",
            resolved_action_scale,
            f"(source: {model_cfg['sources'].get('action_scale', 'unknown')})",
        )
    else:
        resolved_action_scale = 0.2
        print("No action_scale found; using default:", resolved_action_scale)

    policy_env_view = build_policy_env_view(policy_obs_dim=model_obs_dim, action_dim=model_action_dim)
    policy_model = build_policy(
        policy_type=resolved_policy_type,
        policy_env_view=policy_env_view,
        action_scale=resolved_action_scale,
        agent_hidden_layer_size=agent_hidden_layer_size,
        agent_num_hidden_layers=agent_num_hidden_layers,
        agent_arch_variant=agent_arch_variant,
    )
    policy_runner = build_policy_runner(
        model=policy_model,
        policy_type=resolved_policy_type,
        action_mode=args.action_mode,
    )
    print(f"Using policy runner: {type(policy_runner).__name__}")
    use_last_action = model_obs_dim > eval_env.single_observation_space.shape[0]
    if args_file_cfg["use_last_action_in_policy_state"] is not None:
        use_last_action = bool(args_file_cfg["use_last_action_in_policy_state"])
        print(
            "Using use_last_action_in_policy_state from args file:",
            use_last_action,
            f"(source: {args_file_cfg['source']})",
        )
    elif model_cfg["use_last_action_in_policy_state"] is not None:
        use_last_action = bool(model_cfg["use_last_action_in_policy_state"])
    last_action_for_policy = torch.zeros((1, model_action_dim), dtype=torch.float32, device=device)

    policy_model.load_state_dict(policy_state_dict)
    policy_model = policy_model.to(device=device)
    policy_model.eval()

    print("model action scale: ", policy_runner.action_scale_for_logging())
    # model.action_scale = torch.tensor(0.2) # manually scaling just for testing a model
    
    state_dict = eval_env.simulator.get_current_state()
    obs_type = "history"
    
    obs = get_observation_by_type(state_dict, obs_type=obs_type, puck_history=state_dict["pucks"][0]["history"], paddle_history=state_dict['paddles']['paddle_ego']['history'])

    def refresh_post_reset_state(last_action_tensor, gate_policy_until_puck=True):
        current_state_dict = eval_env.simulator.get_current_state()
        current_obs = get_observation_by_type(
            current_state_dict,
            obs_type=obs_type,
            puck_history=current_state_dict["pucks"][0]["history"],
            paddle_history=current_state_dict['paddles']['paddle_ego']['history'],
        )
        delay_counter_val = 0
        episode_halted_val = False
        episode_timestep_val = 0
        if use_last_action:
            last_action_tensor.zero_()
        # Startup gate: always wait until the puck has been detected on
        # STARTUP_REQUIRED_CONSECUTIVE_DETECTIONS consecutive poll frames
        # before the policy starts, regardless of --puck-absence-halt-steps.
        # The consecutive-frames requirement debounces single-frame false
        # positives (e.g. a stray red pixel cluster that happens to be
        # inside the field bounds).
        waiting_for_first_detection_val = bool(gate_policy_until_puck)
        consecutive_puck_detections_val = 0
        # Mid-episode absence gate: only engages if --puck-absence-halt-steps > 0.
        puck_absent_val = int(args.puck_absence_halt_steps) if gate_policy_until_puck and int(args.puck_absence_halt_steps) > 0 else 0
        policy_gated_val = gate_policy_until_puck and int(args.puck_absence_halt_steps) > 0
        return (
            current_state_dict,
            current_obs,
            delay_counter_val,
            episode_halted_val,
            episode_timestep_val,
            puck_absent_val,
            policy_gated_val,
            waiting_for_first_detection_val,
            consecutive_puck_detections_val,
        )

    def puck_occluded_from_state(cur_state_dict):
        # puck_history entries are (x, y, occluded_flag); state_dict carries the newest
        # flag as a shape-(1,) array under "occluded". 1 => not detected, 0 => detected.
        try:
            flag_arr = np.asarray(cur_state_dict["pucks"][0]["occluded"], dtype=float).ravel()
        except (KeyError, IndexError, TypeError):
            return False
        if flag_arr.size == 0:
            return False
        return bool(flag_arr[-1] >= 0.5)

    MIN_EPISODE_TIMESTEPS = 50

    def startup_prompt():
        print(
            f"Initial reset: current position is '{current_reset_position}'. "
            "Press 'c' (or 'y') to reset to current position, "
            "1-5 to reset to a different position "
            "(1=extreme_left, 2=left, 3=middle, 4=right, 5=extreme_right), or 'x' to exit."
        )

    obs_list = list()
    with NonBlockingConsole() as nbc:
        delay_counter = 0
        # Start in halted state so the user must explicitly choose where to
        # reset before the first episode (same prompt/flow as after an episode end).
        episode_halted = True
        total_timestep = 0
        episode_timestep = 0
        puck_absence_threshold = int(args.puck_absence_halt_steps)
        if puck_absence_threshold > 0:
            print(f"Puck-absence gate enabled: policy starts only when the puck is detected (threshold {puck_absence_threshold} frames for mid-episode halts).")
            puck_absent_consecutive = puck_absence_threshold
            policy_gated_by_puck_absence = True
        else:
            print("Puck-absence gate disabled (--puck-absence-halt-steps <= 0).")
            puck_absent_consecutive = 0
            policy_gated_by_puck_absence = False
        # Startup gate: block the policy until the puck has been detected on
        # STARTUP_REQUIRED_CONSECUTIVE_DETECTIONS consecutive poll frames,
        # regardless of --puck-absence-halt-steps. Re-armed on every reset.
        STARTUP_REQUIRED_CONSECUTIVE_DETECTIONS = 5
        waiting_for_first_detection = True
        consecutive_puck_detections = 0
        startup_prompt()
        def do_reset_and_refresh(write_traj):
            actual_write = write_traj
            if write_traj and episode_timestep < MIN_EPISODE_TIMESTEPS:
                print(
                    f"[short-episode-skip] only {episode_timestep} timesteps elapsed "
                    f"(need >= {MIN_EPISODE_TIMESTEPS}); discarding instead of saving."
                )
                actual_write = False
            if actual_write:
                label = "Episode ended — saving trajectory and starting next episode..." if episode_halted else "Saving trajectory and resetting..."
            else:
                label = "Resetting without saving..."
            print(label)
            eval_env.reset(seed=None, write_traj=actual_write)
            if actual_write:
                maybe_generate_gifs_for_saved_trajectory(
                    simulator=eval_env.simulator,
                    auto_gif=args.auto_gif,
                    gif_fps=args.gif_fps,
                    gif_max_frames_per_file=args.gif_max_frames_per_file,
                )
            return refresh_post_reset_state(last_action_for_policy)

        while True:
            key = nbc.get_data()
            if key in RESET_POSITION_KEYS:
                # Number keys (1-5): update reset position and immediately save+reset.
                requested_position = RESET_POSITION_KEYS[key]
                if requested_position != current_reset_position:
                    current_reset_position = requested_position
                apply_reset_position(current_reset_position)
                (
                    state_dict,
                    obs,
                    delay_counter,
                    episode_halted,
                    episode_timestep,
                    puck_absent_consecutive,
                    policy_gated_by_puck_absence,
                    waiting_for_first_detection,
                    consecutive_puck_detections,
                ) = do_reset_and_refresh(write_traj=True)
                if policy_gated_by_puck_absence or waiting_for_first_detection:
                    print(
                        f"[puck-gate] waiting for {STARTUP_REQUIRED_CONSECUTIVE_DETECTIONS} "
                        f"consecutive puck detections before starting policy..."
                    )
                continue
            if key == 'y' or (key == 'c' and episode_halted):
                # Save the just-ended (or in-progress) episode trajectory and start the next episode at the current reset position.
                (
                    state_dict,
                    obs,
                    delay_counter,
                    episode_halted,
                    episode_timestep,
                    puck_absent_consecutive,
                    policy_gated_by_puck_absence,
                    waiting_for_first_detection,
                    consecutive_puck_detections,
                ) = do_reset_and_refresh(write_traj=True)
                if policy_gated_by_puck_absence or waiting_for_first_detection:
                    print(
                        f"[puck-gate] waiting for {STARTUP_REQUIRED_CONSECUTIVE_DETECTIONS} "
                        f"consecutive puck detections before starting policy..."
                    )
                continue
            elif key == 'q':
                # Discard current/just-ended episode (no save) and reset to current position.
                (
                    state_dict,
                    obs,
                    delay_counter,
                    episode_halted,
                    episode_timestep,
                    puck_absent_consecutive,
                    policy_gated_by_puck_absence,
                    waiting_for_first_detection,
                    consecutive_puck_detections,
                ) = do_reset_and_refresh(write_traj=False)
                if policy_gated_by_puck_absence or waiting_for_first_detection:
                    print(
                        f"[puck-gate] waiting for {STARTUP_REQUIRED_CONSECUTIVE_DETECTIONS} "
                        f"consecutive puck detections before starting policy..."
                    )
                continue
            elif key == 'x':
                print("Exiting...")
                break

            if episode_halted:
                continue

            # Startup gate: do NOT advance the env at all until we see
            # STARTUP_REQUIRED_CONSECUTIVE_DETECTIONS unoccluded frames in a
            # row. Calling eval_env.step() with zero action would still tick
            # the env's current_timestep and run termination checks (e.g.
            # puck_low_motion_window_20), which would end the episode before
            # the puck is even placed. Instead, run a detection-only poll
            # that also refreshes paddle telemetry so puck_history AND
            # paddle_history are both fresh when the gate releases.
            if waiting_for_first_detection:
                eval_env.simulator.poll_puck_detection()
                state_dict = eval_env.simulator.get_current_state()
                obs = get_observation_by_type(
                    state_dict,
                    obs_type=obs_type,
                    puck_history=state_dict["pucks"][0]["history"],
                    paddle_history=state_dict["paddles"]["paddle_ego"]["history"],
                )
                if puck_occluded_from_state(state_dict):
                    consecutive_puck_detections = 0
                else:
                    consecutive_puck_detections += 1
                    if consecutive_puck_detections >= STARTUP_REQUIRED_CONSECUTIVE_DETECTIONS:
                        waiting_for_first_detection = False
                        print(
                            f"[puck-gate] puck detected on "
                            f"{consecutive_puck_detections} consecutive frames; "
                            f"starting policy."
                        )
                continue

            puck_currently_occluded = puck_occluded_from_state(state_dict)

            if puck_absence_threshold > 0:
                if puck_currently_occluded:
                    puck_absent_consecutive += 1
                    if (not policy_gated_by_puck_absence
                            and puck_absent_consecutive >= puck_absence_threshold):
                        policy_gated_by_puck_absence = True
                        print(
                            f"[puck-gate] puck undetected for {puck_absent_consecutive} "
                            f"consecutive frames; halting policy (zero action)."
                        )
                else:
                    if policy_gated_by_puck_absence:
                        print("[puck-gate] puck re-detected; resuming policy.")
                    puck_absent_consecutive = 0
                    policy_gated_by_puck_absence = False

            if policy_gated_by_puck_absence:
                action = np.zeros(model_action_dim, dtype=np.float32)
                if use_last_action:
                    last_action_for_policy.zero_()
            else:
                obs_t = torch.tensor(obs).unsqueeze(0).to(device=device).float()
                policy_obs = obs_t
                if use_last_action:
                    policy_obs = torch.cat([policy_obs, last_action_for_policy], dim=-1)
                with torch.no_grad():
                    action_tensor = policy_runner.act(policy_obs)
                    env_action_tensor = policy_runner.normalize_for_env(action_tensor)
                if delay_counter < 10 and delay_counter >= 0:
                    env_action_tensor = env_action_tensor * 0.0
                action = env_action_tensor.detach().cpu().numpy().squeeze()
                if use_last_action:
                    last_action_for_policy = env_action_tensor.detach().clone()
                delay_counter += 1
                print("action", action, obs)

            obs, reward, is_finished, truncated, info = eval_env.step(action)
            total_timestep += 1
            episode_timestep += 1

            # for puck hitting observations
            state_dict = eval_env.simulator.get_current_state()
            obs = get_observation_by_type(state_dict, obs_type=obs_type, puck_history=state_dict["pucks"][0]["history"], paddle_history=state_dict['paddles']['paddle_ego']['history'])
            if bool(is_finished or truncated):
                episode_halted = True
                if use_last_action:
                    last_action_for_policy.zero_()
                end_type = info.get("episode_end_type")
                if end_type is None:
                    # Fallback for envs that do not expose structured end reasons.
                    end_type = "termination" if bool(is_finished and not truncated) else "truncation"
                specific_reasons = info.get("episode_end_reasons", [])
                if not specific_reasons:
                    if end_type == "termination":
                        specific_reasons = info.get("termination_reasons", [])
                    elif end_type == "truncation":
                        specific_reasons = info.get("truncation_reasons", [])
                reason_str = ", ".join([str(r) for r in specific_reasons]) if len(specific_reasons) > 0 else "unspecified"
                if episode_timestep < MIN_EPISODE_TIMESTEPS:
                    # Auto-discard short episodes; do not block on user input.
                    print(
                        f"Episode ended due to: {end_type} ({reason_str}). "
                        f"Episode timesteps: {episode_timestep} (< {MIN_EPISODE_TIMESTEPS}); "
                        f"auto-discarding and resetting to '{current_reset_position}'."
                    )
                    (
                        state_dict,
                        obs,
                        delay_counter,
                        episode_halted,
                        episode_timestep,
                        puck_absent_consecutive,
                        policy_gated_by_puck_absence,
                        waiting_for_first_detection,
                        consecutive_puck_detections,
                    ) = do_reset_and_refresh(write_traj=False)
                    if policy_gated_by_puck_absence or waiting_for_first_detection:
                        print(
                            f"[puck-gate] waiting for {STARTUP_REQUIRED_CONSECUTIVE_DETECTIONS} "
                            f"consecutive puck detections before starting policy..."
                        )
                else:
                    print(
                        f"Episode ended due to: {end_type} ({reason_str}). "
                        f"Episode timesteps: {episode_timestep}, total timesteps: {total_timestep}. "
                        f"Current reset position: '{current_reset_position}'. "
                        "Press 'c' (or 'y') to save trajectory and start next episode "
                        "(policy will wait for the puck to appear), 'q' to discard this episode and reset, "
                        "1-5 to save and reset to a different position "
                        "(1=extreme_left, 2=left, 3=middle, 4=right, 5=extreme_right), or 'x' to end."
                    )




