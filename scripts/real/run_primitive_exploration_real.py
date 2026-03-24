from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from airhockey import AirHockeyEnv
from airhockey.sims.real.multiprocessing import NonBlockingConsole

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for candidate in (REPO_ROOT, SCRIPTS_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

scripts_init = SCRIPTS_ROOT / "__init__.py"
scripts_spec = importlib.util.spec_from_file_location(
    "scripts",
    scripts_init,
    submodule_search_locations=[str(SCRIPTS_ROOT)],
)
if scripts_spec is None or scripts_spec.loader is None:
    raise ImportError(f"Unable to load repo-local scripts package from {scripts_init}")
scripts_module = importlib.util.module_from_spec(scripts_spec)
sys.modules["scripts"] = scripts_module
scripts_spec.loader.exec_module(scripts_module)

from scripts.smooth_policy.amp_history.amp_training.td3.helper.exploration_selector import (  # noqa: E402
    PrimitiveExplorationSelector,
)

SUPPORTED_PRIMITIVES = {
    "stand_still",
    "same_direction",
    "y_aligned",
    "target_position_directional",
}


def load_runner_config(config_path: str, save_path_override: str | None = None) -> tuple[dict, dict]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    params = dict(cfg["air_hockey"])
    params["n_training_steps"] = cfg["n_training_steps"]
    seed_cfg = cfg.get("seed", 0)
    if isinstance(seed_cfg, (list, tuple)):
        seed_cfg = seed_cfg[0] if seed_cfg else 0
    params["seed"] = int(seed_cfg)
    params["return_goal_obs"] = bool(cfg.get("algorithm") == "sac" and "goal" in params.get("task", ""))
    if save_path_override is not None:
        params["simulator_params"]["save_path"] = save_path_override
    return params, dict(cfg.get("primitive_runner", {}))


def extract_primitive_state_tensors(env: AirHockeyEnv, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    zeros = torch.zeros((1, 2), dtype=torch.float32, device=device)
    state_info = getattr(env, "current_state", None)
    simulator = getattr(env, "simulator", None)
    if simulator is not None and hasattr(simulator, "get_current_state"):
        try:
            state_info = simulator.get_current_state()
        except Exception:
            pass
    if not isinstance(state_info, dict):
        return zeros.clone(), zeros.clone(), zeros.clone()
    try:
        paddle = torch.as_tensor(state_info["paddles"]["paddle_ego"]["position"], dtype=torch.float32, device=device).reshape(1, -1)[:, :2]
        puck = torch.as_tensor(state_info["pucks"][0]["position"], dtype=torch.float32, device=device).reshape(1, -1)[:, :2]
        puck_vel = torch.as_tensor(state_info["pucks"][0]["velocity"], dtype=torch.float32, device=device).reshape(1, -1)[:, :2]
        return paddle, puck, puck_vel
    except Exception:
        return zeros.clone(), zeros.clone(), zeros.clone()


def set_one_hot_primitive(selector: PrimitiveExplorationSelector, primitive_name: str) -> None:
    if primitive_name not in SUPPORTED_PRIMITIVES:
        raise ValueError(f"primitive_name must be one of {sorted(SUPPORTED_PRIMITIVES)}")
    selector.set_primitive_weights(
        stand_still=1.0 if primitive_name == "stand_still" else 0.0,
        same_direction=1.0 if primitive_name == "same_direction" else 0.0,
        y_aligned=1.0 if primitive_name == "y_aligned" else 0.0,
        policy_takeover=0.0,
        target_position_directional=1.0 if primitive_name == "target_position_directional" else 0.0,
        pre_contact_hit_variant=0.0,
    )


def build_selector(primitive_cfg: dict, device: torch.device) -> PrimitiveExplorationSelector:
    selector = PrimitiveExplorationSelector(
        num_envs=1,
        chance=1.0,
        takeover_steps=int(primitive_cfg.get("exploration_primitive_steps", 5)),
        device=device,
        dtype=torch.float32,
        direction_y_component_weight=float(primitive_cfg.get("exploration_direction_y_component_weight", 2.0)),
        target_min_distance=float(primitive_cfg.get("exploration_target_position_min_distance", 0.2)),
        target_max_distance=float(primitive_cfg.get("exploration_target_position_max_distance", 0.5)),
        target_action_delta_x=float(primitive_cfg.get("exploration_target_position_delta_x", 0.26)),
        target_action_delta_y=float(primitive_cfg.get("exploration_target_position_delta_y", 0.12)),
        same_direction_min_angle_deg=float(primitive_cfg.get("exploration_same_direction_min_angle_deg", -180.0)),
        same_direction_max_angle_deg=float(primitive_cfg.get("exploration_same_direction_max_angle_deg", 180.0)),
        same_direction_min_magnitude=float(primitive_cfg.get("exploration_same_direction_min_magnitude", 0.012)),
        same_direction_max_magnitude=float(primitive_cfg.get("exploration_same_direction_max_magnitude", 0.26)),
        y_aligned_min_angle_deg=float(primitive_cfg.get("exploration_y_aligned_min_angle_deg", 45.0)),
        y_aligned_max_angle_deg=float(primitive_cfg.get("exploration_y_aligned_max_angle_deg", 135.0)),
        y_aligned_min_magnitude=float(primitive_cfg.get("exploration_y_aligned_min_magnitude", 0.012)),
        y_aligned_max_magnitude=float(primitive_cfg.get("exploration_y_aligned_max_magnitude", 0.12)),
        target_position_directional_min_angle_deg=float(primitive_cfg.get("exploration_target_position_directional_min_angle_deg", -180.0)),
        target_position_directional_max_angle_deg=float(primitive_cfg.get("exploration_target_position_directional_max_angle_deg", 180.0)),
        target_position_directional_min_magnitude=float(primitive_cfg.get("exploration_target_position_directional_min_magnitude", 0.2)),
        target_position_directional_max_magnitude=float(primitive_cfg.get("exploration_target_position_directional_max_magnitude", 0.5)),
        target_takeover_steps=int(primitive_cfg.get("exploration_target_position_steps", 5)),
        pre_contact_hit_variant_chance=0.0,
    )
    set_one_hot_primitive(selector, str(primitive_cfg["primitive_name"]))
    return selector


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuously run a single primitive on real robot.")
    parser.add_argument("--config-path", type=str, default="configs/real_configs/primitive_exploration_config.yaml")
    parser.add_argument("--save-path", type=str, default=None)
    args = parser.parse_args()

    params, primitive_cfg = load_runner_config(args.config_path, save_path_override=args.save_path)
    primitive_name = str(primitive_cfg.get("primitive_name", ""))
    if primitive_name not in SUPPORTED_PRIMITIVES:
        raise ValueError(f"primitive_name must be one of {sorted(SUPPORTED_PRIMITIVES)}")

    sim_params = params.get("simulator_params", {})
    if isinstance(sim_params, dict):
        sim_params["wait_for_space_to_start"] = False

    device = torch.device(str(primitive_cfg.get("device", "cpu")))
    eval_env = AirHockeyEnv(params)
    selector = build_selector(primitive_cfg, device=device)
    action_low = torch.as_tensor(eval_env.action_space.low, dtype=torch.float32, device=device).unsqueeze(0)
    action_high = torch.as_tensor(eval_env.action_space.high, dtype=torch.float32, device=device).unsqueeze(0)
    log_every = max(1, int(primitive_cfg.get("log_every", 25)))
    base_action_mode = str(primitive_cfg.get("base_action_mode", "zero"))
    rng = np.random.default_rng(int(params.get("seed", 0)))
    _, prev_puck_pos, _ = extract_primitive_state_tensors(eval_env, device=device)

    print(f"primitive={primitive_name} chance=1.0 continuous mode. key: x=exit")
    step_count = 0
    with NonBlockingConsole() as nbc:
        while True:
            if nbc.get_data() == "x":
                print("Exiting...")
                break

            if base_action_mode == "random":
                std = float(primitive_cfg.get("random_action_std", 0.5))
                proposed_np = np.clip(rng.normal(0.0, std, size=eval_env.action_space.shape), -1.0, 1.0).astype(np.float32)
            else:
                proposed_np = np.zeros(eval_env.action_space.shape, dtype=np.float32)
            action_tensor = torch.as_tensor(proposed_np, dtype=torch.float32, device=device).reshape(1, -1)

            paddle_pos, puck_pos, puck_vel = extract_primitive_state_tensors(eval_env, device=device)
            if torch.all(puck_vel == 0):
                puck_vel = puck_pos - prev_puck_pos
            prev_puck_pos = puck_pos.clone()

            y_alignment_sign = torch.sign(puck_pos[:, 1] - paddle_pos[:, 1])
            action_tensor, stats = selector.apply(
                action_tensor,
                action_low=action_low,
                action_high=action_high,
                y_alignment_sign=y_alignment_sign,
                current_paddle_position=paddle_pos,
                current_puck_position=puck_pos,
                current_puck_velocity=puck_vel,
                return_stats=True,
            )
            env_action = action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
            _, reward, _, _, _ = eval_env.step(env_action)
            step_count += 1

            if step_count % log_every == 0:
                print(
                    f"[primitive_runner] step={step_count} primitive={primitive_name} "
                    f"action=({float(env_action[0]):+.3f},{float(env_action[1]):+.3f}) "
                    f"reward={float(reward):+.3f} primitive_applied={int(stats['primitive_applied_count'])}"
                )


if __name__ == "__main__":
    main()
