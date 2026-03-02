import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from airhockey import AirHockeyEnv
from airhockey.airhockey_base import get_observation_by_type
from airhockey.sims.real.multiprocessing import NonBlockingConsole
from scripts.real.rollout_new import Agent


def load_air_hockey_params(config_path: str, save_path_override: str = None) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    air_hockey_params = dict(cfg["air_hockey"])
    air_hockey_params["n_training_steps"] = cfg["n_training_steps"]
    if cfg["algorithm"] == "sac" and "goal" in cfg["air_hockey"]["task"]:
        air_hockey_params["return_goal_obs"] = True
    else:
        air_hockey_params["return_goal_obs"] = False

    if save_path_override is not None:
        air_hockey_params["simulator_params"]["save_path"] = save_path_override
    return air_hockey_params


def infer_policy_dims_from_state_dict(state_dict):
    actor_input_dim = int(state_dict["actor.0.weight"].shape[1])
    action_dim = int(state_dict["actor_mean_head.weight"].shape[0])
    return actor_input_dim, action_dim


class ResetPolicyFSM:
    def __init__(self, env: AirHockeyEnv, rng: np.random.Generator):
        self.env = env
        self.rng = rng
        self.phase = "loop_bottom"
        self.phase_steps = 0
        self.arc_idx = 0
        self.arc_waypoints = np.zeros((1, 2), dtype=np.float32)
        self._new_arc()

    def _meters_delta_to_action(self, delta_xy_m: np.ndarray) -> np.ndarray:
        move_lims = np.array(getattr(self.env.simulator, "move_lims", (0.26, 0.12)), dtype=np.float32)
        move_lims = np.maximum(move_lims, 1e-6)
        action = delta_xy_m / move_lims
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def _new_arc(self) -> None:
        start_side = "left" if self.rng.random() < 0.5 else "right"
        y_edge = self.env.table_y_right - self.env.paddle_radius - 0.015
        x_bot = self.env.table_x_bot - self.env.paddle_radius - 0.01
        arc_rx = 0.13
        arc_ry = y_edge
        x_center = x_bot - arc_rx
        if start_side == "left":
            thetas = np.linspace(-np.pi / 2, np.pi / 2, 36)
        else:
            thetas = np.linspace(np.pi / 2, -np.pi / 2, 36)
        points = np.stack(
            [
                x_center + arc_rx * np.cos(thetas),
                arc_ry * np.sin(thetas),
            ],
            axis=1,
        )
        points[:, 0] = np.clip(
            points[:, 0],
            self.env.table_x_top + self.env.paddle_radius,
            self.env.table_x_bot - self.env.paddle_radius,
        )
        points[:, 1] = np.clip(
            points[:, 1],
            self.env.table_y_left + self.env.paddle_radius,
            self.env.table_y_right - self.env.paddle_radius,
        )
        self.arc_waypoints = points.astype(np.float32)
        self.arc_idx = 0
        self.phase = "loop_bottom"
        self.phase_steps = 0

    def _toward_target(self, paddle_pos: np.ndarray, target: np.ndarray, max_delta_m: float) -> np.ndarray:
        delta = target - paddle_pos
        norm = float(np.linalg.norm(delta))
        if norm <= 1e-8:
            return np.zeros(2, dtype=np.float32)
        scaled_delta = delta * min(1.0, max_delta_m / norm)
        return self._meters_delta_to_action(scaled_delta)

    def step(self, state_info: dict) -> np.ndarray:
        paddle_pos = np.array(state_info["paddles"]["paddle_ego"]["position"], dtype=np.float32)
        puck_pos = np.array(state_info["pucks"][0]["position"], dtype=np.float32)
        paddle_puck_dist = float(np.linalg.norm(paddle_pos - puck_pos))

        if self.phase == "loop_bottom":
            target = self.arc_waypoints[self.arc_idx]
            if float(np.linalg.norm(target - paddle_pos)) < 0.03 and self.arc_idx < len(self.arc_waypoints) - 1:
                self.arc_idx += 1
                target = self.arc_waypoints[self.arc_idx]
            if self.arc_idx >= len(self.arc_waypoints) - 1 and float(np.linalg.norm(target - paddle_pos)) < 0.05:
                self.phase = "strike_up_1"
                self.phase_steps = 5
                return self._meters_delta_to_action(np.array([-0.15, 0.0], dtype=np.float32))
            return self._toward_target(paddle_pos, target, max_delta_m=0.15)

        if self.phase == "strike_up_1":
            self.phase_steps -= 1
            if self.phase_steps <= 0:
                self.phase = "backoff_down"
                self.phase_steps = 8
            return self._meters_delta_to_action(np.array([-0.15, 0.0], dtype=np.float32))

        if self.phase == "backoff_down":
            self.phase_steps -= 1
            if self.phase_steps <= 0:
                self.phase = "wait_for_second_strike"
            return self._meters_delta_to_action(np.array([0.12, 0.0], dtype=np.float32))

        if self.phase == "wait_for_second_strike":
            if paddle_puck_dist <= 0.25:
                self.phase = "strike_up_2"
                self.phase_steps = 5
                return self._meters_delta_to_action(np.array([-0.2, 0.0], dtype=np.float32))
            y_align_delta = float(np.clip(puck_pos[1] - paddle_pos[1], -0.05, 0.05))
            return self._meters_delta_to_action(np.array([0.03, y_align_delta], dtype=np.float32))

        if self.phase == "strike_up_2":
            self.phase_steps -= 1
            if self.phase_steps <= 0:
                self.phase = "prepare_next"
                self.phase_steps = 10
            return self._meters_delta_to_action(np.array([-0.2, 0.0], dtype=np.float32))

        self.phase_steps -= 1
        if self.phase_steps <= 0:
            self._new_arc()
        return self._meters_delta_to_action(np.array([0.12, 0.0], dtype=np.float32))


def build_model_if_requested(args, eval_env):
    if args.model is None:
        return None, False, None
    device = torch.device(args.device)
    state_dict = torch.load(args.model, map_location=device)
    model_obs_dim, model_action_dim = infer_policy_dims_from_state_dict(state_dict)
    model = Agent(eval_env, action_scale=args.action_scale, action_bias=0.0, hidden_size=args.agent_hidden_size)
    model.load_state_dict(state_dict)
    model = model.to(device=device)
    use_last_action = model_obs_dim > eval_env.single_observation_space.shape[0]
    last_action = torch.zeros((1, model_action_dim), dtype=torch.float32, device=device)
    return model, use_last_action, last_action


def compute_failure(
    state: dict,
    table_x_bot: float,
    bottom_margin: float,
    bottom_fail_count: int,
    occluded_fail_count: int,
    counters: dict,
) -> bool:
    puck = state["pucks"][0]
    puck_x = float(puck["position"][0])
    puck_occ = int(np.asarray(puck.get("occluded", 0)).reshape(-1)[0]) > 0
    if puck_x >= (table_x_bot - bottom_margin):
        counters["bottom"] += 1
    else:
        counters["bottom"] = 0
    if puck_occ:
        counters["occ"] += 1
    else:
        counters["occ"] = 0
    return counters["bottom"] >= bottom_fail_count or counters["occ"] >= occluded_fail_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-robot rollout with failure-triggered reset policy.")
    parser.add_argument("--config-path", type=str, default="configs/real_configs/rollout_config.yaml")
    parser.add_argument("--save-path", type=str, default=None, help="Override trajectory save path.")
    parser.add_argument("--model", type=str, default=None, help="Optional juggling model path for normal mode.")
    parser.add_argument("--action-scale", type=float, default=0.2)
    parser.add_argument("--agent-hidden-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--normal-action-x", type=float, default=0.0, help="Fallback normal-mode action x.")
    parser.add_argument("--normal-action-y", type=float, default=0.0, help="Fallback normal-mode action y.")
    parser.add_argument("--bottom-margin", type=float, default=0.12, help="Fail when puck_x > table_x_bot - margin.")
    parser.add_argument("--bottom-fail-count", type=int, default=2)
    parser.add_argument("--occluded-fail-count", type=int, default=6)
    parser.add_argument("--startup-hold-steps", type=int, default=10, help="Hold normal action at zero for startup.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    params = load_air_hockey_params(args.config_path, save_path_override=args.save_path)
    params["max_timesteps"] = max(400, int(params.get("max_timesteps", 300)))
    eval_env = AirHockeyEnv(params)
    model, use_last_action, last_action_for_policy = build_model_if_requested(args, eval_env)
    normal_action = np.array([args.normal_action_x, args.normal_action_y], dtype=np.float32)

    state = eval_env.simulator.get_current_state()
    obs_type = params.get("obs_type", "history")
    obs = get_observation_by_type(
        state,
        obs_type=obs_type,
        puck_history=state["pucks"][0]["history"],
        paddle_history=state["paddles"]["paddle_ego"]["history"],
    )
    rng = np.random.default_rng(int(params.get("seed", 0)))
    reset_fsm = None
    mode = "normal"
    fail_counters = {"bottom": 0, "occ": 0}
    startup_counter = 0

    print("Running real rollout with reset policy. Keys: y=save+reset, q=reset, x=exit")
    with NonBlockingConsole() as nbc:
        while True:
            state = eval_env.simulator.get_current_state()
            fail_now = compute_failure(
                state=state,
                table_x_bot=float(eval_env.table_x_bot),
                bottom_margin=float(args.bottom_margin),
                bottom_fail_count=int(args.bottom_fail_count),
                occluded_fail_count=int(args.occluded_fail_count),
                counters=fail_counters,
            )
            if mode == "normal" and fail_now:
                mode = "reset"
                reset_fsm = ResetPolicyFSM(eval_env, rng)
                if args.verbose:
                    print(
                        f"[reset] trigger bottom_count={fail_counters['bottom']} occ_count={fail_counters['occ']} "
                        f"puck={state['pucks'][0]['position']}"
                    )

            if mode == "reset":
                action = reset_fsm.step(state)
                if reset_fsm.phase == "loop_bottom" and reset_fsm.arc_idx == 0 and reset_fsm.phase_steps == 0:
                    mode = "normal"
                    fail_counters = {"bottom": 0, "occ": 0}
                    if args.verbose:
                        print("[reset] cycle complete; returning to normal mode")
            else:
                if model is None:
                    action = np.copy(normal_action)
                else:
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
                    policy_obs = obs_t
                    if use_last_action:
                        policy_obs = torch.cat([policy_obs, last_action_for_policy], dim=-1)
                    action = model(policy_obs).detach().cpu().numpy().squeeze()
                    action = action / float(model.action_scale.item())
                    action = np.clip(action, -1.0, 1.0)
                    if use_last_action:
                        last_action_for_policy = torch.tensor(action, dtype=torch.float32, device=args.device).unsqueeze(0)
                if startup_counter < int(args.startup_hold_steps):
                    action = np.zeros(2, dtype=np.float32)
                    startup_counter += 1

            obs, reward, terminated, truncated, info = eval_env.step(action)
            if args.verbose:
                puck = state["pucks"][0]
                print(
                    f"mode={mode} phase={(reset_fsm.phase if reset_fsm is not None else '-'):<18} "
                    f"action=({float(action[0]):+.3f},{float(action[1]):+.3f}) "
                    f"puck=({float(puck['position'][0]):+.3f},{float(puck['position'][1]):+.3f}) "
                    f"occ={int(np.asarray(puck.get('occluded', 0)).reshape(-1)[0])} "
                    f"rew={float(reward):+.3f} done={terminated or truncated}"
                )

            state = eval_env.simulator.get_current_state()
            obs = get_observation_by_type(
                state,
                obs_type=obs_type,
                puck_history=state["pucks"][0]["history"],
                paddle_history=state["paddles"]["paddle_ego"]["history"],
            )

            key = nbc.get_data()
            if key == "y":
                print("Saving trajectory and resetting...")
                eval_env.reset(seed=None, write_traj=True)
                mode = "normal"
                reset_fsm = None
                fail_counters = {"bottom": 0, "occ": 0}
                startup_counter = 0
                if use_last_action:
                    last_action_for_policy.zero_()
            elif key == "q":
                print("Resetting without saving...")
                eval_env.reset(seed=None, write_traj=False)
                mode = "normal"
                reset_fsm = None
                fail_counters = {"bottom": 0, "occ": 0}
                startup_counter = 0
                if use_last_action:
                    last_action_for_policy.zero_()
            elif key == "x":
                print("Exiting...")
                break
