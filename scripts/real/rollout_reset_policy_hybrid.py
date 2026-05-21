"""Hybrid reset FSM: programmatic loop + first burst, then policy handoff.

Drop-in replacement for ``ResetPolicyFSM``. Reuses phases 1–3 of the legacy
FSM (``goto_start`` → ``edge_loop`` → ``upward_burst`` →
``post_first_upward_check``), then — instead of running the legacy
``wait_for_puck`` / programmatic ``strike`` retry loop — hands control to a
frozen juggle policy (default: ``latest_models/canonical/hist2_motion0_v2/model.pth``)
for the second hit. Success = puck rises above the midline AFTER first
descending (proves the policy actually struck it). Failure = puck stays
below the paddle for ``_RESET_BELOW_PADDLE_MAX_STEPS`` consecutive frames,
or handoff exceeds ``_RESET_HANDOFF_MAX_STEPS``. Failure → restart from
``goto_start``, capped at ``_RESET_MAX_RESTART_ATTEMPTS``; the cap falls
through to ``done_reason="hard_reset_required"`` (same contract as the
legacy stage-2 ceiling, see ``real_reset_runner.py:248``).

Wired in from ``async_td3_real.py`` via a single-symbol code toggle; not a
config field. **Currently opt-in only** — the canonical real-world default
is ``ResetPolicyFSM`` (legacy programmatic strike). To enable this hybrid,
change ``_DEFAULT_RESET_FSM_CLS`` in ``async_td3_real.py`` (or pass
``--use-hybrid-fsm`` to ``rollout_reset_policy_real.py`` for isolated
testing).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from airhockey import AirHockeyEnv
from scripts.real.rollout_reset_policy_real import ResetPolicyFSM
from scripts.td3.helper.real_td3_runtime import (
    _load_train_args,
    augment_policy_observation,
    build_policy_env_view,
    deterministic_actor_action,
)
from scripts.td3.deterministic_agent import DeterministicAgent


# Tunables for the hybrid reset. Kept as module-level constants (not Args /
# YAML) so the canonical recipe is the only thing checked into git; if you
# need to retune, edit here.
_RESET_JUGGLE_ACTOR_PATH: str = "latest_models/canonical/hist2_motion0_v2/model.pth"
_RESET_BELOW_PADDLE_MAX_STEPS: int = 15   # ~0.5 s @ 30 Hz before declaring failure
_RESET_HANDOFF_MAX_STEPS: int = 90        # ~3 s @ 30 Hz absolute cap on handoff duration
_RESET_MAX_RESTART_ATTEMPTS: int = 3      # restart cycles before falling through to hard reset
# First-burst gate ("did the burst connect?"): quarter-line. Just verifies
# the puck rose into the bottom-half-and-above region — does NOT require it
# to reach the midline (the legacy FSM's shared gate at 0.5 is opportunistic
# and expected to fail most of the time, which doesn't fit a hand-off gate).
_RESET_FIRST_BURST_SUCCESS_HEIGHT_PROP: float = 0.25
# Handoff success gate (descent-then-ascent): midline. Used inside
# _step_policy_handoff via _line_tcp_x_from_bottom_proportion(...) directly,
# independent of the base FSM's shared_success_threshold field.
_RESET_HANDOFF_SUCCESS_HEIGHT_PROP: float = 0.5


# Process-level singleton: load the juggle actor once per training process.
_juggle_actor_singleton: DeterministicAgent | None = None
_juggle_actor_device: torch.device | None = None
_juggle_uses_last_action: bool = False


def build_juggle_actor(device: torch.device | None = None) -> tuple[DeterministicAgent, torch.device, bool]:
    """Load the frozen juggle policy from ``_RESET_JUGGLE_ACTOR_PATH``.

    Architecture is read from ``args.yaml`` next to the checkpoint — the
    juggle policy's hidden-layer shape may differ from the current training
    run's, so do NOT reuse the orchestrator's ``train_args``.

    Returns ``(actor, device, use_last_action_in_policy_state)``. Cached as a
    process-level singleton so successive ``ResetPolicyHybridFSM``
    instantiations share one frozen actor.
    """
    global _juggle_actor_singleton, _juggle_actor_device, _juggle_uses_last_action
    if _juggle_actor_singleton is not None:
        return _juggle_actor_singleton, _juggle_actor_device, _juggle_uses_last_action

    chosen_device = device if device is not None else torch.device("cpu")
    args_yaml_path = str(Path(_RESET_JUGGLE_ACTOR_PATH).parent / "args.yaml")
    juggle_train_args = _load_train_args(args_yaml_path)
    # ``model.pth`` is a raw actor state_dict (flat OrderedDict of weights),
    # not a training_state container — the orchestrator's
    # ``_load_training_state_checkpoint`` is the wrong loader here. Use
    # ``torch.load`` directly. The actor uses ``ResidualMLPTrunk`` (see
    # ``scripts/td3/deterministic_agent.py``), so the first
    # Linear layer lives at ``actor.blocks.0.units.0.0.weight``.
    actor_state = torch.load(_RESET_JUGGLE_ACTOR_PATH, map_location="cpu", weights_only=False)
    if not isinstance(actor_state, dict):
        raise TypeError(
            f"Expected actor state_dict at {_RESET_JUGGLE_ACTOR_PATH}, got {type(actor_state).__name__}."
        )
    actor_input_dim = int(actor_state["actor.blocks.0.units.0.0.weight"].shape[1])
    act_dim = int(actor_state["actor_mean_head.weight"].shape[0])

    policy_env_view = build_policy_env_view(actor_input_dim, act_dim)
    actor = DeterministicAgent(
        policy_env_view,
        action_scale=1.0,
        action_bias=0.0,
        hidden_layer_size=juggle_train_args.agent_hidden_layer_size,
        num_hidden_layers=juggle_train_args.agent_num_hidden_layers,
    ).to(chosen_device)
    actor.load_state_dict(actor_state, strict=True)
    actor.eval()

    _juggle_actor_singleton = actor
    _juggle_actor_device = chosen_device
    _juggle_uses_last_action = bool(juggle_train_args.use_last_action_in_policy_state)
    print(
        f"[reset_fsm_hybrid] loaded juggle actor from {_RESET_JUGGLE_ACTOR_PATH} "
        f"(actor_input_dim={actor_input_dim}, act_dim={act_dim}, "
        f"hidden={juggle_train_args.agent_hidden_layer_size}x{juggle_train_args.agent_num_hidden_layers}, "
        f"use_last_action={_juggle_uses_last_action})"
    )
    return actor, chosen_device, _juggle_uses_last_action


class ResetPolicyHybridFSM(ResetPolicyFSM):
    """ResetPolicyFSM with the second hit delegated to a frozen juggle policy."""

    def __init__(
        self,
        env: AirHockeyEnv,
        rng: np.random.Generator,
        *,
        juggle_actor: DeterministicAgent | None = None,
        juggle_device: torch.device | None = None,
        use_last_action_in_policy_state: bool | None = None,
        **base_kwargs,
    ) -> None:
        # The shared_success threshold inherited by the base FSM controls
        # the post_first_upward_check gate (the "did the burst connect?"
        # test). Lower it to the quarter-line — see comment on
        # _RESET_FIRST_BURST_SUCCESS_HEIGHT_PROP for why the legacy 0.5
        # value would gate out almost every handoff.
        base_kwargs["shared_success_threshold_proportion_from_bottom"] = (
            _RESET_FIRST_BURST_SUCCESS_HEIGHT_PROP
        )
        super().__init__(env, rng, **base_kwargs)

        # Resolve the juggle actor: pre-built (preferred — single load per
        # process) or lazily build the singleton on first construction.
        if juggle_actor is None:
            actor, device, uses_last_action = build_juggle_actor(juggle_device)
        else:
            actor = juggle_actor
            device = juggle_device if juggle_device is not None else torch.device("cpu")
            uses_last_action = (
                bool(use_last_action_in_policy_state)
                if use_last_action_in_policy_state is not None
                else _juggle_uses_last_action
            )
        self._juggle_actor = actor
        self._juggle_device = device
        self._juggle_uses_last_action = bool(uses_last_action)

        act_dim = int(env.single_action_space.shape[0])
        self._juggle_act_dim = act_dim
        self._last_action_t = torch.zeros((1, act_dim), dtype=torch.float32, device=device)

        self.restart_attempts = 0
        self._below_paddle_streak = 0
        self._handoff_steps = 0
        self._handoff_saw_descent = False

    # ------------------------------------------------------------------
    # Phase dispatch.
    # ------------------------------------------------------------------

    def step(self, state_info: dict) -> np.ndarray:
        if self.done:
            return np.zeros(2, dtype=np.float32)
        if self.phase == "policy_handoff":
            self.total_steps += 1
            self._maybe_log_robot_normal_force()
            return self._step_policy_handoff(state_info)
        return super().step(state_info)

    # ------------------------------------------------------------------
    # Override the post-burst dispatch: stage-1 pass → policy_handoff
    # (instead of marking success); stage-1 fail → restart (instead of
    # entering wait_for_puck). Stage-2 paths are unreachable here.
    # ------------------------------------------------------------------

    def _finalize_post_upward_window(self, state_info: dict) -> np.ndarray:
        if self._pending_window_finalize is None:
            return np.zeros(2, dtype=np.float32)
        finalize = dict(self._pending_window_finalize)
        if finalize["kind"] != "first":
            # Defensive: hybrid never enters post_second_upward_check.
            self._set_terminal_reason("unexpected_post_second_check")
            return np.zeros(2, dtype=np.float32)
        self._pending_window_finalize = None
        self._clear_post_upward_window_target()
        # The base FSM gates on (height >= 1 AND off_wall >= 5). For the
        # hybrid first-burst test we only care about height — bursts often
        # send the puck up along a wall, which is fine for the policy to
        # take over on. Recompute pass on height alone so off-wall doesn't
        # veto otherwise-effective bursts.
        height_steps = int(finalize.get("height_steps", 0))
        required_height_steps = int(finalize.get("required_height_steps", 1))
        height_passed = height_steps >= max(1, required_height_steps)
        self._log_phase_check(
            state_info,
            phase_name="stage1_upward",
            check_name="hybrid_first_burst_height_only",
            passed=bool(height_passed),
            window_stats=finalize,
        )
        if height_passed:
            self._enter_policy_handoff()
            return np.zeros(2, dtype=np.float32)
        self._restart_round("first_burst_height_check_failed")
        return np.zeros(2, dtype=np.float32)

    def _enter_policy_handoff(self) -> None:
        self.phase = "policy_handoff"
        self.phase_steps = 0
        self._handoff_steps = 0
        self._below_paddle_streak = 0
        self._handoff_saw_descent = False
        # Fresh last-action seed for the juggle actor; the policy doesn't
        # see anything from the programmatic burst phase.
        self._last_action_t.zero_()
        print(
            f"[reset_fsm_hybrid] policy_handoff_start total_steps={self.total_steps} "
            f"max_steps={_RESET_HANDOFF_MAX_STEPS} below_paddle_max={_RESET_BELOW_PADDLE_MAX_STEPS}"
        )

    # ------------------------------------------------------------------
    # Restart-cycle accounting + hard-reset fallthrough.
    # ------------------------------------------------------------------

    def _restart_round(self, reason: str) -> None:
        if self.restart_attempts >= _RESET_MAX_RESTART_ATTEMPTS:
            print(
                f"[reset_fsm_hybrid] hybrid_max_restart_attempts_reached "
                f"count={self.restart_attempts} limit={_RESET_MAX_RESTART_ATTEMPTS} reason={reason} "
                f"-> hard reset required"
            )
            self._set_terminal_reason("hard_reset_required")
            return
        self.restart_attempts += 1
        self._below_paddle_streak = 0
        self._handoff_steps = 0
        self._handoff_saw_descent = False
        super()._restart_round(reason)

    # ------------------------------------------------------------------
    # Policy handoff step.
    # ------------------------------------------------------------------

    def _step_policy_handoff(self, state_info: dict) -> np.ndarray:
        self._handoff_steps += 1

        puck_world_x = float(state_info["pucks"][0]["position"][0])
        paddle_world_x = float(state_info["paddles"]["paddle_ego"]["position"][0])

        # World-frame x grows downward: puck_x >= paddle_x means the puck
        # has descended past the paddle. Same semantics the base FSM uses
        # at rollout_reset_policy_real.py:777-779, but accumulated rather
        # than tripped on a single frame.
        puck_below_paddle = puck_world_x >= paddle_world_x
        if puck_below_paddle:
            self._below_paddle_streak += 1
        else:
            self._below_paddle_streak = 0
        if self._below_paddle_streak >= _RESET_BELOW_PADDLE_MAX_STEPS:
            print(
                f"[reset_fsm_hybrid] policy_handoff_puck_below_paddle "
                f"streak={self._below_paddle_streak} limit={_RESET_BELOW_PADDLE_MAX_STEPS} "
                f"handoff_steps={self._handoff_steps}"
            )
            self._restart_round("policy_handoff_puck_below_paddle")
            return np.zeros(2, dtype=np.float32)

        # Descent-then-ascent gate: only call success once the puck has
        # demonstrably fallen below the midline AND climbed back above it,
        # so a long-tailed first-burst trajectory can't false-positive.
        # Use the handoff threshold (midline) directly, NOT the base FSM's
        # shared_success_height — that field is set to the (lower) first-
        # burst gate so the post_first_upward_check passes more reliably.
        # No off-wall conjunction here: the puck having crossed the midline
        # twice (down then up) is itself strong evidence the policy hit it,
        # and demanding the puck also be near centerline at the success
        # frame causes legitimate wall-skimming returns to time out.
        if not self._puck_is_occluded(state_info):
            puck_pos_tcp = self._get_puck_pos(state_info)
            puck_tcp_x = float(puck_pos_tcp[0])
            success_x = self._line_tcp_x_from_bottom_proportion(
                _RESET_HANDOFF_SUCCESS_HEIGHT_PROP
            )
            if puck_tcp_x > success_x:
                self._handoff_saw_descent = True
            elif self._handoff_saw_descent and puck_tcp_x <= success_x:
                self._mark_success(state_info, stage="policy_handoff")
                return np.zeros(2, dtype=np.float32)

        if self._handoff_steps >= _RESET_HANDOFF_MAX_STEPS:
            print(
                f"[reset_fsm_hybrid] policy_handoff_timeout "
                f"steps={self._handoff_steps} limit={_RESET_HANDOFF_MAX_STEPS} "
                f"saw_descent={int(self._handoff_saw_descent)}"
            )
            self._restart_round("policy_handoff_timeout")
            return np.zeros(2, dtype=np.float32)

        # Query the frozen juggle actor.
        obs_np, _ = self.env.get_current_state()
        obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32, device=self._juggle_device).unsqueeze(0)
        policy_obs = augment_policy_observation(
            obs_tensor,
            self._last_action_t,
            self._juggle_uses_last_action,
        )
        with torch.no_grad():
            action_t = deterministic_actor_action(self._juggle_actor, policy_obs)
        action_t = torch.clamp(action_t, -1.0, 1.0)
        self._last_action_t = action_t.detach()
        action = action_t.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
        return action
