"""Transition-hold state machine for the real-world TD3 collector.

Splits the `begin_transition_hold` closure (formerly inlined in
`async_td3_real.collector_process`) and its five mutable nonlocals into two
sibling dataclasses:

- ``RolloutContext`` — per-rollout state that crosses the PolicyRunner /
  orchestrator boundary (last actions + previous puck position used by the
  primitive selector). PolicyRunner mutates these every env step;
  ``TransitionHoldState.begin`` also mutates them when the orchestrator
  triggers a hold mid-episode (actor-sync) or after a reset.

- ``TransitionHoldState`` — the hold counters and the ``begin``/``tick``
  methods.

Behavior is a structural lift of the current source — see
``async_td3_real.py`` L1394–1442, L1758–1776 for the original code paths.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from airhockey import AirHockeyEnv


def normalize_transition_last_action_mode(mode: str) -> str:
    mode_norm = str(mode).strip().lower()
    if mode_norm in {"zero", "executed", "keep"}:
        return mode_norm
    print(
        "[collector_transition] "
        f"unsupported transition_last_action_mode='{mode}', defaulting to 'zero'"
    )
    return "zero"


def request_sim_transition_hold(env: AirHockeyEnv, steps: int, reason: str) -> bool:
    simulator = getattr(env, "simulator", None)
    if simulator is None:
        return False
    begin_fn = getattr(simulator, "begin_transition_hold", None)
    if not callable(begin_fn):
        return False
    try:
        begin_fn(int(steps), reason=str(reason))
        return True
    except Exception as exc:
        print(f"[collector_transition] begin_transition_hold failed for reason={reason}: {exc}")
        return False


@dataclass
class RolloutContext:
    """Mutable per-rollout state shared between PolicyRunner and orchestrator.

    PolicyRunner mutates all three fields every env step (see source
    L1760–1766). ``TransitionHoldState.begin`` re-extracts
    ``previous_puck_position_for_primitive`` from env and may overwrite
    ``last_action_for_policy`` per the configured ``last_action_mode``.
    """

    last_action_for_policy: torch.Tensor
    last_executed_action: torch.Tensor
    previous_puck_position_for_primitive: torch.Tensor


@dataclass
class TransitionHoldState:
    """Counters and trigger logic for the transition-hold state machine.

    The five fields mirror the original locals in
    ``async_td3_real.collector_process``:
        transition_hold_steps_remaining → ``steps_remaining``
        transition_hold_reason          → ``reason``
        transition_hold_events_total    → ``events_total``
        transition_hold_reason_counts   → ``reason_counts``
        transition_last_action_mode     → ``last_action_mode``

    ``steps_total`` is the accumulated counter
    ``transition_hold_steps_total`` (L1473), separate from
    ``steps_remaining``.
    """

    last_action_mode: str = "zero"  # one of "zero" | "executed" | "keep"
    steps_remaining: int = 0
    reason: str = "none"
    events_total: int = 0
    reason_counts: dict = field(default_factory=dict)
    steps_total: int = 0
    log_every_step: bool = False

    def active(self) -> bool:
        return self.steps_remaining > 0

    def tick(self) -> None:
        """Called once per env step from PolicyRunner.

        Reproduces L1758–1759 (``transition_hold_steps_total += 1``) and
        L1768–1776 (decrement remaining + per-step / completion log)
        against the value of ``active()`` captured at step start.
        """
        if self.steps_remaining > 0:
            self.steps_total += 1
            self.steps_remaining = max(0, int(self.steps_remaining) - 1)
            if self.log_every_step:
                print(
                    "[collector_transition] "
                    f"hold_step reason={self.reason} remaining={self.steps_remaining}"
                )
            elif self.steps_remaining == 0:
                print(f"[collector_transition] hold_complete reason={self.reason}")

    def begin(
        self,
        *,
        reason: str,
        hold_steps: int,
        sim_hold: bool,
        env: AirHockeyEnv,
        ctx: RolloutContext,
        primitive_selector,
        extract_primitive_state_tensors: Callable,
        reset_primitive_rollout_state: Callable,
        use_last_action_in_policy_state: bool,
        device: torch.device,
    ) -> None:
        """Mutate hold + rollout context to start a new hold.

        Lifted from L1405–1436 (``begin_transition_hold`` closure).
        Mutates: ``self`` (counters/reason), ``ctx.last_action_for_policy``
        (per ``last_action_mode``), ``ctx.previous_puck_position_for_primitive``
        (re-extracted from env), and primitive-selector rollout state.
        ``sim_hold=True`` matches the closure's ``request_sim_hold=True``
        default — call sites use ``False`` only for the post-actor-sync hold
        (L1889).
        """
        hold_steps = max(int(hold_steps), 0)
        self.events_total += 1
        self.reason_counts[reason] = int(self.reason_counts.get(reason, 0)) + 1
        self.reason = str(reason)
        self.steps_remaining = max(int(self.steps_remaining), hold_steps)
        reset_primitive_rollout_state(primitive_selector)
        _, ctx.previous_puck_position_for_primitive, _ = extract_primitive_state_tensors(
            env, device=device
        )
        if use_last_action_in_policy_state:
            if self.last_action_mode == "zero":
                ctx.last_action_for_policy.zero_()
            elif self.last_action_mode == "executed":
                ctx.last_action_for_policy = ctx.last_executed_action.detach().clone()
        sim_hold_started = False
        if sim_hold and hold_steps > 0:
            sim_hold_started = request_sim_transition_hold(env, steps=hold_steps, reason=reason)
        print(
            "[collector_transition] "
            f"reason={reason} hold_steps={hold_steps} "
            f"collector_hold_remaining={self.steps_remaining} "
            f"sim_hold_started={sim_hold_started} last_action_mode={self.last_action_mode}"
        )
