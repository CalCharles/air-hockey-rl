"""CUDA-graph-captured TD3 critic / actor updates, replay sampling included.

Why this exists
---------------
The TD3 networks in this project are tiny (2 residual blocks, 64 wide). A
single critic forward+backward issues ~250 CUDA kernels but only ~0.8 ms of
GPU work; eager PyTorch spends ~7 ms of CPU time dispatching those kernels,
and the PER/uniform minibatch sampling adds another ~60 small ops (~1.2 ms)
per update. The per-episode training cycle (25 critic + 6 actor updates)
therefore cost ~560 ms of pure CPU dispatch, which dominated wall-clock.

`GraphedTD3Update` captures, per critic update, the whole chain
    sample minibatch (PER proportional + uniform, success + failure buffers)
    -> target computation -> N-critic loss -> backward -> Adam step
    -> PER priority write-back -> metrics
into one `torch.cuda.CUDAGraph`, and likewise the actor update (uniform
sample -> actor loss -> backward -> Adam). A training cycle is then ~31
`graph.replay()` calls plus a handful of scalar `fill_`s, and runs at the
GPU-time floor.

The same class runs the identical math eagerly (`use_graph=False`) so CPU
training / debugging shares one implementation with the graphed path.

Correctness notes
-----------------
* Sampling reads the replay buffers' *full* storage tensors (static
  addresses) with a mask `arange < size` for filled slots, so shapes are
  static while the buffer fills. Empty slots get probability 0 and are never
  drawn; uniform draws are `floor(rand * size)`. IS weights are normalised by
  their max within each buffer's PER chunk, exactly as the legacy sampler.
* Graphs are captured lazily per (success_count, failure_count) batch
  composition (at most a few variants: one buffer empty vs both filled).
* Warmup iterations are required before capture (Adam state allocation,
  cuBLAS workspaces). They would mutate weights, optimizer state, priorities
  and the CUDA RNG, so all of these are snapshotted and restored **in place**
  afterwards (the graph holds tensor addresses; nothing may be re-allocated).
* Adam is constructed with `capturable=True` (`step` lives on the GPU;
  `load_state_dict` handles older checkpoints).
* Random draws (`torch.randn_like` target-smoothing noise, CQL random actions,
  multinomial / uniform sampling) happen inside the graph. CUDA graphs
  register the default CUDA generator, so every replay advances the Philox
  offset and draws fresh randomness.
* During the actor update the critic parameters are temporarily set to
  `requires_grad=False` so no critic grads are produced (the eager loop
  discarded them at the next `zero_grad()` anyway). Gradients still flow
  through the critic to the action input.
* Age-decayed priorities (`priority_age_decay > 0`) are computed in-graph
  from the buffer's `size` / `position` scalars (synced before each replay).
  REDQ target subsets (`target_critic_subset_size < num_critics`) need a
  host-side permutation per update and are only supported on the eager path.
* Autograd records the current stream on each parameter's AccumulateGrad
  node when it is first created. If a parameter was used by an autograd-
  tracked op (e.g. `p.clone()` without `detach()`) before capture, that node
  lives on the legacy stream and backward inside capture fails with
  "operation would make the legacy stream depend on a capturing blocking
  stream". The trainer never does this; `_capture_on_device` runs
  `gc.collect()` so dead nodes are recreated on the capture stream. Keep any
  parameter inspection under `torch.no_grad()` / `.detach()`.
"""

from __future__ import annotations

import gc
import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn


def h_transform(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x


def h_inverse(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    abs_x = torch.abs(x)
    inner = 1 + 4 * eps * (abs_x + 1 + eps)
    sqrt_inner = torch.sqrt(inner)
    quotient = (sqrt_inner - 1) / (2 * eps)
    return torch.sign(x) * (quotient**2 - 1)


def deterministic_actor_action(actor: nn.Module, policy_obs: torch.Tensor) -> torch.Tensor:
    if hasattr(actor, "get_action_mean_and_logstd"):
        action_mean, _ = actor.get_action_mean_and_logstd(policy_obs)
        return torch.tanh(action_mean) * actor.action_scale + actor.action_bias
    if hasattr(actor, "get_action"):
        return actor.get_action(policy_obs)
    raise TypeError(f"Unsupported actor type for deterministic action: {type(actor)}")


def _snapshot_optimizer_state(optimizer: torch.optim.Optimizer) -> Dict[int, Dict[str, torch.Tensor]]:
    snap: Dict[int, Dict[str, torch.Tensor]] = {}
    for param, state in optimizer.state.items():
        snap[id(param)] = {
            k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in state.items()
        }
    return snap


def _restore_optimizer_state(
    optimizer: torch.optim.Optimizer, snap: Dict[int, Dict[str, torch.Tensor]]
) -> None:
    """Restore optimizer state IN PLACE (tensor addresses must survive)."""
    for param, state in optimizer.state.items():
        saved = snap.get(id(param))
        for k, v in state.items():
            if not torch.is_tensor(v):
                continue
            if saved is not None and k in saved and torch.is_tensor(saved[k]):
                v.copy_(saved[k])
            else:
                # State did not exist before warmup (fresh optimizer): reset to
                # the values Adam would lazily initialise with.
                v.zero_()


class _BufferView:
    """Static handles into one replay buffer used by the in-graph sampler."""

    def __init__(self, rb, device: torch.device) -> None:
        self.rb = rb
        self.prioritized = hasattr(rb, "priorities")
        self.size_t = torch.zeros((), dtype=torch.float32, device=device)
        self.position_t = torch.zeros((), dtype=torch.float32, device=device)
        self.arange = torch.arange(int(rb.buffer_size), dtype=torch.float32, device=device)

    def sync_size(self) -> None:
        self.size_t.fill_(float(self.rb.size))
        self.position_t.fill_(float(self.rb.position))


class GraphedTD3Update:
    """Static-shape critic + actor update with in-graph replay sampling."""

    def __init__(
        self,
        *,
        actor: nn.Module,
        actor_target: nn.Module,
        qfs: Sequence[nn.Module],
        qfs_target: Sequence[nn.Module],
        q_optimizer: torch.optim.Optimizer,
        actor_optimizer: torch.optim.Optimizer,
        success_rb,
        failure_rb,
        batch_size: int,
        obs_dim: int,
        act_dim: int,
        device: torch.device | str,
        gamma: float,
        tau: float,
        policy_noise: float,
        noise_clip: float,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        h_transform_eps: float,
        use_last_action_in_policy_state: bool,
        per_enabled: bool,
        per_eps: float,
        critic_per_fraction: float,
        cql_alpha: float = 0.0,
        cql_n_random: int = 10,
        target_critic_subset_size: int | None = None,
        use_graph: bool = True,
        compile_update: bool = True,
    ) -> None:
        self.actor = actor
        self.actor_target = actor_target
        self.qfs = list(qfs)
        self.qfs_target = list(qfs_target)
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.num_critics = len(self.qfs)
        self.batch_size = int(batch_size)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.device = torch.device(device)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.policy_noise = float(policy_noise)
        self.noise_clip = float(noise_clip)
        self.action_low = action_low.to(self.device)
        self.action_high = action_high.to(self.device)
        self.h_eps = float(h_transform_eps)
        self.use_last_action = bool(use_last_action_in_policy_state)
        self.per_enabled = bool(per_enabled)
        self.per_eps = float(per_eps)
        self.critic_per_fraction = float(critic_per_fraction)
        self.cql_alpha = float(cql_alpha)
        self.cql_n_random = int(cql_n_random)
        self.target_subset = (
            int(target_critic_subset_size)
            if target_critic_subset_size is not None and int(target_critic_subset_size) < self.num_critics
            else None
        )
        self.buffers = [_BufferView(success_rb, self.device), _BufferView(failure_rb, self.device)]
        self.use_graph = (
            bool(use_graph)
            and self.device.type == "cuda"
            and self.target_subset is None
        )
        self.beta_t = torch.zeros((), dtype=torch.float32, device=self.device)
        # torch.compile (inductor) fuses the elementwise / LayerNorm chains of
        # the loss forward+backward: 627 -> 369 kernels, 2.1 -> 1.4 ms per
        # critic update on a Quadro RTX 6000. Only used together with graphs;
        # falls back to the plain math if compilation fails at capture time.
        self.compile_update = bool(compile_update) and self.use_graph
        self._critic_core_fn = self._critic_core
        self._actor_core_fn = self._actor_core
        if self.compile_update:
            try:
                self._critic_core_fn = torch.compile(self._critic_core, dynamic=False)
                self._actor_core_fn = torch.compile(self._actor_core, dynamic=False)
            except Exception as exc:  # pragma: no cover - toolchain dependent
                print(f"torch.compile unavailable for TD3 update ({exc}); using uncompiled graphs.")
                self.compile_update = False
        if self.device.type == "cuda":
            # Graph capture + autograd must run with this device current;
            # otherwise the backward pass touches the *default* device's
            # legacy stream and capture aborts (seen when training on cuda:1+).
            torch.cuda.set_device(self.device)

        # Outputs of the most recent update (GPU tensors; read lazily).
        self.critic_out: Dict[str, torch.Tensor] = {}
        self.actor_out: Dict[str, torch.Tensor] = {}

        # Lazily captured graphs keyed by batch composition.
        self._critic_graphs: Dict[Tuple[int, int], Tuple[torch.cuda.CUDAGraph, Dict[str, torch.Tensor]]] = {}
        self._actor_graphs: Dict[Tuple[int, int], Tuple[torch.cuda.CUDAGraph, Dict[str, torch.Tensor]]] = {}
        self._capture_stream = torch.cuda.Stream(device=self.device) if self.use_graph else None

        # Polyak parameter lists (fixed for the run).
        self._polyak_src: List[torch.Tensor] = [p for p in self.actor.parameters()]
        self._polyak_tgt: List[torch.Tensor] = [p for p in self.actor_target.parameters()]
        for q, qt in zip(self.qfs, self.qfs_target):
            self._polyak_src.extend(q.parameters())
            self._polyak_tgt.extend(qt.parameters())
        assert len(self._polyak_src) == len(self._polyak_tgt)

    # ------------------------------------------------------------- sampling
    def _split_counts(self, count: int) -> Tuple[int, int]:
        if not self.per_enabled:
            return 0, count
        per_count = int(round(count * self.critic_per_fraction))
        per_count = min(max(per_count, 0), count)
        return per_count, count - per_count

    def _uniform_indices(self, view: _BufferView, count: int) -> torch.Tensor:
        u = torch.rand((count,), device=self.device) * view.size_t
        return torch.minimum(u.long(), (view.size_t - 1.0).clamp_min(0.0).long())

    def _per_indices(self, view: _BufferView, count: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rb = view.rb
        priorities = rb.priorities
        mask = view.arange < view.size_t
        valid_priorities = priorities.clamp_min(rb.priority_eps)
        if rb.age_decay > 0.0:
            # Age in slots since written, mirroring TD3PrioritizedReplayBuffer.sample():
            # (size-1-i) while filling, (position-1-i) mod capacity once wrapped.
            # The weight multiplies the priority BEFORE the alpha exponent.
            capacity = float(rb.buffer_size)
            ages_filling = (view.size_t - 1.0) - view.arange
            ages_wrapped = torch.remainder(view.position_t - 1.0 - view.arange, capacity)
            ages = torch.where(view.size_t < capacity, ages_filling, ages_wrapped).clamp_min(0.0)
            valid_priorities = valid_priorities * torch.exp(-rb.age_decay * ages)
        scaled = valid_priorities.pow(rb.alpha) * mask
        scaled_sum = scaled.sum()
        sum_ok = torch.isfinite(scaled_sum) & (scaled_sum > 0.0)
        probs = torch.where(
            sum_ok,
            scaled / scaled_sum.clamp_min(1e-30),
            mask.to(scaled.dtype) / view.size_t.clamp_min(1.0),
        )
        indices = torch.multinomial(probs, num_samples=count, replacement=True)
        sample_probs = probs[indices].clamp_min(1e-12)
        weights = (view.size_t * sample_probs).pow(-self.beta_t)
        weights = weights / weights.max().clamp_min(1e-12)
        return indices, weights

    def _sample_critic(self, counts: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        chunks = []
        per_slots: List[Tuple[_BufferView, torch.Tensor, int, int]] = []
        offset = 0
        for view, count in zip(self.buffers, counts):
            if count <= 0:
                continue
            per_count, uni_count = self._split_counts(count) if view.prioritized else (0, count)
            idx_parts = []
            w_parts = []
            if per_count > 0:
                per_idx, per_w = self._per_indices(view, per_count)
                idx_parts.append(per_idx)
                w_parts.append(per_w)
                per_slots.append((view, per_idx, offset, offset + per_count))
            if uni_count > 0:
                idx_parts.append(self._uniform_indices(view, uni_count))
                w_parts.append(torch.ones((uni_count,), dtype=torch.float32, device=self.device))
            indices = idx_parts[0] if len(idx_parts) == 1 else torch.cat(idx_parts)
            weights = w_parts[0] if len(w_parts) == 1 else torch.cat(w_parts)
            rb = view.rb
            chunks.append(
                {
                    "observations": rb.observations[indices],
                    "next_observations": rb.next_observations[indices],
                    "actions": rb.actions[indices],
                    "prev_actions": rb.prev_actions[indices],
                    "rewards": rb.rewards[indices],
                    "dones": rb.dones[indices],
                    "weights": weights,
                }
            )
            offset += count
        if len(chunks) == 1:
            data = chunks[0]
        else:
            data = {key: torch.cat([c[key] for c in chunks], dim=0) for key in chunks[0]}
        data["_per_slots"] = per_slots  # type: ignore[assignment]
        return data

    def _sample_actor(self, counts: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_parts = []
        prev_parts = []
        for view, count in zip(self.buffers, counts):
            if count <= 0:
                continue
            indices = self._uniform_indices(view, count)
            obs_parts.append(view.rb.observations[indices])
            prev_parts.append(view.rb.prev_actions[indices])
        if len(obs_parts) == 1:
            return obs_parts[0], prev_parts[0]
        return torch.cat(obs_parts, dim=0), torch.cat(prev_parts, dim=0)

    # ------------------------------------------------------------------ math
    def _policy_obs(self, obs: torch.Tensor, prev_actions: torch.Tensor) -> torch.Tensor:
        if not self.use_last_action:
            return obs
        return torch.cat([obs, prev_actions], dim=-1)

    def _critic_core(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        actions: torch.Tensor,
        prev_actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor,
        noise: torch.Tensor,
        cql_random_actions: torch.Tensor | None,
    ):
        """Target computation + N-critic loss. Pure tensor function (random
        draws are passed in) so it can be torch.compile'd."""
        with torch.no_grad():
            next_prev_actions = actions * (1.0 - dones.unsqueeze(-1))
            next_policy_obs = self._policy_obs(next_obs, next_prev_actions)
            target_next_action = deterministic_actor_action(self.actor_target, next_policy_obs)
            target_next_action = torch.clamp(
                target_next_action + noise, self.action_low, self.action_high
            )
            if self.target_subset is None:
                target_critics = self.qfs_target
            else:
                subset = torch.randperm(self.num_critics)[: self.target_subset].tolist()
                target_critics = [self.qfs_target[i] for i in subset]
            next_q_h_list = [qt(next_obs, target_next_action) for qt in target_critics]
            if len(next_q_h_list) == 1:
                min_next_q_h = next_q_h_list[0]
            elif len(next_q_h_list) == 2:
                min_next_q_h = torch.min(next_q_h_list[0], next_q_h_list[1])
            else:
                min_next_q_h = torch.stack(next_q_h_list, dim=0).min(dim=0).values
            min_next_q = h_inverse(min_next_q_h, eps=self.h_eps).view(-1)
            bellman_target_original = rewards + (1.0 - dones) * self.gamma * min_next_q
            next_q_value_h = h_transform(bellman_target_original, eps=self.h_eps)

            cql_terms = None
            if cql_random_actions is not None:
                cql_policy_action = deterministic_actor_action(
                    self.actor, self._policy_obs(obs, prev_actions)
                )
                bsz = obs.shape[0]
                obs_repeat = (
                    obs.unsqueeze(0).expand(self.cql_n_random, -1, -1).reshape(self.cql_n_random * bsz, -1)
                )
                cql_terms = (cql_random_actions, obs_repeat, cql_policy_action, bsz)

        qi_err_list = []
        qi_loss_list = []
        q1_h = None
        for q in self.qfs:
            qi_h = q(obs, actions)
            if q1_h is None:
                q1_h = qi_h
            qi_err = qi_h.view(-1) - next_q_value_h
            qi_err_list.append(qi_err)
            loss_i = (weights * qi_err.pow(2)).mean()
            if cql_terms is not None:
                random_actions, obs_repeat, cql_policy_action, bsz = cql_terms
                q_rand_h = q(obs_repeat, random_actions).view(self.cql_n_random, bsz)
                q_pi_h = q(obs, cql_policy_action).view(-1)
                logsumexp = torch.logsumexp(q_rand_h, dim=0) - math.log(float(self.cql_n_random))
                loss_i = loss_i + self.cql_alpha * (logsumexp - q_pi_h).mean()
            qi_loss_list.append(loss_i)
        q_total_loss = sum(qi_loss_list)
        priority_td_error = (sum(e.abs() for e in qi_err_list) / self.num_critics).detach()
        return (
            q_total_loss,
            priority_td_error,
            q1_h.detach().mean(),
            bellman_target_original.mean(),
            next_q_value_h.mean(),
        )

    def _critic_step(self, counts: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        data = self._sample_critic(counts)
        actions = data["actions"]
        noise = torch.clamp(
            torch.randn_like(actions) * self.policy_noise, -self.noise_clip, self.noise_clip
        )
        cql_random_actions = None
        if self.cql_alpha > 0.0:
            cql_random_actions = torch.empty(
                self.cql_n_random * actions.shape[0], self.act_dim, device=actions.device
            ).uniform_(-1.0, 1.0)
        q_total_loss, priority_td_error, q1_mean, bellman_mean, next_q_h_mean = self._critic_core_fn(
            data["observations"],
            data["next_observations"],
            actions,
            data["prev_actions"],
            data["rewards"],
            data["dones"],
            data["weights"],
            noise,
            cql_random_actions,
        )
        self.q_optimizer.zero_grad(set_to_none=True)
        q_total_loss.backward()
        self.q_optimizer.step()

        with torch.no_grad():
            # PER priority write-back: mean |TD error| across critics (+ eps).
            for view, per_idx, start, end in data["_per_slots"]:  # type: ignore[index]
                rb = view.rb
                new_priorities = (priority_td_error[start:end] + self.per_eps).clamp_min(rb.priority_eps)
                rb.priorities[per_idx] = new_priorities
                torch.maximum(rb._pending_max_priority, new_priorities.max(), out=rb._pending_max_priority)
            out = {
                "priority_td_error": priority_td_error,
                "q_loss": (q_total_loss / self.num_critics).detach(),
                "q_total_loss": q_total_loss.detach(),
                "q1_mean": q1_mean,
                "bellman_target_mean": bellman_mean,
                "next_q_h_mean": next_q_h_mean,
                "priority_td_error_mean": priority_td_error.mean(),
                "sampled_reward_mean": data["rewards"].mean(),
            }
        return out

    def _actor_core(self, obs: torch.Tensor, prev_actions: torch.Tensor):
        policy_obs = self._policy_obs(obs, prev_actions)
        current_policy_actions = deterministic_actor_action(self.actor, policy_obs)
        q1_h = self.qfs[0](obs, current_policy_actions)
        q1 = h_inverse(q1_h, eps=self.h_eps).view(-1)
        norm_q = (1.0 - self.gamma) * q1
        actor_loss = -norm_q.mean()
        return actor_loss, norm_q.detach().mean()

    def _actor_step(self, counts: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        obs, prev_actions = self._sample_actor(counts)
        actor_loss, norm_q_mean = self._actor_core_fn(obs, prev_actions)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        return {
            "actor_loss": actor_loss.detach(),
            "actor_norm_q_mean": norm_q_mean,
        }

    def _set_critic_requires_grad(self, flag: bool) -> None:
        for q in self.qfs:
            for p in q.parameters():
                p.requires_grad_(flag)

    # --------------------------------------------------------------- capture
    def _snapshot(self):
        all_params = [p for q in self.qfs for p in q.parameters()] + list(self.actor.parameters())
        snap = {
            "params": [p.detach().clone() for p in all_params],
            "param_refs": all_params,
            "q_state": _snapshot_optimizer_state(self.q_optimizer),
            "a_state": _snapshot_optimizer_state(self.actor_optimizer),
            "rng": torch.cuda.get_rng_state(self.device),
            "priorities": [
                (v.rb.priorities.detach().clone(), v.rb._pending_max_priority.detach().clone())
                for v in self.buffers
                if v.prioritized
            ],
        }
        return snap

    def _restore(self, snap) -> None:
        with torch.no_grad():
            for p, s in zip(snap["param_refs"], snap["params"]):
                p.copy_(s)
            _restore_optimizer_state(self.q_optimizer, snap["q_state"])
            _restore_optimizer_state(self.actor_optimizer, snap["a_state"])
            prioritized = [v for v in self.buffers if v.prioritized]
            for v, (pr, pm) in zip(prioritized, snap["priorities"]):
                v.rb.priorities.copy_(pr)
                v.rb._pending_max_priority.copy_(pm)
        torch.cuda.set_rng_state(snap["rng"], self.device)
        torch.cuda.synchronize(self.device)

    def _capture(self, kind: str, counts: Tuple[int, int]) -> None:
        with torch.cuda.device(self.device):
            self._capture_on_device(kind, counts)

    def _capture_on_device(self, kind: str, counts: Tuple[int, int]) -> None:
        gc.collect()  # drop stale AccumulateGrad nodes bound to the legacy stream (see module docstring)
        try:
            self._capture_once(kind, counts)
        except Exception as exc:
            if not self.compile_update:
                raise
            print(f"[GraphedTD3Update] capture with torch.compile failed ({str(exc).splitlines()[0]}); retrying uncompiled.")
            self.compile_update = False
            self._critic_core_fn = self._critic_core
            self._actor_core_fn = self._actor_core
            self._critic_graphs.clear()
            self._actor_graphs.clear()
            torch.cuda.synchronize(self.device)
            gc.collect()
            self._capture_once(kind, counts)

    def _capture_once(self, kind: str, counts: Tuple[int, int]) -> None:
        snap = self._snapshot()
        stream = self._capture_stream
        stream.wait_stream(torch.cuda.current_stream(self.device))
        if kind == "actor":
            self._set_critic_requires_grad(False)
        try:
            with torch.cuda.stream(stream):
                for _ in range(3):
                    if kind == "critic":
                        self._critic_step(counts)
                    else:
                        self._actor_step(counts)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            torch.cuda.synchronize(self.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                out = self._critic_step(counts) if kind == "critic" else self._actor_step(counts)
            torch.cuda.synchronize(self.device)
        finally:
            if kind == "actor":
                self._set_critic_requires_grad(True)
            self._restore(snap)
        (self._critic_graphs if kind == "critic" else self._actor_graphs)[counts] = (graph, out)

    # ------------------------------------------------------------------ API
    def critic_update(self, success_count: int, failure_count: int, per_beta: float) -> Dict[str, torch.Tensor]:
        counts = (int(success_count), int(failure_count))
        for view in self.buffers:
            view.sync_size()
        self.beta_t.fill_(float(per_beta))
        if self.use_graph:
            if counts not in self._critic_graphs:
                self._capture("critic", counts)
            graph, out = self._critic_graphs[counts]
            graph.replay()
            self.critic_out = out
            return out
        self.critic_out = self._critic_step(counts)
        return self.critic_out

    def actor_update(self, success_count: int, failure_count: int) -> Dict[str, torch.Tensor]:
        counts = (int(success_count), int(failure_count))
        for view in self.buffers:
            view.sync_size()
        if self.use_graph:
            if counts not in self._actor_graphs:
                self._capture("actor", counts)
            graph, out = self._actor_graphs[counts]
            graph.replay()
            self.actor_out = out
            return out
        self._set_critic_requires_grad(False)
        try:
            self.actor_out = self._actor_step(counts)
        finally:
            self._set_critic_requires_grad(True)
        return self.actor_out

    def polyak(self) -> None:
        """target <- (1 - tau) * target + tau * online, fused via foreach."""
        with torch.no_grad():
            torch._foreach_lerp_(self._polyak_tgt, self._polyak_src, self.tau)
