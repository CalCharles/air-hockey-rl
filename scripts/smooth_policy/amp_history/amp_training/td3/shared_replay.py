"""Shared-memory replay service for async TD3 collector/learner processes."""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, Tuple

import torch


PARTITIONS: Tuple[str, str] = ("success", "failure")


def _as_cpu_float_tensor(value: torch.Tensor | object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
    else:
        tensor = torch.as_tensor(value)
    if tensor.device.type != "cpu":
        tensor = tensor.to("cpu")
    return tensor.to(dtype=torch.float32)


@dataclass
class SharedReplayPartition:
    observations: torch.Tensor
    next_observations: torch.Tensor
    actions: torch.Tensor
    prev_actions: torch.Tensor
    task_rewards: torch.Tensor
    motion_rewards: torch.Tensor
    dones: torch.Tensor
    position: mp.Value
    size: mp.Value
    lock: mp.Lock
    capacity: int

    @classmethod
    def build(cls, capacity: int, obs_shape: Tuple[int, ...], action_shape: Tuple[int, ...]) -> "SharedReplayPartition":
        observations = torch.zeros((capacity, *obs_shape), dtype=torch.float32).share_memory_()
        next_observations = torch.zeros((capacity, *obs_shape), dtype=torch.float32).share_memory_()
        actions = torch.zeros((capacity, *action_shape), dtype=torch.float32).share_memory_()
        prev_actions = torch.zeros((capacity, *action_shape), dtype=torch.float32).share_memory_()
        task_rewards = torch.zeros((capacity,), dtype=torch.float32).share_memory_()
        motion_rewards = torch.zeros((capacity,), dtype=torch.float32).share_memory_()
        dones = torch.zeros((capacity,), dtype=torch.float32).share_memory_()
        return cls(
            observations=observations,
            next_observations=next_observations,
            actions=actions,
            prev_actions=prev_actions,
            task_rewards=task_rewards,
            motion_rewards=motion_rewards,
            dones=dones,
            position=mp.Value("i", 0, lock=False),
            size=mp.Value("i", 0, lock=False),
            lock=mp.Lock(),
            capacity=int(capacity),
        )

    def add_batch(
        self,
        observations: torch.Tensor,
        next_observations: torch.Tensor,
        actions: torch.Tensor,
        prev_actions: torch.Tensor,
        task_rewards: torch.Tensor,
        motion_rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> int:
        obs = _as_cpu_float_tensor(observations)
        next_obs = _as_cpu_float_tensor(next_observations)
        act = _as_cpu_float_tensor(actions)
        prev_act = _as_cpu_float_tensor(prev_actions)
        task_rew = _as_cpu_float_tensor(task_rewards).reshape(-1)
        motion_rew = _as_cpu_float_tensor(motion_rewards).reshape(-1)
        done_vals = _as_cpu_float_tensor(dones).reshape(-1)

        batch_size = int(obs.shape[0])
        if batch_size <= 0:
            return 0

        if batch_size > self.capacity:
            start_idx = batch_size - self.capacity
            obs = obs[start_idx:]
            next_obs = next_obs[start_idx:]
            act = act[start_idx:]
            prev_act = prev_act[start_idx:]
            task_rew = task_rew[start_idx:]
            motion_rew = motion_rew[start_idx:]
            done_vals = done_vals[start_idx:]
            batch_size = self.capacity

        with self.lock:
            position = int(self.position.value)
            first_chunk = min(batch_size, self.capacity - position)
            first_slice = slice(position, position + first_chunk)

            self.observations[first_slice] = obs[:first_chunk]
            self.next_observations[first_slice] = next_obs[:first_chunk]
            self.actions[first_slice] = act[:first_chunk]
            self.prev_actions[first_slice] = prev_act[:first_chunk]
            self.task_rewards[first_slice] = task_rew[:first_chunk]
            self.motion_rewards[first_slice] = motion_rew[:first_chunk]
            self.dones[first_slice] = done_vals[:first_chunk]

            second_chunk = batch_size - first_chunk
            if second_chunk > 0:
                second_slice = slice(0, second_chunk)
                self.observations[second_slice] = obs[first_chunk:]
                self.next_observations[second_slice] = next_obs[first_chunk:]
                self.actions[second_slice] = act[first_chunk:]
                self.prev_actions[second_slice] = prev_act[first_chunk:]
                self.task_rewards[second_slice] = task_rew[first_chunk:]
                self.motion_rewards[second_slice] = motion_rew[first_chunk:]
                self.dones[second_slice] = done_vals[first_chunk:]

            self.position.value = (position + batch_size) % self.capacity
            self.size.value = min(int(self.size.value) + batch_size, self.capacity)
        return batch_size

    def sample(self, batch_size: int, device: str | torch.device) -> Dict[str, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        with self.lock:
            current_size = int(self.size.value)
            if current_size <= 0:
                raise ValueError("Cannot sample from empty shared replay partition.")
            indices = torch.randint(0, current_size, (batch_size,), device="cpu")
            batch = {
                "observations": self.observations[indices].clone(),
                "next_observations": self.next_observations[indices].clone(),
                "actions": self.actions[indices].clone(),
                "prev_actions": self.prev_actions[indices].clone(),
                "task_rewards": self.task_rewards[indices].clone(),
                "motion_rewards": self.motion_rewards[indices].clone(),
                "dones": self.dones[indices].clone(),
            }

        if device == "cpu":
            return batch
        return {key: value.to(device) for key, value in batch.items()}

    def __len__(self) -> int:
        return int(self.size.value)


class SharedTD3Replay:
    """Shared replay with success/failure partitions accessed by multiple processes."""

    def __init__(
        self,
        success_capacity: int,
        failure_capacity: int,
        obs_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
    ) -> None:
        self.partitions: Dict[str, SharedReplayPartition] = {
            "success": SharedReplayPartition.build(int(success_capacity), obs_shape, action_shape),
            "failure": SharedReplayPartition.build(int(failure_capacity), obs_shape, action_shape),
        }

    def _partition(self, name: str) -> SharedReplayPartition:
        if name not in self.partitions:
            raise KeyError(f"Unknown replay partition '{name}'. Expected one of {PARTITIONS}.")
        return self.partitions[name]

    def add_episode(self, partition: str, episode_tensors: Dict[str, torch.Tensor]) -> int:
        replay_partition = self._partition(partition)
        return replay_partition.add_batch(
            observations=episode_tensors["observations"],
            next_observations=episode_tensors["next_observations"],
            actions=episode_tensors["actions"],
            prev_actions=episode_tensors["prev_actions"],
            task_rewards=episode_tensors["task_rewards"],
            motion_rewards=episode_tensors["motion_rewards"],
            dones=episode_tensors["dones"],
        )

    def sample(self, partition: str, batch_size: int, device: str | torch.device) -> Dict[str, torch.Tensor]:
        return self._partition(partition).sample(batch_size=batch_size, device=device)

    def len(self, partition: str) -> int:
        return len(self._partition(partition))

    def state_snapshot(self) -> Dict[str, Dict[str, int]]:
        return {
            partition_name: {
                "size": len(partition),
                "position": int(partition.position.value),
                "capacity": int(partition.capacity),
            }
            for partition_name, partition in self.partitions.items()
        }
