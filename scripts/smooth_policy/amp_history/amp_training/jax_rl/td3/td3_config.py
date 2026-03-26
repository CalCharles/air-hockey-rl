"""TD3 hyperparameters.

Algorithm-level knobs only. Environment, logging, and checkpointing
settings live in the training script.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TD3Config:
    # Core TD3
    task_gamma: float = 0.975
    motion_gamma: float = 0.8
    tau: float = 0.005
    batch_size: int = 256
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    q_weight_decay: float = 1e-4
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    exploration_noise: float = 0.1
    h_transform_eps: float = 1e-3

    # Update schedule
    q_updates: int = 1
    actor_updates_per_iteration: int = 1
    target_network_frequency: int = 1

    # Dual-head reward decomposition
    task_reward_weight: float = 1.0
    motion_reward_weight: float = 1.0

    # Prioritized experience replay
    per_enabled: bool = True
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_anneal_steps: int = 200000
    per_eps: float = 1e-6
    critic_per_fraction: float = 0.7
    critic_uniform_fraction: float = 0.3

    # Success / failure replay split
    success_buffer_size: int = int(2e5)
    failure_buffer_size: int = int(8e5)
    success_top_fraction: float = 0.2
    recent_episode_window_size: int = 500
    critic_success_sample_fraction: float = 0.3
    critic_failure_sample_fraction: float = 0.7

    # Network architecture
    actor_hidden_dim: int = 64
    actor_num_blocks: int = 2
    q_hidden_dim: int = 128
    q_num_blocks: int = 2
    action_scale: float = 0.02
    use_last_action_in_policy_state: bool = False
