"""Range / mutual-exclusion validation for the TD3 training Args dataclass.

Kept separate from the dataclass so the `Args` definition stays a clean
schema. The validator duck-types `args` — any object with the right
attribute names works.
"""


def validate_optional_exploration_range(
    *,
    primitive_name: str,
    min_angle_deg: float | None,
    max_angle_deg: float | None,
    min_magnitude: float | None,
    max_magnitude: float | None,
) -> None:
    values = (min_angle_deg, max_angle_deg, min_magnitude, max_magnitude)
    if all(value is None for value in values):
        return
    if any(value is None for value in values):
        raise ValueError(
            f"{primitive_name} exploration range requires all four fields: "
            "min_angle_deg, max_angle_deg, min_magnitude, max_magnitude."
        )


def validate_args(args) -> None:
    """Range / mutual-exclusion checks. Raises ValueError on misconfig."""

    def _positive(name: str, value: float) -> None:
        if value <= 0:
            raise ValueError(f"{name} must be > 0.")

    def _fraction(name: str, value: float, *, exclusive: bool = False) -> None:
        lo_ok = 0.0 < value if exclusive else 0.0 <= value
        hi_ok = value < 1.0 if exclusive else value <= 1.0
        if not (lo_ok and hi_ok):
            bracket = "(0, 1)" if exclusive else "[0, 1]"
            raise ValueError(f"{name} must be in {bracket}, got {value}.")

    def _sums_to_one(name1: str, name2: str, v1: float, v2: float) -> None:
        total = float(v1 + v2)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"{name1} + {name2} must equal 1.0, got {total:.6f}.")

    if args.num_envs != 1:
        raise ValueError(
            "This training script currently supports only single-environment collection. "
            f"Set num_envs=1, got {args.num_envs}."
        )
    _fraction("critic_per_fraction", args.critic_per_fraction)
    _fraction("critic_uniform_fraction", args.critic_uniform_fraction)
    _sums_to_one(
        "critic_per_fraction", "critic_uniform_fraction",
        args.critic_per_fraction, args.critic_uniform_fraction,
    )
    _positive("success_buffer_size", args.success_buffer_size)
    _positive("failure_buffer_size", args.failure_buffer_size)
    _positive("recent_episode_window_size", args.recent_episode_window_size)
    _fraction("success_top_fraction", args.success_top_fraction, exclusive=True)
    _fraction("critic_success_sample_fraction", args.critic_success_sample_fraction)
    _fraction("critic_failure_sample_fraction", args.critic_failure_sample_fraction)
    _sums_to_one(
        "critic_success_sample_fraction", "critic_failure_sample_fraction",
        args.critic_success_sample_fraction, args.critic_failure_sample_fraction,
    )
    _positive("q_updates", args.q_updates)
    _positive("target_network_frequency", args.target_network_frequency)
    _positive("actor_updates_per_iteration", args.actor_updates_per_iteration)
    if args.target_network_frequency > args.q_updates:
        # Polyak gate counts completed critic updates globally (see
        # total_critic_updates), so this still fires — just less often than
        # once per cycle. Loud warning since it's almost always a config typo.
        print(
            f"[warn] target_network_frequency ({args.target_network_frequency}) "
            f"> q_updates ({args.q_updates}); target nets will update less than "
            f"once per training cycle."
        )
    for name in ("same_direction", "y_aligned", "target_position_directional"):
        validate_optional_exploration_range(
            primitive_name=name,
            min_angle_deg=getattr(args, f"exploration_{name}_min_angle_deg"),
            max_angle_deg=getattr(args, f"exploration_{name}_max_angle_deg"),
            min_magnitude=getattr(args, f"exploration_{name}_min_magnitude"),
            max_magnitude=getattr(args, f"exploration_{name}_max_magnitude"),
        )
