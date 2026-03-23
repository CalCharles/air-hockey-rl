"""
Generate paired GIFs for the same seeds:
1) policy takeover for full episode duration
2) pre-contact hit variant for full episode duration
"""

from dataclasses import dataclass

import tyro
import yaml

from scripts.smooth_policy.amp_history.amp_training.td3.extras.visualize_exploration_primitives_box2d import (
    Args as PrimitiveVizArgs,
    render_primitive_gif,
)


@dataclass
class Args:
    config: str = (
        "scripts/smooth_policy/amp_history/configs/new_juggle/"
        "pid_noise_constant_upper_half_custom_sim_params.yaml"
    )
    output_dir: str = "runs/td3/primitive_viz_compare"
    seed_start: int = 0
    num_seeds: int = 4
    max_timesteps: int = 200
    fps: int = 20
    device: str = "cpu"

    exploration_policy_takeover_base_dir: str = (
        "scripts/smooth_policy/amp_history/amp_training/td3/extras/warm_start"
    )
    exploration_policy_takeover_model_path: str = "model2.pth"
    exploration_policy_takeover_args_path: str = "args.yaml"
    exploration_policy_takeover_config_path: str = "config.yaml"


def build_primitive_args(args: Args, primitive_name: str) -> PrimitiveVizArgs:
    return PrimitiveVizArgs(
        config=args.config,
        output_dir=args.output_dir,
        primitive=primitive_name,
        gifs_per_primitive=1,
        max_timesteps=args.max_timesteps,
        fps=args.fps,
        seed=args.seed_start,
        device=args.device,
        base_action_mode="zero",
        exploration_primitive_steps=args.max_timesteps,
        exploration_pre_contact_hit_variant_chance=1.0,
        exploration_pre_contact_hit_variant_steps=5,
        exploration_pre_contact_hit_variant_distance_threshold=0.2,
        exploration_pre_contact_hit_variant_scale_min=0.5,
        exploration_pre_contact_hit_variant_scale_max=1.5,
        exploration_policy_takeover_enabled=True,
        exploration_policy_takeover_base_dir=args.exploration_policy_takeover_base_dir,
        exploration_policy_takeover_model_path=args.exploration_policy_takeover_model_path,
        exploration_policy_takeover_args_path=args.exploration_policy_takeover_args_path,
        exploration_policy_takeover_config_path=args.exploration_policy_takeover_config_path,
        debug_logging=True,
    )


if __name__ == "__main__":
    args = tyro.cli(Args)
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    env_params = dict(config["air_hockey"])
    env_params["simulator"] = "box2d"

    takeover_args = build_primitive_args(args, "policy_takeover")
    pre_contact_args = build_primitive_args(args, "pre_contact_hit_variant")

    for seed in range(args.seed_start, args.seed_start + args.num_seeds):
        takeover_out = render_primitive_gif(
            args=takeover_args,
            env_params=env_params,
            primitive_name="policy_takeover",
            seed=seed,
            gif_index=0,
        )
        pre_contact_out = render_primitive_gif(
            args=pre_contact_args,
            env_params=env_params,
            primitive_name="pre_contact_hit_variant",
            seed=seed,
            gif_index=0,
        )
        print(f"seed={seed}: takeover={takeover_out}")
        print(f"seed={seed}: pre_contact={pre_contact_out}")
