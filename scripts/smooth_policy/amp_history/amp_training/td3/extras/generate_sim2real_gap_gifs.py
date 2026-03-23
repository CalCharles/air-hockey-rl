import argparse
import os

import yaml

from scripts.smooth_policy.evaluate import evaluate_agent


def main():
    parser = argparse.ArgumentParser(description="Generate rollout GIFs for the sim2real-gap env variant.")
    parser.add_argument(
        "--env-config-path",
        type=str,
        default=(
            "scripts/smooth_policy/amp_history/configs/new_juggle/"
            "pid_noise_sim2real_gap_upper_mid_band.yaml"
        ),
    )
    parser.add_argument(
        "--warm-start-dir",
        type=str,
        default="scripts/smooth_policy/amp_history/amp_training/td3/extras/warm_start",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="model2.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/sim2real_gap_viz/model2",
    )
    parser.add_argument("--n-gifs", type=int, default=3)
    parser.add_argument("--n-eps-per-gif", type=int, default=1)
    args = parser.parse_args()

    env_config_path = os.path.abspath(args.env_config_path)
    warm_start_dir = os.path.abspath(args.warm_start_dir)
    model_path = os.path.join(warm_start_dir, args.model_name)
    warm_args_path = os.path.join(warm_start_dir, "args.yaml")
    warm_config_path = os.path.join(warm_start_dir, "config.yaml")
    output_dir = os.path.abspath(args.output_dir)

    with open(env_config_path, "r") as f:
        env_cfg = yaml.load(f, Loader=yaml.FullLoader)
    with open(warm_args_path, "r") as f:
        warm_args = yaml.load(f, Loader=yaml.FullLoader)
    with open(warm_config_path, "r") as f:
        warm_cfg = yaml.load(f, Loader=yaml.FullLoader)

    action_scale = float(warm_args.get("action_scale", 0.02))
    if warm_cfg.get("air_hockey", {}).get("use_pid", False):
        action_scale = 1.0

    evaluate_agent(
        model_path=model_path,
        save_dir=output_dir,
        air_hockey_params=env_cfg["air_hockey"],
        n_eps=int(args.n_eps_per_gif),
        n_gifs=int(args.n_gifs),
        action_scale=action_scale,
        agent_hidden_layer_size=int(warm_args.get("agent_hidden_layer_size", 64)),
        agent_num_hidden_layers=int(warm_args.get("agent_num_hidden_layers", 2)),
        use_last_action_in_policy_state=bool(
            warm_args.get("use_last_action_in_policy_state", False)
        ),
        policy_type="deterministic_agent",
    )
    print(f"Saved {args.n_gifs} GIFs to: {output_dir}")


if __name__ == "__main__":
    main()
