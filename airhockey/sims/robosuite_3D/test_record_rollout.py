# examples/record_rollout.py

from __future__ import annotations
import argparse
from pathlib import Path

from airhockey.sims.robosuite_3D.rollout import record_rollout_video
from airhockey.sims.robosuite_3D.robosuite_env import RobosuiteAirHockeyAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="reach", choices=("reach", "strike", "block"))
    parser.add_argument("--output", default="videos/random_rollout.mp4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--camera", default="overview",
                        choices=("overview", "agentview", "frontview", "birdview"))
    args = parser.parse_args()

    env = RobosuiteAirHockeyAdapter(
        task=args.task,
        has_offscreen_renderer=True,
        camera_name=args.camera,
        max_episode_steps=args.max_steps,
        seed=args.seed,
    )

    def random_policy(obs, deterministic):
        return env.action_space.sample()

    stats = record_rollout_video(
        env, random_policy, Path(args.output),
        camera_name=args.camera,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    env.close()
    print(f"Saved {stats['num_frames']} frames → {stats['video_path']} "
          f"return={stats['return']:.3f}")


if __name__ == "__main__":
    main()