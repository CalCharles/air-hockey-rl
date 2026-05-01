"""
Render multiple robosuite camera views (birdview, frontview, sideview,
agentview, backview) of one rollout for diagnostic inspection.

Run:
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \\
      .venv/bin/python scripts/render_robosuite_views.py

Outputs:
    eval_gifs/views/<camera>.gif        per-camera animated GIF
    eval_gifs/views/<camera>_f000.png   first frame for quick inspection
    eval_gifs/views/grid.gif            stacked 5-panel grid GIF
"""
import os
import shutil

import imageio.v2 as iio
import numpy as np
import yaml

from airhockey import AirHockeyEnv

# Cameras for which the raw MuJoCo framebuffer is upside-down and needs
# np.flipud to display right-side up. Birdview/agentview happen to already be
# oriented with the robot at the bottom of the image in this MuJoCo build.
NEEDS_VFLIP = {"sideview", "frontview", "backview"}

CAMERAS = ["birdview", "agentview", "frontview", "sideview", "backview"]

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval_gifs", "views")


def build_env(num_steps: int = 60):
    cfg_fp = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "baseline_configs", "robosuite", "puck_height_robosuite.yaml",
    )
    with open(cfg_fp, "r") as f:
        cfg = yaml.safe_load(f)
    ah = cfg["air_hockey"]
    ah["n_training_steps"] = cfg["n_training_steps"]
    ah["return_goal_obs"] = False
    ah["seed"] = 43
    ah["max_timesteps"] = num_steps
    ah["terminate_on_out_of_bounds"] = False
    sp = ah.setdefault("simulator_params", {})
    sp["seed"] = 43
    sp["has_renderer"] = False
    sp["has_offscreen_renderer"] = True
    sp["camera_names"] = CAMERAS
    sp["camera_heights"] = 512
    sp["camera_widths"] = 512
    return AirHockeyEnv(ah)


def main():
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)

    env = build_env()
    obs, info = env.reset()

    per_cam_frames = {c: [] for c in CAMERAS}
    step = 0
    done = False
    while not done:
        for cam in CAMERAS:
            key = f"{cam}_image"
            img = env.current_state.get(key)
            if img is None:
                # Camera missing from this build — skip.
                continue
            if cam in NEEDS_VFLIP:
                img = np.flipud(img)
            per_cam_frames[cam].append(img)

        action = np.zeros(2)
        obs, rew, terminated, truncated, info = env.step(action)
        step += 1
        done = terminated or truncated

    print(f"rolled out {step} steps")

    # Per-camera GIFs + sample first frame for quick inspection
    duration_ms = int(1000 / 20)  # 20 fps
    for cam in CAMERAS:
        if not per_cam_frames[cam]:
            print(f"  {cam}: NO FRAMES (camera not present in scene)")
            continue
        gif = os.path.join(OUT_DIR, f"{cam}.gif")
        iio.mimsave(gif, per_cam_frames[cam], format="GIF", loop=0, duration=duration_ms)
        iio.imwrite(os.path.join(OUT_DIR, f"{cam}_f000.png"), per_cam_frames[cam][0])
        iio.imwrite(os.path.join(OUT_DIR, f"{cam}_f{len(per_cam_frames[cam]) - 1:03d}.png"), per_cam_frames[cam][-1])
        print(f"  {cam}: {len(per_cam_frames[cam])} frames -> {gif}")

    # Grid GIF: stack into a 2-row layout. Skip empty cameras.
    cams_with_frames = [c for c in CAMERAS if per_cam_frames[c]]
    if cams_with_frames:
        # 2 rows: top row = top-down (birdview, agentview); bottom row = side-on (frontview, sideview, backview)
        top = [c for c in ("birdview", "agentview") if c in cams_with_frames]
        bot = [c for c in ("frontview", "sideview", "backview") if c in cams_with_frames]
        target_h = 256

        def resize(img, h):
            from PIL import Image
            pil = Image.fromarray(img)
            new_w = int(pil.width * h / pil.height)
            return np.asarray(pil.resize((new_w, h)))

        n_frames = min(len(per_cam_frames[c]) for c in cams_with_frames)
        grid_frames = []
        for i in range(n_frames):
            top_imgs = [resize(per_cam_frames[c][i], target_h) for c in top] if top else []
            bot_imgs = [resize(per_cam_frames[c][i], target_h) for c in bot] if bot else []
            top_row = np.concatenate(top_imgs, axis=1) if top_imgs else None
            bot_row = np.concatenate(bot_imgs, axis=1) if bot_imgs else None
            # Pad rows to equal width before concatenating vertically
            if top_row is not None and bot_row is not None:
                max_w = max(top_row.shape[1], bot_row.shape[1])
                if top_row.shape[1] < max_w:
                    pad = np.zeros((top_row.shape[0], max_w - top_row.shape[1], 3), dtype=top_row.dtype)
                    top_row = np.concatenate([top_row, pad], axis=1)
                if bot_row.shape[1] < max_w:
                    pad = np.zeros((bot_row.shape[0], max_w - bot_row.shape[1], 3), dtype=bot_row.dtype)
                    bot_row = np.concatenate([bot_row, pad], axis=1)
                grid = np.concatenate([top_row, bot_row], axis=0)
            else:
                grid = top_row if top_row is not None else bot_row
            grid_frames.append(grid)

        grid_gif = os.path.join(OUT_DIR, "grid.gif")
        iio.mimsave(grid_gif, grid_frames, format="GIF", loop=0, duration=duration_ms)
        iio.imwrite(os.path.join(OUT_DIR, "grid_f000.png"), grid_frames[0])
        iio.imwrite(os.path.join(OUT_DIR, f"grid_f{n_frames - 1:03d}.png"), grid_frames[-1])
        print(f"  grid: {n_frames} frames -> {grid_gif}")


if __name__ == "__main__":
    main()
