"""
Render the air-hockey puck-juggle task in robosuite with the RoundGripper
paddle attached. Verifies that:
  - The yellow round paddle is on the EEF
  - The puck spawns on the table and slides under the (tilted) table's
    gravity-induced slope
  - Several camera views show the interaction

Run:
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \\
      .venv/bin/python scripts/render_puck_juggle.py
"""
import os
import shutil

import imageio.v2 as iio
import numpy as np
import yaml

from airhockey import AirHockeyEnv

NEEDS_VFLIP = {"sideview", "frontview", "backview"}
CAMERAS = ["birdview", "sideview", "frontview", "backview", "puckview"]
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval_gifs", "juggle")


def build_env(num_steps=120):
    cfg_fp = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "baseline_configs", "robosuite", "puck_juggle_robosuite.yaml",
    )
    with open(cfg_fp, "r") as f:
        cfg = yaml.safe_load(f)
    ah = cfg["air_hockey"]
    ah["n_training_steps"] = cfg["n_training_steps"]
    ah["return_goal_obs"] = False
    ah["seed"] = 7
    ah["max_timesteps"] = num_steps
    # Disable boundary kills so the rollout actually plays out for visualization.
    ah["terminate_on_out_of_bounds"] = False
    ah["terminate_on_puck_hit_bottom"] = False
    ah["terminate_on_puck_pass_paddle"] = False
    sp = ah.setdefault("simulator_params", {})
    sp["seed"] = 7
    sp["has_renderer"] = False
    sp["has_offscreen_renderer"] = True
    sp["camera_names"] = CAMERAS
    sp["camera_heights"] = 512
    sp["camera_widths"] = 512
    # Attach the round paddle gripper to the EEF (yellow cylinder, radius
    # 0.0508 m — matches the env's paddle_radius). Registered by
    # airhockey/sims/grippers/__init__.py.
    sp["gripper_types"] = "RoundGripper"
    return AirHockeyEnv(ah)


def main():
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)

    env = build_env()
    obs, info = env.reset()

    sim = env.simulator.robosuite_env.sim
    puck_id = sim.model.body_name2id("puck_0")
    eef_id = sim.model.body_name2id("gripper0_right_eef")
    print(f"After env init/reset:")
    print(f"  puck world pos = {sim.data.body_xpos[puck_id]}")
    print(f"  EEF world pos  = {sim.data.body_xpos[eef_id]}")
    print(f"  puck z gravity affected? "
          f"gravcomp={sim.model.body_gravcomp[puck_id] if hasattr(sim.model, 'body_gravcomp') else 'N/A'}")

    per_cam_frames = {c: [] for c in CAMERAS}
    step = 0
    done = False
    puck_z_history = []
    while not done:
        for cam in CAMERAS:
            key = f"{cam}_image"
            img = env.current_state.get(key)
            if img is None:
                continue
            if cam in NEEDS_VFLIP:
                img = np.flipud(img)
            per_cam_frames[cam].append(img)

        # First half: paddle holds at start position so puck slides into it
        # (testing gravity + tilt + paddle-puck contact). Second half: drive
        # the paddle forward in pulses to try to "juggle" the puck back.
        if step < 60:
            action = np.zeros(2)
        else:
            # Strong forward pulse every 8 steps to strike the puck
            phase = (step - 60) % 16
            ax = -0.8 if phase < 4 else (0.6 if phase < 8 else 0.0)
            action = np.array([ax, 0.0])
        obs, rew, terminated, truncated, info = env.step(action)
        puck_z_history.append(float(sim.data.body_xpos[puck_id][2]))
        step += 1
        done = terminated or truncated

    print(f"Rolled out {step} steps")
    print(f"Puck z range: [{min(puck_z_history):.3f}, {max(puck_z_history):.3f}]")
    print(f"Final puck pos: {sim.data.body_xpos[puck_id]}")
    print(f"Final EEF pos:  {sim.data.body_xpos[eef_id]}")

    duration_ms = int(1000 / 20)
    for cam in CAMERAS:
        if not per_cam_frames[cam]:
            continue
        iio.mimsave(
            os.path.join(OUT_DIR, f"{cam}.gif"),
            per_cam_frames[cam], format="GIF", loop=0, duration=duration_ms,
        )
        iio.imwrite(os.path.join(OUT_DIR, f"{cam}_f000.png"), per_cam_frames[cam][0])
        iio.imwrite(os.path.join(OUT_DIR, f"{cam}_f{len(per_cam_frames[cam]) - 1:03d}.png"), per_cam_frames[cam][-1])
        print(f"  {cam}: {len(per_cam_frames[cam])} frames")


if __name__ == "__main__":
    main()
