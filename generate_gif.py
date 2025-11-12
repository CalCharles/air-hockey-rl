import numpy as np
from airhockey import AirHockeyEnv
import yaml
import imageio

with open('configs/baseline_configs/robosuite/puck_height_robosuite.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

env = AirHockeyEnv(cfg)
frames = []
obs = env.reset()

print("Recording episode...")
for step in range(100):
    action = np.array([0.0, 0.0])
    obs, reward, done, info = env.step(action)
    frame = env.simulator.render(mode='rgb_array', width=640, height=480)
    frames.append(frame)
    if done:
        break

imageio.mimsave('puck_falling.gif', frames, fps=20)
print(f"Saved {len(frames)} frames to puck_falling.gif")
