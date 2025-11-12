import numpy as np
from airhockey import AirHockeyEnv
import yaml
import imageio

with open('configs/baseline_configs/robosuite/puck_height_robosuite.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

env = AirHockeyEnv(cfg)
obs = env.reset()

print("Running teleoperation demo...")
print("Moving paddle to hit puck\n")

frames = []

for step in range(150):
    if step < 50:
        action = np.array([1.0, 0.0])
        if step % 10 == 0:
            print(f"Step {step}: Moving forward")
    elif step < 100:
        action = np.array([0.5, -1.0])
        if step == 50:
            print("Sweeping left...")
    else:
        action = np.array([0.5, 1.0])
        if step == 100:
            print("Sweeping right...")
    
    obs, reward, done, info = env.step(action)
    frame = env.simulator.render(mode='rgb_array', width=640, height=480)
    frames.append(frame)
    
    if done:
        print(f"Episode ended at step {step}")
        break

imageio.mimsave('teleoperation_demo.gif', frames, fps=20)
print(f"\nSaved teleoperation_demo.gif ({len(frames)} frames)")
