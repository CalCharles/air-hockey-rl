from airhockey.envs.air_hockey_env import AirHockeyEnv
from airhockey.renderers.render import AirHockeyRenderer
import yaml

# Load config
with open('configs/baseline_configs/robosuite/puck_height_robosuite.yaml') as f:
    config = yaml.safe_load(f)

# Create env
env = AirHockeyEnv(**config['air_hockey'])
renderer = AirHockeyRenderer(env)

# Run one episode and save GIF
obs = env.reset()
frames = []
for i in range(100):
    action = env.action_space.sample()  # random action
    obs, reward, done, info = env.step(action)
    frames.append(renderer.render())
    if done:
        break

# Save GIF
renderer.save_gif(frames, 'test_height.gif')
print("GIF saved as test_height.gif")
