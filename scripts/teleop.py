import os
import cv2
import numpy as np
import yaml
import pygame
from stable_baselines3.common.vec_env import DummyVecEnv
from airhockey import AirHockeyEnv
from airhockey.renderers import AirHockeyRenderer
# import pdb; pdb.set_trace()
def get_mouse_delta(prev_pos):
    x, y = pygame.mouse.get_pos()
    dx = x - prev_pos[0]
    dy = y - prev_pos[1]
    return np.array([[dx, dy]], dtype=np.float32), (x, y)

def get_frames(renderer, env, cfg):
    frames = []
    env = env_test.envs[0]
    robosuite_frames = {} if cfg['air_hockey']['simulator'] == 'robosuite' else None

    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption('Air Hockey Evaluation')
    clock = pygame.time.Clock()

    obs = env_test.reset()
    done = False

    prev_mouse_pos = pygame.mouse.get_pos()
    for i in range(200):
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return frames, robosuite_frames

            action, prev_mouse_pos = get_mouse_delta(prev_mouse_pos)
            obs, rew, done, info = env_test.step(action)
            done = done or info[0]['TimeLimit.truncated']

            frame = renderer.get_frame()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            aspect_ratio = frame.shape[1] / frame.shape[0]
            frame = cv2.resize(frame, (160, int(160 / aspect_ratio)))
            frames.append(frame)

            if cfg['air_hockey']['simulator'] == 'robosuite':
                for key in env.current_state:
                    if 'image' not in key:
                        continue
                    current_img = env.current_state[key]
                    current_img = cv2.flip(current_img, 0)
                    current_img = cv2.resize(current_img, (160, int(160 / aspect_ratio)))
                    current_img = np.concatenate([frame, current_img], axis=1)
                    if key not in robosuite_frames:
                        robosuite_frames[key] = [current_img]
                    else:
                        robosuite_frames[key].append(current_img)

                display_frame = robosuite_frames['birdview_image'][-1]
            else:
                display_frame = frame

            # Convert display_frame to Pygame surface and display it
            surface = pygame.surfarray.make_surface(cv2.transpose(display_frame))
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            clock.tick(30)

    pygame.quit()
    return frames, robosuite_frames

if __name__ == '__main__':
    with open("configs/baseline_configs/robosuite/puck_vel_robosuite.yaml", 'r') as f:
        model_cfg = yaml.safe_load(f)
    air_hockey_params = model_cfg['air_hockey']
    air_hockey_params['seed'] = 0
    air_hockey_params['n_training_steps'] = model_cfg['n_training_steps']
    air_hockey_params['return_goal_obs'] = 'goal' in model_cfg['air_hockey']['task']
    air_hockey_params['max_timesteps'] = 200

    env_test = AirHockeyEnv(air_hockey_params)
    renderer = AirHockeyRenderer(env_test)

    env_test = DummyVecEnv([lambda: env_test])

    get_frames(renderer, env_test, model_cfg)
