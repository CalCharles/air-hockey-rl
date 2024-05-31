import os
import cv2
import numpy as np
import yaml
# import pygame
from stable_baselines3.common.vec_env import DummyVecEnv
from airhockey import AirHockeyEnv
from airhockey.renderers import AirHockeyRenderer
from airhockey.sims.real.control_parameters import single_point_homography
# import pdb; pdb.set_trace()
mousepos = (0,0)

def move_event(event, x, y, flags, params):
    global mousepos
    if event==cv2.EVENT_MOUSEMOVE:
  
        # displaying the coordinates
        # on the Shell
        # print(x, ' ', y)
  
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (x, y)
        mousepos = (x,y)


# 0.5545621421040268, 0.48758756141253956
# 265, 466
# 1.0064265409081836, 0.4957963720907306
# 265, 582
# 0.5529227565159962, -0.4840302450683018
# 504, 464
# 0.9633642320739517, -0.4887553926441414
# 503, 577
def get_frames(renderer, env, cfg):
    frames = []
    env = env_test.envs[0]
    robosuite_frames = {} if cfg['air_hockey']['simulator'] == 'robosuite' else None

    # pygame.init()
    # screen = pygame.display.set_mode((800, 600))
    # pygame.display.set_caption('Air Hockey Evaluation')
    # clock = pygame.time.Clock()
    

    obs = env_test.reset()
    done = False
    IMAGE_SIZE = 320
    mimg = np.load(os.path.join("assets", "robosuite", "Mimg.npy"))
    mrob = np.load(os.path.join("assets", "robosuite", "Mrob.npy"))

    # prev_mouse_pos = pygame.mouse.get_pos()
    for i in range(2000):
        done = False
        while not done:
            frame = renderer.get_frame()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            aspect_ratio = frame.shape[1] / frame.shape[0]
            frame = cv2.resize(frame, (IMAGE_SIZE, int(IMAGE_SIZE / aspect_ratio)))
            frames.append(frame)

            if cfg['air_hockey']['simulator'] == 'robosuite':
                for key in env.current_state:
                    if 'image' not in key:
                        continue
                    current_img = env.current_state[key]
                    # current_img = cv2.flip(current_img, 0)
                    current_img = cv2.resize(current_img, (int(current_img.shape[1] * 1.5), int(current_img.shape[0] * 1.5)))
                    # current_img = cv2.resize(current_img, (IMAGE_SIZE, int(IMAGE_SIZE / aspect_ratio)))
                    # current_img = np.concatenate([frame, current_img], axis=1)
                    if key not in robosuite_frames:
                        robosuite_frames[key] = [current_img]
                    else:
                        robosuite_frames[key].append(current_img)

                display_frame = robosuite_frames['birdview_image'][-1]
            else:
                display_frame = frame
            

            cv2.imshow("robot", display_frame)
            cv2.imshow("side", robosuite_frames['sideview_image'][-1])
            cv2.imshow("box2d", frame)
            cv2.setMouseCallback('robot', move_event)
            # print(list(env.current_state.keys()), env.current_state["paddles"]["paddle_ego"]["position"], mousepos)
            cv2.waitKey(300)

            mouse_robo_pos = single_point_homography(mimg, (mousepos[1], mousepos[0]))
            mouse_robo_pos[1] = - mouse_robo_pos[1]
            action = - np.array([(env.current_state["paddles"]["paddle_ego"]["position"] - mouse_robo_pos) * 1]) * 0.1
            action[0,1] = - action[0,1]
            print(mouse_robo_pos, env.current_state["paddles"]["paddle_ego"]["position"][1], action[0,1])
            # action[0,1] = (np.random.rand() - 0.5) * 2
            obs, rew, done, info = env_test.step(action)
            # done = done or info[0]['TimeLimit.truncated']

            # Convert display_frame to Pygame surface and display it
            # surface = pygame.surfarray.make_surface(cv2.transpose(display_frame))
            # screen.blit(surface, (0, 0))
    #         pygame.display.flip()
    #         clock.tick(30)

    # pygame.quit()
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
