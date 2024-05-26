import argparse
import os
import yaml
import cv2
import time
import copy
import numpy as np
import sys
 
from dataset_management.repair_data import read_new_real_data, read_real_data
from airhockey import AirHockeyEnv
from airhockey.airhockey_base import populate_state_info
from airhockey.renderers import AirHockeyRenderer
from dataset_management.clean_data import homography_transform


class Visualizer:
    def __init__(self, air_hockey_cfg, dataset_path, num_load):
        """
        Initializes the Demonstrator class.

        Creates an instance of the AirHockeyBox2D class with specified parameters,
        an instance of the AirHockeyRenderer class, and sets the keyboard scheme.

        Parameters:
        None

        Returns:
        None
        """
        air_hockey_params = air_hockey_cfg['air_hockey']
        air_hockey_params['n_training_steps'] = air_hockey_cfg['n_training_steps']
        air_hockey_params['seed'] = air_hockey_cfg['seed']
        if 'goal' in air_hockey_cfg['air_hockey']['task']:
            air_hockey_params['return_goal_obs'] = True
        else:
            air_hockey_params['return_goal_obs'] = False
        self.air_hockey = AirHockeyEnv(air_hockey_params)
        self.renderer = AirHockeyRenderer(self.air_hockey)
        self.keyboard_scheme = 'wasd'
        self.print_reward = air_hockey_cfg['print_reward']
        # self.values, self.images, self.dones = read_new_real_data(dataset_path, num_load)
        self.data_dict = read_real_data(dataset_path, num_load)

    def step_frame(self, key):
        """
        Performs the demonstration of the air hockey game.

        Captures the frame from the renderer, displays it, and waits for user input.
        Based on the keyboard scheme, determines the action to be taken.
        If the orientation is vertical, adjusts the action accordingly.

        Parameters:
        None

        Returns:
        action (numpy.array): The action to be taken in the game.
        """
        action = np.array([0,0])
        frame = self.renderer.get_frame()
        cv2.imshow('Air Hockey 2D Demonstration',frame)
        key = cv2.waitKey(20)
        DEMOFORCE = 0.005
        action = 0
        if key == ord('a'):
            action = -1
        elif key == ord('d'):
            action = 1
        print(action)
        return action
        
    def run(self):
        """
        Runs the air hockey demonstration.

        Iterates through a loop, capturing user input and updating the game state.
        Prints the frames per second (fps) every 1000 iterations.
        Resets the game state every 300 iterations.

        Parameters:
        None

        Returns:
        None
        """
        start = time.time()
        frame_idx = 0
        Mimg = np.load('assets/real/Mimg.npy')

        while True:
            frame, save_image = homography_transform(self.data_dict["image"][frame_idx], from_save = True, rotate=False, Mimg = Mimg)
            for r in self.air_hockey.reward_regions:
                # print("reward_regions", r.state, r.radius)
                offset_constants = np.array((2100, 500))
                # ((0.09870443) *1000  + 500)/2
                pixel_coord = (((r.state[0] - 1)*1000, (-r.state[1])*1000 ) + offset_constants) /2
                center_coordinates = (int(np.round(pixel_coord[0])), int(np.round(pixel_coord[1])))  # Example coordinates (x, y)
                
                pixel_radius = (int((r.radius[0])*1000 /2), int((r.radius[1])*1000 /2))
                # radius = int(np.round((pixel_radius[0]))) #, int(np.round(-pixel_radius[1])))
                color = (0, 0, 255)  # Red color in BGR
                thickness = 2  # Thickness of 2 px, use -1 for a filled circle
                # center_coordinates = (480, 360)
                center_coordinates_new = (center_coordinates[0], center_coordinates[1])
                # print("reward_regions", center_coordinates_new, pixel_radius)
                # Draw the circle on the frame
                cv2.ellipse(frame, center_coordinates_new, pixel_radius, angle=0, startAngle=0, endAngle=360, color=color, thickness=thickness)

                # cv2.circle(frame, center_coordinates_new, radius, color=color, thickness=thickness)
            cv2.imshow("frame", frame)
            # cv2.imwrite('./images_debug.png', frame) 
            key = cv2.waitKey(10)
            # convert from pixel to robot coordinates with:
            # offset_constants = np.array((2100, 500))
            # x, y = (pixel_coord - self.offset_constants) * 0.001
            # y= -y

            # (px, py) = (x+1 , y) * 1000 + self.offset_constants)
            # py = -py

            frame_idx += self.step_frame(key)
            frame_idx = min(max(0, frame_idx), len(self.data_dict["pose"]) - 1)
            paddles = [copy.deepcopy(self.data_dict["pose"][frame_idx][:2])]
            # print("paddle", self.values[frame_idx]["pose"][:2])
            # paddle_coord = ((paddles[0][0])*3000, (paddles[0][1])*3000 ) + offset_constants
            # print('debug p',paddles[0][0], paddles[0][0], (paddles[0][0])*1000 *3, paddle_coord/2)
            paddles[0][0] = paddles[0][0] + 1.0
            pucks = [self.data_dict["puck"][frame_idx][:2] if "puck" in self.data_dict else [-1,0,0]]
            self.air_hockey.simulator.paddles['paddle_ego'].position = paddles[0]
            state_dict = populate_state_info(paddles, pucks, [])
            rew = self.air_hockey.get_base_reward(state_dict)
            # if self.print_reward:
            print("reward: ", rew)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demonstrate the air hockey game.')
    parser.add_argument('--cfg', type=str, default=None, help='Path to the configuration file.')
    parser.add_argument('--data', type=str, default="/datastor1/calebc/public/data/mouse/cleaned_new/", help='Path to the trajectories folder.')
    parser.add_argument('--num-load', type=int, default=2, help='number of trajectories to load.')
    # /datastor1/calebc/public/data/mouse/cleaned_new/
    args = parser.parse_args()
    if args.cfg is None:
        # Then our default path is demonstrate.yaml in the config file
        # dir_path = os.path.dirname(os.path.realpath(__file__) + "../")
        dir_path = "./"
        air_hockey_cfg_fp = os.path.join(dir_path, 'configs', 'demonstrate.yaml')
    else:
        air_hockey_cfg_fp = args.cfg
    with open(air_hockey_cfg_fp, 'r') as f:
        air_hockey_cfg = yaml.safe_load(f)

    visualizer = Visualizer(air_hockey_cfg, args.data, args.num_load)
    visualizer.run()
    cv2.destroyAllWindows()