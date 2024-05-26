import cv2
import numpy as np
import time
from airhockey import AirHockeyEnv
from airhockey.renderers import AirHockeyRenderer
import argparse
import yaml
import os

class Demonstrator:
    def __init__(self, air_hockey_cfg):
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
    
    def demonstrate(self):
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
        if self.keyboard_scheme == 'qweasdzxc':
            if key == ord('k'):
                action = -1
            elif key == ord('q'):
                action = np.array([-DEMOFORCE,-DEMOFORCE])
            elif key == ord('w'):
                action = np.array([-DEMOFORCE,0])
            elif key == ord('e'):
                action = np.array([-DEMOFORCE,DEMOFORCE])
            elif key == ord('a'):
                action = np.array([0,-DEMOFORCE])
            elif key == ord('s'):
                action = np.array([0,0])
            elif key == ord('d'):
                action = np.array([0,DEMOFORCE])
            elif key == ord('z'):
                action = np.array([DEMOFORCE,-DEMOFORCE])
            elif key == ord('x'):
                action = np.array([DEMOFORCE,0])
            elif key == ord('c'):
                action = np.array([DEMOFORCE,DEMOFORCE])
        elif self.keyboard_scheme == 'wasd':
            if key == ord('w'):
                action = np.array([-DEMOFORCE,0])
            elif key == ord('a'):
                action = np.array([0,-DEMOFORCE])
            elif key == ord('s'):
                action = np.array([DEMOFORCE,0])
            elif key == ord('d'):
                action = np.array([0,DEMOFORCE])
            print(action)
        else:
            raise ValueError("Invalid keyboard scheme")
        if self.renderer.orientation == 'vertical':
            action = np.array([action[0], action[1]])
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
        for i in range(1000000):
            if i % 1000 == 0:
                print("fps", 1000 / (time.time() - start))
                start = time.time()
            action = self.demonstrate()
            _, rew, _, _, _ = self.air_hockey.step(action)
            # if self.print_reward:
            print("reward: ", rew)
            if i % 2000 == 0:
                self.air_hockey.reset()
                
    def play_against_agent(self, policy):
        """
        Plays the air hockey game against an agent.

        Iterates through a loop, capturing user input and updating the game state.
        Prints the frames per second (fps) every 1000 iterations.
        Resets the game state every 300 iterations.

        Parameters:
        policy (function): The policy function of the agent.

        Returns:
        None
        """
        goal_pos = np.array([self.air_hockey.length / 2, 0]) # this is the home base, will be reversed for the alt policy
        (ego_obs, alt_obs), _ = self.air_hockey.reset(alt_goal_pos=goal_pos, ego_goal_pos=goal_pos, goal_radius_type='home')
        start = time.time()
        for i in range(1000000):
            if i % 1000 == 0:
                print("fps", 1000 / (time.time() - start))
                start = time.time()
            action = self.demonstrate()
            other_action = policy.predict(alt_obs, deterministic=True)[0]
            # invert other action because it's upside down
            other_action = np.array([-other_action[0], -other_action[1]])
            joint_action = (action, other_action)
            (ego_obs, alt_obs), (ego_rew, alt_rew), is_finished, truncated, info = self.air_hockey.step(joint_action)
            if i % 300 == 0:
                self.air_hockey.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demonstrate the air hockey game.')
    parser.add_argument('--cfg', type=str, default=None, help='Path to the configuration file.')
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

    demonstrator = Demonstrator(air_hockey_cfg)
    demonstrator.run()
    cv2.destroyAllWindows()