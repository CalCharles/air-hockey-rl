from Box2D.b2 import world
from Box2D import (b2CircleShape, b2FixtureDef, b2LoopShape, b2PolygonShape,
                   b2_dynamicBody, b2_staticBody, b2Filter)
from gymnasium import Env
import numpy as np
import random
from gymnasium.spaces import Box


class AirHockey2D(Env):
    def __init__(self, num_paddles, num_pucks, num_blocks, num_obstacles, num_targets, absorb_target, use_cue, length, width,
                 paddle_radius, reward_type,
                 force_scaling, paddle_damping, render_size, render_masks=False, max_timesteps=1000,  gravity=-5):
        self.gravity = gravity
        if type(self.gravity) == int:
            self.world = world(gravity=(0, self.gravity), doSleep=True)
        else: # it's a range
            self.world = world(gravity=(0, np.random.uniform(low=self.gravity[0], high=self.gravity[1])), doSleep=True)
        self.length, self.width = length, width
        self.force_scaling = force_scaling
        self.num_paddles = num_paddles
        self.paddle_radius = paddle_radius
        self.num_pucks = num_pucks
        self.num_blocks = num_blocks
        self.num_obstacles = num_obstacles
        self.num_targets = num_targets
        self.absorb_target = absorb_target
        self.goal_conditioned = True if reward_type == 'goal' else False
        self.render_width = int(render_size)
        self.ppm = render_size / self.width
        self.render_length = int(self.ppm * self.length)
        self.render_masks = render_masks
        self.puck_min_height = (-length / 2) + (length / 3)
        self.paddle_max_height = 0
        self.block_min_height = 0
        self.max_speed_start = width
        self.min_speed_start = 0
        self.paddle_damping = paddle_damping
        self.terminate_on_out_of_bounds = False
        self.use_cue = use_cue
        self.multiagent = num_paddles == 2
        self.ground_body = self.world.CreateBody(
            shapes=b2LoopShape(vertices=[(-width/2, -length/2),
                                         (-width/2, length/2), (width/2, length/2),
                                         (width/2, -length/2)]),
        )
        self.max_timesteps = max_timesteps
        self.current_timestep = 0
        
        low = np.array([-width/2, -length/2, -width/2, -length/2, -10, -10])
        high = np.array([width/2, length/2, width/2, length/2, 10, 10])
        
        # if goal-conditioned, then we need to add the goal position
        if self.goal_conditioned:
            self.min_goal_radius = width / 8
            self.max_goal_radius = width / 3
            low = np.array([-width/2, -length/2, -width/2, -length/2, -10, -10, -width/2, 0, self.min_goal_radius])
            high = np.array([width/2, length/2, width/2, length/2, 10, 10, width/2, length/2, self.max_goal_radius])
            self.observation_space = Box(low=low, high=high, shape=(9,), dtype=np.float64)
        else:
            self.observation_space = Box(low=low, high=high, shape=(6,), dtype=np.float64)
        
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32) # 2D action space

        if self.goal_conditioned:
            self.reward_type = 'goal'
        else:
            self.reward_type = reward_type

        self.reward_range = Box(low=0, high=1) # need to make sure rewards are between 0 and 1
        self.metadata = {}
        
        self.reset()

    @staticmethod
    def from_dict(state_dict):
        return AirHockey2D(**state_dict)

    def reset(self, seed=None, 
              goal_radius_type='random',
              ego_goal_pos=None,
              alt_goal_pos=None,
              object_state_dict=None, 
              type_instance_dict=None, 
              max_count_dict=None):
        if seed is None:
            seed = np.random.randint(10e8)
        np.random.seed(seed)
        if hasattr(self, "object_dict"):
            for body in self.object_dict.values():
                self.world.DestroyBody(body)
        if type(self.gravity) == list:
            self.world.gravity = (0, np.random.uniform(low=self.gravity[0], high=self.gravity[1]))
        self.paddles = dict()
        self.pucks = dict()
        self.blocks = dict()
        self.obstacles = dict()
        self.targets = dict()
        self.paddle_attrs = None
        self.target_attrs = None
        self.current_timestep = 0
        if self.goal_conditioned:
            if goal_radius_type == 'random':
                ego_goal_radius = np.random.uniform(low=self.min_goal_radius, high=self.max_goal_radius)
                if self.multiagent:
                    alt_goal_radius = np.random.uniform(low=self.min_goal_radius, high=self.max_goal_radius)          
                self.ego_goal_radius = ego_goal_radius
                if self.multiagent:
                    self.alt_goal_radius = alt_goal_radius
            elif goal_radius_type == 'home':
                self.ego_goal_radius = 0.16 * self.width
                if self.multiagent:
                    self.alt_goal_radius = 0.16 * self.width
            # uniform distribution
            if ego_goal_pos is None:
                # should fit in the x range [-width/2, width/2] and y range [-length/2, length/2]
                # however, we also need to take into account radius
                # -> [-width / 2 + ego_goal_radius, width / 2 - ego_goal_radius] and [-length / 2 + ego_goal_radius, length / 2 - ego_goal_radius
                min_x = -self.width / 2 + self.ego_goal_radius
                max_x = self.width / 2 - self.ego_goal_radius
                min_y = 0 + self.ego_goal_radius
                max_y = self.length / 2 - self.ego_goal_radius
                self.ego_goal_pos = np.random.uniform(low=(min_y, min_x), high=(max_y, max_x))
                if self.multiagent:
                    self.alt_goal_pos = np.random.uniform(low=(-self.length / 2, -self.width / 2), high=(0, self.width / 2))
            else: # If they share a goal, this might be a good testbed for human ai coordination
                self.ego_goal_pos = ego_goal_pos
                if self.multiagent:
                    self.alt_goal_pos = alt_goal_pos
        for i in range(self.num_pucks):
            rad = max(0.25, np.random.rand() * (self.width/ 8))
            name, puck_attrs = self.create_puck(i, radius = rad, min_height=self.puck_min_height)
            self.pucks[name] = puck_attrs

        for i in range(self.num_blocks):
            name, block_attrs = self.create_block_type(i, name_type = "Block", dynamic=False, min_height = self.block_min_height)
            self.blocks[name] = block_attrs

        for i in range(self.num_obstacles): # could replace with arbitary polygons
            name, obs_attrs = self.create_block_type(i, name_type = "Obstacle", angle=np.random.rand() * np.pi, dynamic = False, color=(0, 127, 127), min_height = self.block_min_height)
            self.obstacles[name] = obs_attrs

        for i in range(self.num_targets):
            name, target_attrs = self.create_block_type(i, name_type = "Target", color=(255, 255, 0))
            self.targets[name] = target_attrs
        
        if not self.multiagent:
            name, paddle_attrs = self.create_paddle(i, name="paddle_ego", density=1000, ldamp=self.paddle_damping, color=(0, 255, 0), max_height=self.paddle_max_height)
            self.paddles[name] = paddle_attrs
        else:
            name_home, paddle_ego_attrs = self.create_paddle(i=i, name="paddle_ego", density=1000, ldamp=self.paddle_damping, color=(0, 255, 0), max_height=self.paddle_max_height, 
                                             home_paddle=True)
            name_other, paddle_other_attrs = self.create_paddle(i=i, name="paddle_alt", density=1000, ldamp=self.paddle_damping, color=(0, 255, 0), max_height=self.paddle_max_height, 
                                             home_paddle=False)
            self.paddles[name_home] = paddle_ego_attrs
            self.paddles[name_other] = paddle_other_attrs
        
        if self.use_cue:
            self.cue = self.create_puck(-1,name="Cue", radius=0.25, vel=(0,0), pos=(0,0), color=(200,100,0))
        else: self.cue = ("Cue", None)
        # names and object dict
        self.puck_names = list(self.pucks.keys())
        self.puck_names.sort()
        self.paddle_names = list(self.paddles.keys())
        self.block_names = list(self.blocks.keys())
        self.block_names.sort()
        self.obstacle_names = list(self.obstacles.keys())
        self.obstacle_names.sort()
        self.target_names = list(self.targets.keys())
        self.target_names.sort()
        self.object_dict = {**{name: self.pucks[name][0] for name in self.pucks.keys()},
                            **{name: self.paddles[name][0] for name in self.paddles.keys()},
                             **({self.cue[0]: self.cue[1][0]} if self.cue[1] is not None else dict()),
                             **{name: self.blocks[name][0] for name in self.blocks.keys()},
                             **{name: self.targets[name][0] for name in self.targets.keys()},
                             **{name: self.obstacles[name][0] for name in self.obstacles.keys()},
                             }
        ego_paddle_x_pos = self.paddles['paddle_ego'][0].position[0]
        ego_paddle_y_pos = self.paddles['paddle_ego'][0].position[1]
        puck_x_pos = self.pucks[self.puck_names[0]][0].position[0]
        puck_y_pos = self.pucks[self.puck_names[0]][0].position[1]
        puck_x_vel = self.pucks[self.puck_names[0]][0].linearVelocity[0]
        puck_y_vel = self.pucks[self.puck_names[0]][0].linearVelocity[1]
            

        if not self.multiagent:
            if not self.goal_conditioned:
                obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel])
            else:
                obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel,
                                self.ego_goal_pos[0], self.ego_goal_pos[1], self.ego_goal_radius])
        else:
            alt_paddle_x_pos = self.paddles['paddle_alt'][0].position[0]
            alt_paddle_y_pos = self.paddles['paddle_alt'][0].position[1]
            
            if not self.goal_conditioned:
                obs_ego = np.array([ego_paddle_x_pos, ego_paddle_y_pos, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel])
                obs_alt = np.array([-alt_paddle_x_pos, -alt_paddle_y_pos, -puck_x_pos, -puck_y_pos, -puck_x_vel, -puck_y_vel])
            else:
                obs_ego = np.array([ego_paddle_x_pos, ego_paddle_y_pos, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel,
                                    self.ego_goal_pos[0], self.ego_goal_pos[1], self.ego_goal_radius])
                obs_alt = np.array([-alt_paddle_x_pos, -alt_paddle_y_pos, -puck_x_pos, -puck_y_pos, -puck_x_vel, -puck_y_vel,
                                    -self.alt_goal_pos[0], -self.alt_goal_pos[1], self.alt_goal_radius])
            obs = (obs_ego, obs_alt)
        return obs, {}
        
    def create_paddle(self, i, 
                        name=None, 
                        color=(127, 127, 127), 
                        density=10, 
                        vel=None, 
                        pos=None, 
                        ldamp=1, 
                        collidable=True, 
                        min_height=0, 
                        max_height=60,
                        home_paddle=True):
        if not self.multiagent:
            if pos is None:
                pos = (0, -self.length / 2 + 0.01) # start at home region
            # below code is for a random position
            # if pos is None: pos = ((np.random.rand() - 0.5) * 2 * (self.width / 2), 
            #                        max(min_height,-self.length / 2) + (np.random.rand() * ((min(max_height,self.length / 2)) - (max(min_height,-self.length / 2)))))
        else:
            if pos is None: 
                if home_paddle:
                    pos = (0, -self.length / 2 + 0.01)
                else:
                    pos = (0, self.length / 2 - 0.01)
                    
        if vel is None: 
            vel = (np.random.rand() * (self.max_speed_start - self.min_speed_start) + self.min_speed_start,
                   np.random.rand() * (self.max_speed_start - self.min_speed_start) + self.min_speed_start)
        radius = self.paddle_radius
        paddle = self.world.CreateDynamicBody(
            fixtures=b2FixtureDef(
                shape=b2CircleShape(radius=radius),
                density=100.0,
                restitution = 1.0,
                filter=b2Filter (maskBits=1,
                                 categoryBits=1 if collidable else 0)),
            bullet=True,
            position=pos,
            # linearVelocity=0.0,
            linearDamping=ldamp
        )
        color =  color # randomize color
        default_paddle_name = "paddle" + str(i)
        paddle.gravityScale = 0
        return ((default_paddle_name, (paddle, color)) if name is None else (name, (paddle, color)))

    # puck = bouncing ball
    def create_puck(self, i, 
                        name=None, 
                        color=(127, 127, 127), 
                        radius=-1,
                        density=10, 
                        vel=None, 
                        pos=None, 
                        ldamp=0.1, 
                        collidable=True,
                        min_height=-30,
                        max_height=30):
        if not self.multiagent:
            # then we want it to start at the top, which is max_height, 0
            if pos is None: 
                pos = ((np.random.rand() - 0.5) * 2 * (self.width / 2),
                       min(max_height, self.length / 2) - 0.01)
        else: 
            if pos is None: 
                pos = ((np.random.rand() - 0.5) * 2 * (self.width / 2), 
                       max(min_height,-self.length / 2) + (np.random.rand() * ((min(max_height,self.length / 2)) - (max(min_height,-self.length / 2)))))
        # print(name, pos, min_height, max_height)
        if not self.multiagent:
            if vel is None: 
                vel = (np.random.rand() * (self.max_speed_start - self.min_speed_start) + self.min_speed_start,
                       30 if random.random() > 0.5 else -30)
        else:
            if vel is None: 
                vel = (np.random.rand() * (self.max_speed_start - self.min_speed_start) + self.min_speed_start,
                       10 * np.random.rand() * (self.max_speed_start - self.min_speed_start) + self.min_speed_start)
        if radius < 0: 
            radius = max(1, np.random.rand() * (self.width/ 5))
        puck = self.world.CreateDynamicBody(
            fixtures=b2FixtureDef(
                shape=b2CircleShape(radius=radius),
                density=1.0,
                restitution = 1.0,
                filter=b2Filter (maskBits=1,
                                 categoryBits=1 if collidable else 0)),
            bullet=True,
            position=pos,
            linearVelocity=vel,
            linearDamping=ldamp
        )
        color =  color # randomize color
        puck_name = "puck" + str(i)
        return ((puck_name, (puck, color)) if name is None else (name, (puck, color)))

    def create_block_type(self, i, name=None,name_type=None, color=(127, 127, 127), width=-1, height=-1, vel=None, pos=None, dynamic=True, angle=0, angular_vel=0, fixed_rotation=False, collidable=True, min_height=-30):
        if pos is None: pos = ((np.random.rand() - 0.5) * 2 * (self.width / 2), min_height + (np.random.rand() * (self.length - (min_height + self.length / 2))))
        if vel is None: vel = ((np.random.rand() - 0.5) * 2 * (self.width),(np.random.rand() - 0.5) * 2 * (self.length))
        if not dynamic: vel = np.zeros((2,))
        if width < 0: width = max(0.75, np.random.rand() * 3)
        if height < 0: height = max(0.5, np.random.rand())
        # TODO: possibly create obstacles of arbitrary shape
        vertices = [([-width / 2, -height / 2]), ([width / 2, -height / 2]), ([width / 2, height / 2]), ([-width / 2, height / 2])]
        block_name  = name_type # Block, Obstacle, Target

        fixture = b2FixtureDef(
            shape=b2PolygonShape(vertices=vertices),
            density=1,
            restitution=0.1,
            filter=b2Filter (maskBits=1,
                                 categoryBits=1 if collidable else 0),
        )

        body = self.world.CreateBody(type=b2_dynamicBody if dynamic else b2_staticBody,
                                    position=pos,
                                    linearVelocity=vel,
                                    angularVelocity=angular_vel,
                                    angle=angle,
                                    fixtures=fixture,
                                    fixedRotation=fixed_rotation,
                                    )
        color =  color # randomize color
        block_name = block_name + str(i)
        return (block_name if name is None else name), (body, color)

    def step(self, action, other_action=None, time_step=0.018):
        if not self.multiagent:
            return self.single_agent_step(action, time_step)
        else:
            return self.multi_step(action, time_step)
        
    def is_within_goal_region(self, point, body, radius):
        point = np.array([point[1], point[0]])
        dist = np.linalg.norm(body.position - point)
        return dist < radius

    def is_within_home_region(self, point, body):
        # this is for the two base regions of each side of the eboard
        # TODO: this may need to be tuned :) let's provide a rough estimate of where the goal is
        # 90 / 560 = 0.16 <- normalized dist in pixels
        return self.is_within_goal_region(point, body, 0.16 * self.width)

    def has_finished(self, multiagent=False):
        truncated = False
        terminated = False
        puck_within_alt_home = False
        puck_within_home = False

        if self.current_timestep > self.max_timesteps:
            terminated = True
        else:
            if self.terminate_on_out_of_bounds:
                if self.paddles['paddle_ego'][0].position[1] > 0: 
                    truncated = True

        # confusing, but we need to swap x and y for this function
        bottom_center_point = np.array([-self.length / 2, 0])
        puck_within_home = self.is_within_home_region(bottom_center_point, self.pucks[self.puck_names[0]][0])
        truncated = truncated or puck_within_home

        # confusing, but we need to swap x and y for this function
        top_center_point = np.array([self.length / 2, 0])
        puck_within_alt_home = self.is_within_home_region(top_center_point, self.pucks[self.puck_names[0]][0])

        if self.reward_type == 'alt_home':
            if not terminated and puck_within_alt_home:
                terminated = True
        if multiagent:
            truncated = False
            terminated = terminated or truncated or puck_within_alt_home or puck_within_home
        puck_within_ego_goal = False
        puck_within_alt_goal = False
        if self.goal_conditioned:
            if self.is_within_goal_region(self.ego_goal_pos, self.pucks[self.puck_names[0]][0], self.ego_goal_radius):
                puck_within_ego_goal = True
                if multiagent:
                    if self.is_within_goal_region(self.alt_goal_pos, self.pucks[self.puck_names[0]][0], self.alt_goal_radius):
                        puck_within_alt_goal = True
        return terminated, truncated, puck_within_home, puck_within_alt_home, puck_within_ego_goal, puck_within_alt_goal
    
    def get_reward(self, hit_target, puck_within_home, puck_within_alt_home, puck_within_goal):
        if self.reward_type == 'goal':
            reward = 1 if puck_within_goal else 0
        elif self.reward_type == 'puck_height':
            reward = self.pucks[self.puck_names[0]][0].position[1]# - self.paddle[1][0].position[1]
            # min acceptable reward is 0 height and above
            reward = max(reward, 0)
            # let's normalize reward w.r.t. the top half length of the table
            # aka within the range [0, self.length / 2]
            max_rew = self.length / 2
            min_rew = 0
            reward = (reward - min_rew) / (max_rew - min_rew)
        elif self.reward_type == 'puck_vel':
        # reward for positive velocity towards the right side of the board
            reward = self.pucks[self.puck_names[0]][0].linearVelocity[0] / self.max_speed_start
            min_vel = 0.5
            reward = max(reward, 0)
            reward = min(reward, min_vel) if reward > 0 else 0
        elif self.reward_type == 'puck_touch':
            reward = 1 if hit_target else 0
        elif self.reward_type == 'alt_home':
            reward = 1 if puck_within_alt_home else 0
        return reward
    
    def get_joint_reward(self, ego_hit_target, alt_hit_target, 
                         puck_within_ego_home, puck_within_alt_home,
                         puck_within_ego_goal, puck_within_alt_goal):
        ego_reward = 0
        if self.reward_type == 'goal':
            ego_reward = 1 if puck_within_ego_goal else 0
        elif self.reward_type == 'puck_height':
            ego_reward = self.pucks[self.puck_names[0]][0].position[1]# - self.paddle[1][0].position[1]
            # min acceptable reward is 0 height and above
            ego_reward = max(ego_reward, 0)
            # let's normalize reward w.r.t. the top half length of the table
            # aka within the range [0, self.length / 2]
            max_rew = self.length / 2
            min_rew = 0
            ego_reward = (ego_reward - min_rew) / (max_rew - min_rew)
        elif self.reward_type == 'puck_vel':
            # reward for positive velocity towards the right side of the board
            ego_reward = self.pucks[self.puck_names[0]][0].linearVelocity[0] / self.max_speed_start
            min_vel = 0.5
            ego_reward = max(ego_reward, 0)
            ego_reward = min(ego_reward, min_vel) if ego_reward > 0 else 0
        elif self.reward_type == 'puck_touch':
            ego_reward = 1 if ego_hit_target else 0
        elif self.reward_type == 'alt_home':
            ego_reward = 1 if puck_within_alt_home else 0
            
        alt_reward = 0
        if self.reward_type == 'goal':
            alt_reward = 1 if puck_within_alt_goal else 0
        elif self.reward_type == 'puck_height':
            alt_reward = - self.pucks[self.puck_names[0]][0].position[1]# - self.paddle[1][0].position[1]
            # min acceptable reward is 0 height and above
            alt_reward = max(alt_reward, 0)
            # let's normalize reward w.r.t. the top half length of the table
            # aka within the range [0, self.length / 2]
            max_rew = self.length / 2
            min_rew = 0
            alt_reward = (alt_reward - min_rew) / (max_rew - min_rew)
        elif self.reward_type == 'puck_vel':
            alt_reward = -self.pucks[self.puck_names[0]][0].linearVelocity[0] / self.max_speed_start
            min_vel = 0.5
            alt_reward = max(alt_reward, 0)
            alt_reward = min(alt_reward, min_vel) if alt_reward > 0 else 0
        elif self.reward_type == 'puck_touch':
            alt_reward = 1 if alt_hit_target else 0
        elif self.reward_type == 'alt_home':
            alt_reward = 1 if puck_within_ego_home else 0

        return ego_reward, alt_reward

    def single_agent_step(self, action, time_step=0.018):
        force = self.force_scaling * self.paddles['paddle_ego'][0].mass * np.array((action[0], action[1])).astype(float)
        if self.paddles['paddle_ego'][0].position[1] > self.paddle_max_height: force[1] = min(self.force_scaling * self.paddles['paddle_ego'][0].mass * action[1], 0)
        if 'paddle_ego' in self.paddles: 
            self.paddles['paddle_ego'][0].ApplyForceToCenter(force, True)

        self.world.Step(time_step, 10, 10)
        contacts, contact_names = self.get_contacts()
        hit_target = self.respond_contacts(contact_names)
        # hacky way of determing if puck was hit below TODO: fix later!
        hit_target = np.any(contacts) # check if any are true
        is_finished, truncated, puck_within_home, puck_within_alt_home, puck_within_goal, _ = self.has_finished()
        if not truncated:
            reward = self.get_reward(hit_target, puck_within_home, puck_within_alt_home, puck_within_goal)
        else:
            reward = -0.01
        self.current_timestep += 1
        
        ego_paddle_x_pos = self.paddles['paddle_ego'][0].position[0]
        ego_paddle_y_pos = self.paddles['paddle_ego'][0].position[1]
        puck_x_pos = self.pucks[self.puck_names[0]][0].position[0]
        puck_y_pos = self.pucks[self.puck_names[0]][0].position[1]
        puck_x_vel = self.pucks[self.puck_names[0]][0].linearVelocity[0]
        puck_y_vel = self.pucks[self.puck_names[0]][0].linearVelocity[1]
        
        if not self.goal_conditioned:
            obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel])
        else:
            obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel,
                            self.ego_goal_pos[0], self.ego_goal_pos[1], self.ego_goal_radius])
        return obs, reward, is_finished, truncated, {}
    
    def multi_step(self, joint_action, time_step=0.018):
        action_ego, action_alt = joint_action
        force_ego = self.force_scaling * self.paddles['paddle_ego'][0].mass * np.array((action_ego[0], action_ego[1])).astype(float)
        force_alt = self.force_scaling * self.paddles['paddle_alt'][0].mass * np.array((action_alt[0], action_alt[1])).astype(float)
        
        # legacy code snippet
        if self.paddles['paddle_ego'][0].position[1] > 0: force_ego[1] = min(self.force_scaling * self.paddles['paddle_ego'][0].mass * action_ego[1], 0)
        if self.paddles['paddle_alt'][0].position[1] < 0: force_alt[1] = min(self.force_scaling * self.paddles['paddle_alt'][0].mass * action_alt[1], 0)

        self.paddles['paddle_ego'][0].ApplyForceToCenter(force_ego, True)
        self.paddles['paddle_alt'][0].ApplyForceToCenter(force_alt, True)

        self.world.Step(time_step, 10, 10)
        contacts, contact_names = self.get_contacts()
        hit_target = self.respond_contacts(contact_names)
        
        # hacky way of determing if puck was hit below TODO: fix later!
        hit_target = np.any(contacts) # check if any are true
        
        # TODO: fix later!
        egp_hit_target = hit_target
        alt_hit_target = hit_target
        
        is_finished, truncated, puck_within_home, puck_within_alt_home, puck_within_ego_goal, puck_within_alt_goal = self.has_finished(multiagent=True)
        ego_reward, alt_reward = self.get_joint_reward(egp_hit_target, alt_hit_target, 
                                                       puck_within_home, puck_within_alt_home, 
                                                       puck_within_ego_goal, puck_within_alt_goal)
        self.current_timestep += 1

        ego_paddle_x_pos = self.paddles['paddle_ego'][0].position[0]
        ego_paddle_y_pos = self.paddles['paddle_ego'][0].position[1]
        puck_x_pos = self.pucks[self.puck_names[0]][0].position[0]
        puck_y_pos = self.pucks[self.puck_names[0]][0].position[1]
        puck_x_vel = self.pucks[self.puck_names[0]][0].linearVelocity[0]
        puck_y_vel = self.pucks[self.puck_names[0]][0].linearVelocity[1]
        
        alt_paddle_x_pos = self.paddles['paddle_alt'][0].position[0]
        alt_paddle_y_pos = self.paddles['paddle_alt'][0].position[1]
        
        if not self.goal_conditioned:
            obs_ego = np.array([ego_paddle_x_pos, ego_paddle_y_pos, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel])
            obs_alt = np.array([-alt_paddle_x_pos, -alt_paddle_y_pos, -puck_x_pos, -puck_y_pos, -puck_x_vel, -puck_y_vel])
        else:
            obs_ego = np.array([ego_paddle_x_pos, ego_paddle_y_pos, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel,
                                self.ego_goal_pos[0], self.ego_goal_pos[1], self.ego_goal_radius])
            obs_alt = np.array([-alt_paddle_x_pos, -alt_paddle_y_pos, -puck_x_pos, -puck_y_pos, -puck_x_vel, -puck_y_vel,
                                -self.alt_goal_pos[0], -self.alt_goal_pos[1], self.alt_goal_radius])

        return (obs_ego, obs_alt), (ego_reward, alt_reward), is_finished, truncated, {}

    def get_contacts(self):
        contacts = list()
        shape_pointers = ([self.paddles[bn][0] for bn in self.paddle_names]  + \
                        ([self.cue[1][0]] if self.cue[1] is not None else list()) + \
                         [self.pucks[bn][0] for bn in self.puck_names] + [self.blocks[pn][0] for pn in self.block_names] + \
                         [self.obstacles[pn][0] for pn in self.obstacle_names] + [self.targets[pn][0] for pn in self.target_names])
        names = self.paddle_names + self.puck_names + self.block_names + self.obstacle_names + self.target_names + \
                ([self.cue[0]] if self.cue[1] is not None else list())
        # print(list(self.object_dict.keys()))
        contact_names = {n: list() for n in names}
        for bn in names:
            all_contacts = np.zeros(len(shape_pointers)).astype(bool)
            for contact in self.object_dict[bn].contacts:
                if contact.contact.touching:
                    contact_bool = np.array([(contact.other == bp and contact.contact.touching) for bp in shape_pointers])
                    contact_names[bn] += [sn for sn, bp in zip(names, shape_pointers) if (contact.other == bp)]
                else:
                    contact_bool = np.zeros(len(shape_pointers)).astype(bool)
                all_contacts += contact_bool
            contacts.append(all_contacts)
        return np.stack(contacts, axis=0), contact_names

    def respond_contacts(self, contact_names):
        hit_target = list()
        for tn in self.target_names:
            for cn in contact_names[tn]: 
                if cn.find("puck") != -1:
                    hit_target.append(cn)
        if self.absorb_target:
            for cn in hit_target:
                self.world.DestroyBody(self.object_dict[cn])
                del self.object_dict[cn]
        return hit_target # TODO: record a destroyed flag
