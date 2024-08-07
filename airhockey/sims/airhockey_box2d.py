from Box2D.b2 import world, contactListener
from Box2D import (b2CircleShape, b2FixtureDef, b2LoopShape, b2PolygonShape,
                   b2_dynamicBody, b2_staticBody, b2Filter, b2Vec2)
import numpy as np
import yaml
import inspect
from types import SimpleNamespace
from ..utils import dict_to_namespace

from matplotlib import pyplot as plt
import pstats

class CollisionForceListener(contactListener):
    def __init__(self, wall_bounce_scale=0.01):
        contactListener.__init__(self)
        self.collision_forces = list()
        self.wall_bounce_scale = wall_bounce_scale
    
    def reset(self):
        del self.collision_forces
        self.collision_forces = list()

    def PostSolve(self, contact, impulse):
        fixtureA = contact.fixtureA
        fixtureB = contact.fixtureB
        bodyA = fixtureA.body
        bodyB = fixtureB.body
        world_manifold = contact.worldManifold

        # Calculate the forces for each contact point
        for i in range(contact.manifold.pointCount):
            if i < len(impulse.normalImpulses):
                normal_impulse = impulse.normalImpulses[i]
                normal = world_manifold.normal

                self.collision_forces.append({
                    'bodyA': bodyA.userData,
                    'bodyB': bodyB.userData,
                    'normal_force': normal_impulse / 60.0,
                    'contact_normal': (normal.x, normal.y)
                })

                # If puck is involved, nudge it away from the wall
                if bodyA.userData is not None and "puck" in bodyA.userData:
                    speed = np.linalg.norm(bodyA.GetLinearVelocityFromWorldPoint(contact.worldManifold.points[i]))
                    bodyA.ApplyLinearImpulse(normal * self.wall_bounce_scale * speed, bodyA.worldCenter, True)
                if bodyB.userData is not None and "puck" in bodyB.userData:
                    speed = np.linalg.norm(bodyB.GetLinearVelocityFromWorldPoint(contact.worldManifold.points[i]))
                    bodyB.ApplyLinearImpulse(normal * self.wall_bounce_scale * speed, bodyB.worldCenter, True)

class AirHockeyBox2D:
    def __init__(self, **kwargs):
        defaults = {
            'action_x_scaling': 1.0,
            'action_y_scaling': 1.0,
            'render_masks': False,
            'gravity': -5,
            'paddle_density': 1000,
            'puck_density': 250,
            'block_density': 1000,
            'max_paddle_vel': 2,
            'time_frequency': 20,
            'paddle_bounds': [],
            'paddle_edge_bounds': [],
            'center_offset_constant': 1.2
        }

        kwargs = {**defaults, **kwargs}
        config = dict_to_namespace(kwargs)

        # physics / world params
        self.length, self.width = config.length, config.width
        self.paddle_radius = config.paddle_radius
        self.puck_radius = config.puck_radius
        self.block_width = config.block_width
        self.max_force_timestep = config.max_force_timestep
        self.time_frequency = config.time_frequency
        self.time_per_step = 1 / self.time_frequency
        self.force_scaling = config.force_scaling
        self.absorb_target = config.absorb_target
        self.paddle_damping = config.paddle_damping
        self.puck_damping = config.puck_damping
        self.gravity = config.gravity
        self.puck_min_height = (-config.length / 2) + (config.length / 3)
        self.paddle_max_height = 0
        self.block_min_height = 0
        self.max_speed_start = config.width
        self.min_speed_start = -config.width
        self.paddle_density = config.paddle_density
        self.puck_density = config.puck_density
        self.block_density = config.block_density
        self.action_x_scaling = config.action_x_scaling
        self.action_y_scaling = config.action_y_scaling
        self.center_offset_constant = config.center_offset_constant
        self.wall_bounce_scale = config.wall_bounce_scale

        # these assume 2d, in 3d since we have height it would be higher mass
        self.paddle_mass = self.paddle_density * np.pi * self.paddle_radius ** 2
        self.puck_mass = self.puck_density * np.pi * self.puck_radius ** 2
        self.chump_dict = {}
        # these 2 will depend on the other parameters
        self.max_paddle_vel = config.max_paddle_vel # m/s. This will be dependent on the robot arm
        # compute maximum force based on max paddle velocity
        max_a = self.max_paddle_vel / self.time_per_step
        max_f = self.paddle_mass * max_a
        # assume maximum force transfer
        puck_max_a = max_f / self.puck_mass
        self.max_puck_vel = puck_max_a * self.time_per_step
        self.world = world(gravity=(0, self.gravity), doSleep=True) # gravity is negative usually

        # box2d visualization params (but the visualization is done in the Render file)
        self.ppm = config.render_size / self.width
        self.render_width = int(config.render_size)
        self.render_length = int(self.ppm * self.length)
        self.render_masks = config.render_masks

        self.table_x_min = -self.width / 2
        self.table_x_max = self.width / 2
        self.table_y_min = -self.length / 2
        self.table_y_max = self.length / 2

        self.min_goal_radius = self.width / 16
        self.max_goal_radius = self.width / 4

        self.metadata = {}

        # creating the ground -- need to only call once! otherwise it can be laggy
        self.ground_body = self.world.CreateBody(
            shapes=b2LoopShape(vertices=[(self.table_x_min, self.table_y_min),
                                         (self.table_x_min, self.table_y_max),
                                         (self.table_x_max, self.table_y_max),
                                         (self.table_x_max, self.table_y_min)]),
        )
        # self.ground_body.fixtures[0].friction = 0.0
        self.reset(config.seed)

        # Initialize the contact listener
        self.collision_listener = CollisionForceListener(wall_bounce_scale=self.wall_bounce_scale)
        self.world.contactListener = self.collision_listener
        self.total_timesteps = 0
        from cProfile import Profile
        from pstats import SortKey, Stats
        self.profiler = Profile()

    def start_callbacks(self, **kwargs):
        return

    @staticmethod
    def from_dict(state_dict):
        # create a dictionary of only the relevant parameters
        return AirHockeyBox2D(**state_dict)

    def reset(self, seed, **kwargs):
        self.rng = np.random.RandomState(seed)
        self.timestep = 0

        if hasattr(self, "object_dict"):
            for body in self.object_dict.values():
                self.world.DestroyBody(body)

        if type(self.gravity) == list:
            self.world.gravity = (0, self.rng.uniform(low=self.gravity[0], high=self.gravity[1]))
        
        if hasattr(self, "collision_listener"): self.collision_listener.reset()

        self.paddles = dict()
        self.pucks = dict()
        self.blocks = dict()
        self.block_initial_positions = dict()
        self.obstacles = dict()
        self.targets = dict()
        
        self.multiagent = False

        self.puck_history = list()
        self.paddle_attrs = None
        self.target_attrs = None

        self.object_dict = dict()
        state_info = self.get_current_state()
        return state_info
    
    def convert_from_box2d_coords(self, state_info):
        # traverse through state_info until we find tuple, then correct
        for key, value in state_info.items():
            if type(value) == list:
                for i in range(len(value)):
                    for key2, value2 in value[i].items():
                        if type(value2) == tuple:
                            state_info[key][i][key2] = (-value2[1], value2[0])
            else:
                for key2, value2 in value.items():
                    for key3, value3 in value2.items():
                        state_info[key][key2][key3] = (-value3[1], value3[0])
        return state_info
    
    def base_coord_to_box2d(self, coord):
        return (coord[1], -coord[0])
    
    def get_current_state(self):

        state_info = {}
        
        if 'paddle_ego' in self.paddles:
            ego_paddle_x_pos = self.paddles['paddle_ego'].position[0]
            ego_paddle_y_pos = self.paddles['paddle_ego'].position[1]
            ego_paddle_x_vel = self.paddles['paddle_ego'].linearVelocity[0]
            ego_paddle_y_vel = self.paddles['paddle_ego'].linearVelocity[1]
            ego_paddle_x_acc = self.paddles['paddle_ego_acceleration'][0]
            ego_paddle_y_acc = self.paddles['paddle_ego_acceleration'][0]
            ego_paddle_x_force = self.paddles['paddle_ego_force'][0]
            ego_paddle_y_force = self.paddles['paddle_ego_force'][0]
            
            state_info['paddles'] = {'paddle_ego': {'position': (ego_paddle_x_pos, ego_paddle_y_pos),
                                                    'velocity': (ego_paddle_x_vel, ego_paddle_y_vel),
                                                    'acceleration': (ego_paddle_x_acc, ego_paddle_y_acc),
                                                    'force': (ego_paddle_x_force, ego_paddle_y_force)
                                                    }}

        if 'paddle_alt' in self.paddles:
            alt_paddle_x_pos = self.paddles['paddle_alt'].position[0]
            alt_paddle_y_pos = self.paddles['paddle_alt'].position[1]
            alt_paddle_x_vel = self.paddles['paddle_alt'].linearVelocity[0]
            alt_paddle_y_vel = self.paddles['paddle_alt'].linearVelocity[1]
            
            state_info['paddles']['paddle_alt'] = {'position': (alt_paddle_x_pos, alt_paddle_y_pos),
                                                   'velocity': (alt_paddle_x_vel, alt_paddle_y_vel)}

        if len(self.blocks) > 0:
            state_info['blocks'] = []
            for block_name in self.blocks:
                block_x_pos = self.blocks[block_name].position[0]
                block_y_pos = self.blocks[block_name].position[1]
                initial_x_pos = self.block_initial_positions[block_name][0]
                initial_y_pos = self.block_initial_positions[block_name][1]

                state_info['blocks'].append({'current_position': (block_x_pos, block_y_pos),
                                        'initial_position': (initial_x_pos, initial_y_pos)})

        if len(self.pucks) > 0:
            state_info['pucks'] = []
            for puck_name in self.pucks:
                puck_x_pos = self.pucks[puck_name].position[0]
                puck_y_pos = self.pucks[puck_name].position[1]
                puck_x_vel = self.pucks[puck_name].linearVelocity[0]
                puck_y_vel = self.pucks[puck_name].linearVelocity[1]
                state_info['pucks'].append({'position': (puck_x_pos, puck_y_pos), 
                                'velocity': (puck_x_vel, puck_y_vel)})
        
        return self.convert_from_box2d_coords(state_info)
    
    def instantiate_objects(self):
        pass # we don't need to do anything here
    
    def spawn_paddle(self, pos, vel, name, affected_by_gravity=False, movable=True):
        assert name == 'paddle_ego' or name == 'paddle_alt'
        pos = self.base_coord_to_box2d(pos)
        vel = self.base_coord_to_box2d(vel)
        radius = self.paddle_radius
        paddle = self.world.CreateDynamicBody(
            fixtures=b2FixtureDef(
                shape=b2CircleShape(radius=radius),
                density=self.paddle_density,
                restitution = 1.0,
                filter=b2Filter (maskBits=1,
                                 categoryBits=1)),
            bullet=True,
            position=pos,
            linearVelocity=vel,
            linearDamping=self.paddle_damping
        )
        if not affected_by_gravity:
            paddle.gravityScale = 0
        
        self.paddles[name] = paddle
        if name == "paddle_ego":
            self.paddles['paddle_ego_acceleration'] = (0, 0)
            self.paddles['paddle_ego_force'] = (0, 0)
        self.object_dict[name] = paddle
        
        if 'paddle_ego' in self.paddles and 'paddle_alt' in self.paddles:
            self.multiagent = True
    
    def spawn_puck(self, pos, vel, name, affected_by_gravity=True, movable=True):
        pos = self.base_coord_to_box2d(pos)
        vel = self.base_coord_to_box2d(vel)
        radius = self.puck_radius
        puck = self.world.CreateDynamicBody(
            fixtures=b2FixtureDef(
                shape=b2CircleShape(radius=radius),
                density=self.puck_density,
                restitution = 1.0,
                filter=b2Filter (maskBits=1,
                                 categoryBits=1),
                friction=0.0),
            bullet=True,
            position=pos,
            linearVelocity=vel,
            linearDamping=self.puck_damping,
            angularDamping=100000,
            userData=name
        )
        if not affected_by_gravity:
            puck.gravityScale = 0
        self.pucks[name] = puck
        self.object_dict[name] = puck
        self.puck_history += [(-2,0,1) for i in range(5)]
        
    def spawn_block(self, pos, vel, name, affected_by_gravity=False, movable=True):
        pos = self.base_coord_to_box2d(pos)
        vel = self.base_coord_to_box2d(vel)
        vertices = [([-self.block_width / 2, -self.block_width / 2]), ([self.block_width / 2, -self.block_width / 2]), ([self.block_width / 2, self.block_width / 2]), ([-self.block_width / 2, self.block_width / 2])]
        block = self.world.CreateDynamicBody(
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(vertices=vertices),
                density=self.block_density,
                restitution=1.0,
                filter=b2Filter(maskBits=1, categoryBits=1)),
            bullet=True,
            position=pos,
            linearVelocity=vel,
            linearDamping=self.puck_damping
        )
        if not affected_by_gravity:
            block.gravityScale = 0
        self.blocks[name] = block
        self.block_initial_positions[name] = pos
        self.object_dict[name] = block

    def convert_to_box2d_coords(self, action):
        action = np.array((action[1], -action[0]))
        return action

    # s, a -> s'
    def get_transition(self, action, other_action=None):
        if self.multiagent:
            return self.get_multiagent_transition(action, other_action)
        else:
            action = self.convert_to_box2d_coords(action)
            return self.get_singleagent_transition(action)

    # @mprofile
    def get_singleagent_transition(self, action):

        # self.profiler.enable()  # Start profiling
        # check if out of bounds and correct
        pos = [self.paddles['paddle_ego'].position[0], self.paddles['paddle_ego'].position[1]]
        if pos[1] > 0 - 3 * self.paddle_radius:
            action[1] = min(action[1], 0)
        
        # action is delta position
        # let's use simple time-optimal control to figure out the force to apply
        delta_pos = np.array([action[0], action[1]])
        # if delta_pos[0] == 0 and delta_pos[1] == 0:
        #     force = np.array([0, 0])
        # else:
        current_vel = np.array([self.paddles['paddle_ego'].linearVelocity[0], self.paddles['paddle_ego'].linearVelocity[1]])
        
        # force = np.array([self.paddles['paddle_ego'][0].mass * accel[0], self.paddles['paddle_ego'][0].mass * accel[1]])
        
        # # first let's determine velocity
        vel = delta_pos / self.time_per_step
        vel_mag = np.linalg.norm(vel)
        vel_unit = vel / (vel_mag + 1e-8)

        if vel_mag > self.max_paddle_vel:
            vel = vel_unit * self.max_paddle_vel

        force = self.paddles['paddle_ego'].mass * vel / self.time_per_step
        force_mag = np.linalg.norm(force)
        force_unit = force / (force_mag + 1e-8)
        if force_mag > self.max_force_timestep:
            force = force_unit * self.max_force_timestep
            
        force = force.astype(float)
        if self.paddles['paddle_ego'].position[1] > 0: 
            new_force = self.force_scaling * self.paddles['paddle_ego'].mass * action[1]
            if new_force < -self.max_force_timestep:
                new_force = -self.max_force_timestep
            force[1] = min(new_force, 0)
        else:
            force = force * np.array([self.action_x_scaling, self.action_y_scaling])
        if 'paddle_ego' in self.paddles:
            self.paddles['paddle_ego'].ApplyForceToCenter(force, True)
        self.world.Step(self.time_per_step, 10, 10)
        
        # correct blocks for t=0
        if self.timestep == 0 and len(self.blocks) > 0:
            for block_name in self.blocks:
                block = self.blocks[block_name]
                x, y = self.block_initial_positions[block_name]
                block.position = (x, y)
        
        vel = np.array([self.paddles['paddle_ego'].linearVelocity[0], self.paddles['paddle_ego'].linearVelocity[1]])
        vel_mag = np.linalg.norm(vel)

        # keep velocity at a maximum value
        if vel_mag > self.max_paddle_vel:
            self.paddles['paddle_ego'].linearVelocity = b2Vec2(vel[0] / vel_mag * self.max_paddle_vel, vel[1] / vel_mag * self.max_paddle_vel)
            
        # check if out of bounds and correct
        pos = [self.paddles['paddle_ego'].position[0], self.paddles['paddle_ego'].position[1]]
        if pos[0] < self.table_x_min:
            pos[0] = self.table_x_min
        if pos[0] > self.table_x_max:
            pos[0] = self.table_x_max
        if pos[1] > 0:
            pos[1] = 0
        if pos[1] > self.table_y_max:
            pos[1] = self.table_y_max
        self.paddles['paddle_ego'].position = (pos[0], pos[1])
        
        state_info = self.get_current_state()
        if 'pucks' in state_info:
            for puck in state_info['pucks']:
                self.puck_history.append(list(puck["position"]) + [0])
        else:
            for i in range(len(self.pucks.keys())):
                self.puck_history.append([-2 + self.center_offset_constant,0,1])
        
        self.paddles['paddle_ego_acceleration'] = vel - current_vel

        total_force = np.array(force)

        collision_forces = self.get_collision_forces()
        for collision in collision_forces:
            if collision['bodyA'] == 'paddle_ego':
                total_force[0] += collision['normal_force'] * collision['contact_normal'][0]
                total_force[1] += collision['normal_force'] * collision['contact_normal'][1]
            elif collision['bodyB'] == 'paddle_ego':
                total_force[0] -= collision['normal_force'] * collision['contact_normal'][0]
                total_force[1] -= collision['normal_force'] * collision['contact_normal'][1]

        self.paddles['paddle_ego_force'] = total_force



        self.timestep += 1
        # self.total_timesteps += 1

        

        # self.profiler.disable()  # Stop profiling
        
        # # # if self.total_timesteps % 1000 == 0:
        # # #     # self.profiler.print_stats(sort='time')  # Print the statistics sorted by time
        # # #     # Save the statistics to a file
        
        # if self.total_timesteps % 1000 == 0:
        #     with open('single_agent_transition_profile.txt', 'w' if self.total_timesteps <= 1000 else 'a') as f:
        #         f.write(f'timesteps: {self.total_timesteps}\n')
        #         stats = pstats.Stats(self.profiler, stream=f)
            
        #         stats.sort_stats('time')
        #         # stats.print_stats()
        #         keys = stats.stats.keys()
                
        #         for key in keys:
        #             cbdk = ''.join([str(k) for k in key])
        #             self.chump_dict[cbdk].append(stats.stats[key][3]) if cbdk in self.chump_dict else self.chump_dict.update({cbdk: [stats.stats[key][3]]})
        #         stats.strip_dirs().sort_stats("cumtime").print_stats()
                
        #         f.write(f'------------------------------------\n')
        #     with open('data.yaml', 'w') as file:
        #         yaml.dump(self.chump_dict, file)
                
        #     # Plot the data
        #     plt.figure(figsize=(12, 8))

        #     for key, values in self.chump_dict.items():
        #         plt.plot(values, label=key)

        #     plt.xlabel('Index')
        #     plt.ylabel('Value')
        #     plt.title('cumulative run times in single agent transition function')
        #     plt.legend()
        #     plt.grid(True)
        #     plt.savefig('fuckmejeans.png')
        return state_info
    
    def get_multiagent_transition(self, joint_action):
        raise NotImplementedError

    def get_contacts(self):
        contacts = list()
        shape_pointers = ([self.paddles[bn][0] for bn in self.paddle_names]  + \
                         [self.pucks[bn][0] for bn in self.puck_names] + [self.blocks[pn][0] for pn in self.block_names] + \
                         [self.obstacles[pn][0] for pn in self.obstacle_names] + [self.targets[pn][0] for pn in self.target_names])
        names = self.paddle_names + self.puck_names + self.block_names + self.obstacle_names + self.target_names
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
        hit_a_puck = list()
        for tn in self.target_names:
            for cn in contact_names[tn]: 
                if cn.find("puck") != -1:
                    hit_a_puck.append(cn)
        if self.absorb_target:
            for cn in hit_a_puck:
                self.world.DestroyBody(self.object_dict[cn])
                del self.object_dict[cn]
        return hit_a_puck # TODO: record a destroyed flag

    def get_collision_forces(self):
        # Extract forces from the collision listener
        return self.collision_listener.collision_forces