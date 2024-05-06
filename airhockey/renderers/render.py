import numpy as np
import cv2
import os
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

TARGET_FPS = 60

class AirHockeyRenderer:
    """
    A class that renders the air hockey game.

    Args:
        airhockey_env (AirHockeySimulator): The air hockey simulator object.
        orientation (str, optional): The orientation of the game table. Defaults to 'vertical'.
    """

    def __init__(self, airhockey_env, orientation='vertical'):
        """
        Initializes the AirHockeyRenderer object.

        Args:
            airhockey_env (AirHockeySimulator): The air hockey simulator object.
            orientation (str, optional): The orientation of the game table. Defaults to 'vertical'.
        """
        self.airhockey_env = airhockey_env
        self.orientation = orientation
        # Adjust dimensions based on the specified orientation
        self.render_width = self.airhockey_env.render_width
        self.render_length = self.airhockey_env.render_length
        self.render_masks = self.airhockey_env.render_masks
        self.width = self.airhockey_env.width
        self.length = self.airhockey_env.length
        self.screen_width, self.screen_height = 120, 120
        self.ppm = self.airhockey_env.ppm 
        
        # get directory where this file is
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # make sure we can find assets folder
        assets_folder = os.path.abspath(os.path.join(dir_path, '../../assets'))
        assert os.path.exists(assets_folder), f"Could not find assets folder at {assets_folder}"
        
        air_hockey_table_fp = os.path.join(assets_folder, 'air_hockey_table.png')
        puck_fp = os.path.join(assets_folder, 'puck.png')
        paddle_fp = os.path.join(assets_folder, 'paddle.png')
        block_fp = os.path.join(assets_folder, 'block.png')
        
        # Load and resize the image
        self.paddle_img = cv2.imread(paddle_fp, cv2.IMREAD_UNCHANGED)  # Ensure the image has an alpha channel
        self.current_paddle_shape = None
        
        self.puck_img = cv2.imread(puck_fp, cv2.IMREAD_UNCHANGED)  # Ensure the image has an alpha channel
        self.current_puck_shape = None

        self.square_img = cv2.imread(block_fp, cv2.IMREAD_UNCHANGED)  # Ensure the image has an alpha channel
        self.current_square_shape = None
        
        self.air_hockey_table_img = cv2.imread(air_hockey_table_fp)
        # rotate clockwise 90 deg
        self.air_hockey_table_img = cv2.rotate(self.air_hockey_table_img, cv2.ROTATE_90_CLOCKWISE)
        self.air_hockey_table_img = cv2.resize(self.air_hockey_table_img, (self.render_length, self.render_width))
        
    def convert_to_render_coords_sys(self, pos):
        return np.array((pos[1], -pos[0]))  # coords -> box2d
            
    def draw_circle_with_image(self, pos, circle_radius, circle_type='puck'):
        """
        Draws a circle with an image overlay on the frame.

        Args:
            body_attrs (tuple): A tuple containing the body and color attributes of the circle.
            circle_type (str, optional): The type of circle. Defaults to 'puck'.
        """
        # center = np.array(body.position) + np.array((self.width / 2, self.length / 2))
        center = np.array(pos) + np.array((self.width / 2, self.length / 2))
        center = np.array((center[1], center[0])) * self.ppm  # Default horizontal orientation
        radius = int(circle_radius * self.ppm)
        
        # Calculate top-left corner of the image for overlay
        top_left = (center - radius).astype(int)
        diameter = 2 * radius
        bottom_right = top_left + diameter
        
        if circle_type == 'puck':
            if self.current_puck_shape != circle_radius:
                self.current_puck_shape = circle_radius
                radius = int(circle_radius * self.ppm)
                diameter = int(radius * 2)
                self.puck_img = cv2.resize(self.puck_img, (diameter, diameter))
            resized_img = self.puck_img
        elif circle_type == 'paddle':
            if self.current_paddle_shape != circle_radius:
                self.current_paddle_shape = circle_radius
                radius = int(circle_radius * self.ppm)
                diameter = int(radius * 2)
                self.paddle_img = cv2.resize(self.paddle_img, (diameter, diameter))
            resized_img = self.paddle_img
            
        # w.r.t. image, y_start == 0 if within frame, otherwise top_left is negative
        if top_left[1] < 0:
            y_start = max(0, -top_left[1] + 1)
        else:
            y_start = 0
        if top_left[0] < 0:
            x_start = max(0, -top_left[0] + 1)
        else:
            x_start = 0

        x_start = max(0, -top_left[0])
        y_start = max(0, -top_left[1])
        
        frame_top_left = [max(0, top_left[0]), max(0, top_left[1])]
        frame_bottom_right = [min(self.frame.shape[1], bottom_right[0]), min(self.frame.shape[0], bottom_right[1])]
        
        # w.r.t. image, y_end == resized_img.shape[0] if within frame, otherwise resized_img.shape[0] 
        y_end_offset = bottom_right[1] - frame_bottom_right[1]
        x_end_offset = bottom_right[0] - frame_bottom_right[0]
        y_end = resized_img.shape[0] - y_end_offset
        x_end = resized_img.shape[1] - x_end_offset
        
        # Overlay the image
        # check if circle is within the image
        if y_start >= resized_img.shape[0] or x_start >= resized_img.shape[1] or y_end <= 0 or x_end <= 0:
            print("Circle (puck) is out of bounds. Not rendering...")
            return
        mask = resized_img[y_start:y_end, x_start:x_end, 3] > 0
        self.frame[frame_top_left[1] : frame_bottom_right[1], frame_top_left[0]: frame_bottom_right[0]][mask] = resized_img[y_start:y_end, x_start:x_end, :3][mask]

    def check_position(self, pos, length, width):
        assert pos[0] >= -length / 2 and pos[0] <= length / 2, f"Position x-coordinate {pos[0]} is out of bounds {-length / 2} to {length / 2}"
        assert pos[1] >= -width / 2 and pos[1] <= width / 2, f"Position y-coordinate {pos[1]} is out of bounds {-width / 2} to {width / 2}"
        # assert pos[1] >= -length / 2 and pos[1] <= length / 2, f"Position x-coordinate {pos[1]} is out of bounds {-length / 2} to {length / 2}"
        # assert pos[0] >= -width / 2 and pos[0] <= width / 2, f"Position y-coordinate {pos[0]} is out of bounds {-width / 2} to {width / 2}"

    def draw_square_with_image(self, pos, block_width, square_type='block'):
        """
        Draws a square with an image overlay on the frame.

        Args:
            body_attrs (tuple): A tuple containing the body and color attributes of the circle.
            circle_type (str, optional): The type of circle. Defaults to 'puck'.
        """
        # first let's make sure it is within the dims        
        color = (0, 0, 255)
        # center = np.array(body.position) + np.array((self.width / 2, self.length / 2))
        center = np.array(pos) + np.array((self.width / 2, self.length / 2))
        center = np.array((center[1], center[0])) * self.ppm  # Default horizontal orientation
        # such as [([-width / 2, -height / 2]), ([width / 2, -height / 2]), ([width / 2, height / 2]), ([-width / 2, height / 2])]
        # width = np.abs(-2 * width)
        width = int(block_width * self.ppm)

        # Calculate top-left corner of the image for overlay
        top_left = (center - width / 2).astype(int)
        bottom_right = top_left + width

        if square_type == 'block':
            if self.current_puck_shape != width:
                self.current_puck_shape = width
                # such as [([-width / 2, -height / 2]), ([width / 2, -height / 2]), ([width / 2, height / 2]), ([-width / 2, height / 2])]
                # width = np.abs(-2 * width)
                width = int(block_width * self.ppm)
                self.square_img = cv2.resize(self.square_img, (width, width))
            resized_img = self.square_img

        # w.r.t. image, y_start == 0 if within frame, otherwise top_left is negative
        if top_left[1] < 0:
            y_start = max(0, -top_left[1] + 1)
        else:
            y_start = 0
        if top_left[0] < 0:
            x_start = max(0, -top_left[0] + 1)
        else:
            x_start = 0

        x_start = max(0, -top_left[0])
        y_start = max(0, -top_left[1])

        frame_top_left = [max(0, top_left[0]), max(0, top_left[1])]
        frame_bottom_right = [min(self.frame.shape[1], bottom_right[0]), min(self.frame.shape[0], bottom_right[1])]

        # w.r.t. image, y_end == resized_img.shape[0] if within frame, otherwise resized_img.shape[0]
        y_end_offset = bottom_right[1] - frame_bottom_right[1]
        x_end_offset = bottom_right[0] - frame_bottom_right[0]
        y_end = resized_img.shape[0] - y_end_offset
        x_end = resized_img.shape[1] - x_end_offset
        
        if y_start >= resized_img.shape[0] or x_start >= resized_img.shape[1] or y_end <= 0 or x_end <= 0:
            print("Square (block) is out of bounds. Not rendering...")
            return

        # Overlay the image
        mask = resized_img[y_start:y_end, x_start:x_end, 3] > 0
        self.frame[frame_top_left[1]: frame_bottom_right[1], frame_top_left[0]: frame_bottom_right[0]][mask] = resized_img[y_start:y_end, x_start:x_end, :3][mask]

    def draw_region(self, goal, radius, color=(0, 255, 0), shape="circle"):
        goal_position = goal
        goal_radius = radius
        goal_position = np.array((goal_position[1], -goal_position[0])) # coords -> box2d
        center = np.array(goal_position) + np.array((self.width / 2, self.length / 2))
        center = np.array((center[1], center[0])) * self.ppm
        goal_radius = (np.array(goal_radius) * self.ppm).astype(int) if isinstance(goal_radius, Iterable) else int(goal_radius * self.ppm)
        # print(center, goal_radius, shape)
        if shape == "circle": 
            if isinstance(goal_radius, Iterable): goal_radius = goal_radius[0] 
            cv2.circle(self.frame, center.astype(int), goal_radius, color, 2)
        elif shape == "ellipse": 
            goal_radius = np.array((goal_radius[1], goal_radius[0]))
            cv2.ellipse(self.frame, center.astype(int), (goal_radius[0], goal_radius[1]), 0, 0, 360, color, 2)
        elif shape == "rect": 
            # goal_radius = np.array((goal_radius[1], goal_radius[0])).astype(int)
            start_point = [int(center[0] - goal_radius[0]), int(center[1] - goal_radius[1])]
            end_point = [int(center[0] + goal_radius[0]), int(center[1] + goal_radius[1])]            
            
            cv2.rectangle(self.frame, start_point, end_point, color, 2)
        elif shape == "square": 
            start_point = [int(center[0] - goal_radius), int(center[1] - goal_radius)]
            end_point = [int(center[0] + goal_radius), int(center[1] + goal_radius)]            
            
            cv2.rectangle(self.frame, start_point, end_point, color, 2)

    def draw_polygon(self, body_attrs):
        """
        Draws a polygon on the frame.

        Args:
            body_attrs (tuple): A tuple containing the body and color attributes of the polygon.
        """
        color = (0, 0, 0)
        body = body_attrs
        for fixture in body.fixtures:
            shape = fixture.shape
            rotation = np.stack([body.transform.R.x_axis, body.transform.R.y_axis], axis = 1)
            vertices = [np.matmul(rotation, v) for v in shape.vertices]
            vertices = [body.position + v for v in vertices]
            vertices = [np.array(v) + np.array((self.width / 2, self.length / 2)) for v in vertices]
            vertices = [np.array((v[1], v[0])) * self.ppm for v in vertices]  # Default horizontal orientation
            vertices = np.array(vertices).astype(int)
            cv2.fillPoly(self.frame, pts=[vertices], color=color)

    def get_frame(self):
        """
        Gets the current frame of the air hockey game.

        Returns:
            numpy.ndarray: The frame of the air hockey game.
        """
        self.frame = self.air_hockey_table_img.copy()
        
        if self.airhockey_env.goal_conditioned:
            # get the goal position and radius and draw it
                
            green = (0, 255, 0)
            self.draw_region(self.airhockey_env.goal_pos, self.airhockey_env.goal_radius, color=green)
            if self.airhockey_env.multiagent:
                blue = (255, 0, 0)
                self.draw_region(self.airhockey_env.alt_goal_pos, self.airhockey_env.alt_goal_radius, color=blue)
        
        if len(self.airhockey_env.reward_regions) > 0:
            for region in self.airhockey_env.reward_regions:
                if region.reward_value < 0: color = (0,0,255)
                elif region.reward_value == 0: color = (0,0,0)
                else: color = (0,255,0)
                # print("region", region.state, region.radius, region.shape, color)
                self.draw_region(region.state, region.radius, color=color, shape=region.shape)


        
        state_info = self.airhockey_env.current_state
        if 'pucks' in state_info:
            for i in range(len(state_info['pucks'])):
                pos = state_info['pucks'][i]['position']
                # self.check_position(pos, self.length, self.width)
                pos = self.convert_to_render_coords_sys(pos)
                radius = self.airhockey_env.puck_radius
                self.draw_circle_with_image(pos, radius, circle_type='puck')
        if 'blocks' in state_info:
            for i in range(len(state_info['blocks'])):
                pos = state_info['blocks'][i]['current_position']
                # self.check_position(pos, self.length, self.width)
                pos = self.convert_to_render_coords_sys(pos)
                width = self.airhockey_env.block_width
                self.draw_square_with_image(pos, width, square_type='block')
        if 'paddles' in state_info:
            for paddle_name in state_info['paddles']:
                pos = state_info['paddles'][paddle_name]['position']
                # self.check_position(pos, self.length, self.width)
                pos = self.convert_to_render_coords_sys(pos)
                radius = self.airhockey_env.paddle_radius
                self.draw_circle_with_image(pos, radius, circle_type='paddle')
            
        # if self.airhockey_env.paddle[1] is not None: self.draw_circle_with_image(self.airhockey_env.paddle[1], circle_type='paddle')
        if self.orientation == 'vertical':
            self.frame = cv2.rotate(self.frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return self.frame

    def render(self):
        """
        Renders the air hockey game.
        """
        frame = self.get_frame()
        cv2.imshow('Air Hockey 2D', frame)
        cv2.waitKey(20)

