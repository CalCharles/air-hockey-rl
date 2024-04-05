import numpy as np
import cv2
import os

TARGET_FPS = 60

class AirHockeyRenderer:
    """
    A class that renders the air hockey game.

    Args:
        airhockey_sim (AirHockeySimulator): The air hockey simulator object.
        orientation (str, optional): The orientation of the game table. Defaults to 'vertical'.
    """

    def __init__(self, airhockey_env, orientation='vertical'):
        """
        Initializes the AirHockeyRenderer object.

        Args:
            airhockey_sim (AirHockeySimulator): The air hockey simulator object.
            orientation (str, optional): The orientation of the game table. Defaults to 'vertical'.
        """
        self.airhockey_env = airhockey_env
        self.airhockey_sim = airhockey_env.simulator
        self.orientation = orientation
        # Adjust dimensions based on the specified orientation
        self.render_width = self.airhockey_sim.render_width
        self.render_length = self.airhockey_sim.render_length
        self.render_masks = self.airhockey_sim.render_masks
        self.width = self.airhockey_sim.width
        self.length = self.airhockey_sim.length
        self.screen_width, self.screen_height = 120, 120
        self.ppm = self.airhockey_sim.ppm 
        
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
        
    
    def draw_circle(self, body_attrs):
        """
        Draws a circle on the frame.

        Args:
            body_attrs (tuple): A tuple containing the body and color attributes of the circle.
        """
        body, color = body_attrs
        for fixture in body.fixtures:
            shape = fixture.shape
            center = np.array(body.position) + np.array((self.width / 2, self.length / 2))
            center = np.array((center[1], center[0])) * self.ppm  # Default horizontal orientation
            radius = int(shape.radius * self.ppm)
            cv2.circle(self.frame, center.astype(int), radius, color, -1)
            
    def draw_circle_with_image(self, body_attrs, circle_type='puck'):
        """
        Draws a circle with an image overlay on the frame.

        Args:
            body_attrs (tuple): A tuple containing the body and color attributes of the circle.
            circle_type (str, optional): The type of circle. Defaults to 'puck'.
        """
        body, color = body_attrs  # color is unused but kept for compatibility
        for fixture in body.fixtures:
            shape = fixture.shape
            # center = np.array(body.position) + np.array((self.width / 2, self.length / 2))
            center = np.array(body.position) + np.array((self.width / 2, self.length / 2))
            center = np.array((center[1], center[0])) * self.ppm  # Default horizontal orientation
            radius = int(shape.radius * self.ppm)
            
            # Calculate top-left corner of the image for overlay
            top_left = (center - radius).astype(int)
            diameter = 2 * radius
            bottom_right = top_left + diameter
            
            if circle_type == 'puck':
                if self.current_puck_shape != shape:
                    self.current_puck_shape = shape
                    radius = int(shape.radius * self.ppm)
                    diameter = int(radius * 2)
                    self.puck_img = cv2.resize(self.puck_img, (diameter, diameter))
                resized_img = self.puck_img
            elif circle_type == 'paddle':
                if self.current_paddle_shape != shape:
                    self.current_paddle_shape = shape
                    radius = int(shape.radius * self.ppm)
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
            mask = resized_img[y_start:y_end, x_start:x_end, 3] > 0
            self.frame[frame_top_left[1] : frame_bottom_right[1], frame_top_left[0]: frame_bottom_right[0]][mask] = resized_img[y_start:y_end, x_start:x_end, :3][mask]
            # for i in range(resized_img.shape[0]):
            #     for j in range(resized_img.shape[1]):
            #         if top_left[1]+i >= self.frame.shape[0] or top_left[0]+j >= self.frame.shape[1]:
            #             continue  # Skip pixels outside the frame
            #         alpha = resized_img[i, j, 3] / 255.0  # Assuming the alpha channel is the last
            #         if alpha > 0:  # If pixel is not transparent
            #             self.frame[top_left[1]+i, top_left[0]+j, :3] = (1 - alpha) * self.frame[top_left[1]+i, top_left[0]+j, :3] + alpha * resized_img[i, j, :3]

    def draw_square_with_image(self, body_attrs, square_type='block'):
        """
        Draws a circle with an image overlay on the frame.

        Args:
            body_attrs (tuple): A tuple containing the body and color attributes of the circle.
            circle_type (str, optional): The type of circle. Defaults to 'puck'.
        """
        body, color = body_attrs  # color is unused but kept for compatibility
        for fixture in body.fixtures:
            shape = fixture.shape
            # center = np.array(body.position) + np.array((self.width / 2, self.length / 2))
            center = np.array(body.position) + np.array((self.width / 2, self.length / 2))
            center = np.array((center[1], center[0])) * self.ppm  # Default horizontal orientation
            vertices = shape.vertices
            # such as [([-width / 2, -height / 2]), ([width / 2, -height / 2]), ([width / 2, height / 2]), ([-width / 2, height / 2])]
            width = np.abs(-2 * vertices[0][0])
            width = int(width * self.ppm)

            # Calculate top-left corner of the image for overlay
            top_left = (center - width / 2).astype(int)
            bottom_right = top_left + width

            if square_type == 'block':
                if self.current_puck_shape != shape:
                    self.current_puck_shape = shape
                    vertices = shape.vertices
                    # such as [([-width / 2, -height / 2]), ([width / 2, -height / 2]), ([width / 2, height / 2]), ([-width / 2, height / 2])]
                    width = np.abs(-2 * vertices[0][0])
                    width = int(width * self.ppm)
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

            # Overlay the image
            mask = resized_img[y_start:y_end, x_start:x_end, 3] > 0
            self.frame[frame_top_left[1]: frame_bottom_right[1], frame_top_left[0]: frame_bottom_right[0]][mask] = \
            resized_img[y_start:y_end, x_start:x_end, :3][mask]
            # for i in range(resized_img.shape[0]):
            #     for j in range(resized_img.shape[1]):
            #         if top_left[1]+i >= self.frame.shape[0] or top_left[0]+j >= self.frame.shape[1]:
            #             continue  # Skip pixels outside the frame
            #         alpha = resized_img[i, j, 3] / 255.0  # Assuming the alpha channel is the last
            #         if alpha > 0:  # If pixel is not transparent
            #             self.frame[top_left[1]+i, top_left[0]+j, :3] = (1 - alpha) * self.frame[top_left[1]+i, top_left[0]+j, :3] + alpha * resized_img[i, j, :3]

    def draw_polygon(self, body_attrs):
        """
        Draws a polygon on the frame.

        Args:
            body_attrs (tuple): A tuple containing the body and color attributes of the polygon.
        """
        body, color = body_attrs
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
            def draw_goal(goal, radius, color=(0, 255, 0)):
                goal_position = goal
                goal_radius = radius
                goal_position = np.array((goal_position[1], -goal_position[0])) # coords -> box2d
                center = np.array(goal_position) + np.array((self.width / 2, self.length / 2))
                center = np.array((center[1], center[0])) * self.ppm
                goal_radius = int(goal_radius * self.ppm)
                cv2.circle(self.frame, center.astype(int), goal_radius, color, 2)
                
            green = (0, 255, 0)
            draw_goal(self.airhockey_env.ego_goal_pos, self.airhockey_env.ego_goal_radius, color=green)
            if self.airhockey_env.multiagent:
                blue = (255, 0, 0)
                draw_goal(self.airhockey_env.alt_goal_pos, self.airhockey_env.alt_goal_radius, color=blue)
        
        for puck_attrs in self.airhockey_sim.pucks.values():
            self.draw_circle_with_image(puck_attrs, circle_type='puck')
        for block_attrs in self.airhockey_sim.blocks.values():
            self.draw_square_with_image(block_attrs, square_type='block')
        for obstacle_attrs in self.airhockey_sim.obstacles.values():
            self.draw_polygon(obstacle_attrs)
        for paddle_attrs in self.airhockey_sim.paddles.values():
            self.draw_circle_with_image(paddle_attrs, circle_type='paddle')
            
        # if self.airhockey_sim.paddle[1] is not None: self.draw_circle_with_image(self.airhockey_sim.paddle[1], circle_type='paddle')
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

