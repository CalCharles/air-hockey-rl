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

    def __init__(self, airhockey_env, orientation='vertical', robosuite_view="",
                 show_target_position=True, show_acceleration_arrow=False):
        """
        Initializes the AirHockeyRenderer object.

        Args:
            airhockey_env (AirHockeySimulator): The air hockey simulator object.
            orientation (str, optional): The orientation of the game table. Defaults to 'vertical'.
            robosuite_view: which view to take from robosuite. options: empty (none), sideview, topview
            show_target_position (bool): Whether to draw target position marker. Defaults to True.
            show_acceleration_arrow (bool): Whether to draw acceleration arrows. Defaults to False.
        """
        self.airhockey_env = airhockey_env
        self.orientation = orientation
        self.show_target_position = show_target_position
        self.show_acceleration_arrow = show_acceleration_arrow
        # Adjust dimensions based on the specified orientation
        self.render_width = self.airhockey_env.render_width
        self.render_length = self.airhockey_env.render_length
        self.render_masks = self.airhockey_env.render_masks
        self.width = self.airhockey_env.width
        self.length = self.airhockey_env.length
        self.screen_width, self.screen_height = 120, 120
        self.ppm = self.airhockey_env.ppm 
        self.robosuite_view = robosuite_view
        
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
        
        # Track previous paddle velocities for acceleration calculation
        self.prev_paddle_velocities = {}
        
        # Store current action for target position visualization
        self.current_action = None
        
    def convert_to_render_coords_sys(self, pos):
        return np.array((pos[1], -pos[0]))  # coords -> box2d

    def _base_to_prerotate_pixel(self, x, y):
        """
        Base-frame world (x, y) → pixel coords on ``self.frame`` **before**
        ``cv2.rotate(..., ROTATE_90_COUNTERCLOCKWISE)``. Matches
        ``draw_circle_with_image`` / ``draw_square_with_image`` (not
        ``world_xy_to_output_pixel``, which is for the buffer **after** rotate).
        """
        render_xy = self.convert_to_render_coords_sys((float(x), float(y)))
        center = np.asarray(render_xy, dtype=float) + np.array(
            (self.width / 2, self.length / 2), dtype=float
        )
        pre_rot = np.array((center[1], center[0]), dtype=float) * float(self.ppm)
        return float(pre_rot[0]), float(pre_rot[1])

    def world_xy_to_output_pixel(self, x, y):
        """
        Map world/base-frame (x, y) to pixel coordinates in get_frame() output.
        """
        render_xy = self.convert_to_render_coords_sys((x, y))
        center = render_xy + np.array((self.width / 2, self.length / 2), dtype=float)
        pre_rot = np.array((center[1], center[0]), dtype=float) * float(self.ppm)
        x_px, y_px = float(pre_rot[0]), float(pre_rot[1])
        if self.orientation == 'vertical':
            pre_width = float(self.render_length)
            # Match cv2.rotate(..., ROTATE_90_COUNTERCLOCKWISE) in get_frame().
            return y_px, pre_width - 1.0 - x_px
        return x_px, y_px
    
    def set_action(self, action):
        """
        Set the current action for visualization purposes.
        
        Args:
            action: Action vector (typically 2D delta position for paddle)
        """
        self.current_action = action
    
    def draw_target_position(self, current_pos, action, color=(255, 165, 0)):
        """
        Draws the target position (current position + action delta).
        If the target goes outside the screen, it's clamped to the edge.
        
        Args:
            current_pos: Current position in base coordinates (x, y)
            action: Action delta in base coordinates (dx, dy)
            color: Color of the target marker (BGR format), default is orange
        """
        if action is None or len(action) < 2:
            return
            
        # Calculate target position in base coordinates
        target_pos = np.array(current_pos) + np.array(action[:2])
        self.draw_target_marker(target_pos, color=color)

    def draw_target_marker(self, target_pos, color=(255, 165, 0)):
        """
        Draw an explicit target position in base coordinates on the **current** ``self.frame``.

        Pixel mapping uses ``world_xy_to_output_pixel``, which matches the final layout after
        ``cv2.rotate(..., ROTATE_90_COUNTERCLOCKWISE)`` when ``orientation == 'vertical'``.
        Call this **after** that rotation in ``get_frame()`` so the overlay lines up with the
        paddle (which uses the same mapping implicitly via pre-rotate drawing + rotate).
        """
        if target_pos is None or len(target_pos) < 2:
            return

        clamped_target = np.array(target_pos[:2], dtype=float)
        clamped_target[0] = np.clip(clamped_target[0], -self.length / 2, self.length / 2)
        clamped_target[1] = np.clip(clamped_target[1], -self.width / 2, self.width / 2)

        cx, cy = self.world_xy_to_output_pixel(
            float(clamped_target[0]), float(clamped_target[1])
        )
        center_int = (int(round(cx)), int(round(cy)))
        
        # Draw target marker as a cross with circle
        marker_size = 15
        thickness = 3
        
        # Draw outer circle (black outline)
        cv2.circle(self.frame, center_int, marker_size, (0, 0, 0), thickness + 2)
        # Draw inner circle (colored)
        cv2.circle(self.frame, center_int, marker_size, color, thickness)
        
        # Draw cross lines (black outline)
        cv2.line(self.frame, 
                (center_int[0] - marker_size, center_int[1]), 
                (center_int[0] + marker_size, center_int[1]), 
                (0, 0, 0), thickness + 2)
        cv2.line(self.frame, 
                (center_int[0], center_int[1] - marker_size), 
                (center_int[0], center_int[1] + marker_size), 
                (0, 0, 0), thickness + 2)
        
        # Draw cross lines (colored)
        cv2.line(self.frame, 
                (center_int[0] - marker_size, center_int[1]), 
                (center_int[0] + marker_size, center_int[1]), 
                color, thickness)
        cv2.line(self.frame, 
                (center_int[0], center_int[1] - marker_size), 
                (center_int[0], center_int[1] + marker_size), 
                color, thickness)
    
    def draw_acceleration_arrow(self, pos, acceleration, max_arrow_length=1000, color=(0, 140, 255)):
        """
        Args:
            pos: Position of the paddle in render coordinates
            acceleration: Acceleration vector (ax, ay) in base coordinates
                Base coords: X is vertical (down positive), Y is horizontal
            max_arrow_length: Maximum length of the arrow in pixels
            color: Color of the arrow (BGR format)
        """
        # Convert acceleration to render coordinates if needed
        if hasattr(pos, '__len__') and len(pos) == 2:
            # pos is already in render coordinates from the caller
            center = np.array(pos) + np.array((self.width / 2, self.length / 2))
            center = np.array((center[1], center[0])) * self.ppm  # Default horizontal orientation
        else:
            return
            
        # Convert acceleration vector from base coords to render coords
        acc_render = self.convert_to_render_coords_sys(acceleration)
        acc_render = np.array([acc_render[1], acc_render[0]]) # use the same transformation, without the translation or the scaling
        
        # Calculate arrow length based on acceleration magnitude
        acc_magnitude = np.linalg.norm(acc_render)
        if acc_magnitude < 1e-6:  # Very small acceleration, don't draw arrow
            return
            
        # Scale arrow length with better scaling
        arrow_scale = 200.0  # Slightly smaller for better proportions
        arrow_length = min(acc_magnitude * arrow_scale, max_arrow_length)
        
        # Minimum arrow length for visibility
        if arrow_length < 15:
            arrow_length = 15
        
        # Normalize acceleration direction
        acc_direction = acc_render / acc_magnitude
        
        # Calculate arrow end point
        arrow_end = center + acc_direction * arrow_length
        
        # Draw arrow with outline for better visibility
        outline_color = (0, 0, 0)  # Black outline
        
        # Draw outline (thicker, black)
        cv2.arrowedLine(self.frame, 
                       tuple(center.astype(int)), 
                       tuple(arrow_end.astype(int)), 
                       outline_color, 
                       thickness=5, 
                       tipLength=0.25)
        
        # Draw main arrow (thinner, colored)
        cv2.arrowedLine(self.frame, 
                       tuple(center.astype(int)), 
                       tuple(arrow_end.astype(int)), 
                       color, 
                       thickness=3, 
                       tipLength=0.25)
        
        # Add acceleration magnitude text with background for readability
        if acc_magnitude > 0.5:  # Only show text for significant acceleration
            text = f"{acc_magnitude:.1f}"
            
            # Position text to the side of the arrow to avoid overlap
            perpendicular = np.array([-acc_direction[1], acc_direction[0]])  # Perpendicular vector
            text_offset = perpendicular * 20  # Offset to the side
            text_pos = tuple((arrow_end + text_offset).astype(int))
            
            # Get text size for background rectangle
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            font_thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            
            # Draw background rectangle for text
            bg_top_left = (text_pos[0] - 2, text_pos[1] - text_height - 2)
            bg_bottom_right = (text_pos[0] + text_width + 2, text_pos[1] + baseline + 2)
            cv2.rectangle(self.frame, bg_top_left, bg_bottom_right, (255, 255, 255), -1)
            cv2.rectangle(self.frame, bg_top_left, bg_bottom_right, (0, 0, 0), 1)
            
            # Draw text
            cv2.putText(self.frame, text, text_pos, font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
            
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
            # print("Circle (puck) is out of bounds. Not rendering...")
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
            cv2.ellipse(self.frame, center.astype(int), (goal_radius[1], goal_radius[0]), 0, 0, 360, color, 2)
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
    
    def merge_robosuite_frame(self, frame):
        print((self.airhockey_env.current_state.keys()))
        current_img = self.airhockey_env.current_state[self.robosuite_view]
        # flip upside down
        current_img = cv2.flip(current_img, 0)
        # concatenate with frame
        current_img = cv2.resize(current_img, (frame.shape[1], frame.shape[0]))
        current_img = np.concatenate([frame, current_img], axis=1)
        return current_img

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
            goal_shape = getattr(self.airhockey_env, "goal_draw_shape", "circle")
            self.draw_region(
                self.airhockey_env.goal_pos,
                self.airhockey_env.goal_radius,
                color=green,
                shape=goal_shape,
            )
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
        if 'obstacles' in state_info:
            gray = (90, 90, 90)
            for obs in state_info['obstacles']:
                verts = obs.get('vertices') or []
                if len(verts) < 3:
                    continue
                pts = []
                for vx, vy in verts:
                    px, py = self._base_to_prerotate_pixel(float(vx), float(vy))
                    pts.append([int(round(px)), int(round(py))])
                cv2.fillPoly(self.frame, [np.array(pts, dtype=np.int32)], color=gray)
        pending_target_positions = []
        if 'paddles' in state_info:
            for paddle_name in state_info['paddles']:
                pos = state_info['paddles'][paddle_name]['position']
                velocity = state_info['paddles'][paddle_name]['velocity']
                
                # Calculate acceleration from velocity change
                if paddle_name in self.prev_paddle_velocities:
                    prev_velocity = self.prev_paddle_velocities[paddle_name]
                    acceleration = np.array(velocity) - np.array(prev_velocity)
                else:
                    acceleration = np.array([0.0, 0.0])  # No previous velocity, assume zero acceleration
                
                # Update previous velocity for next frame
                self.prev_paddle_velocities[paddle_name] = velocity
                
                # self.check_position(pos, self.length, self.width)
                pos_render = self.convert_to_render_coords_sys(pos)
                radius = self.airhockey_env.paddle_radius
                self.draw_circle_with_image(pos_render, radius, circle_type='paddle')
                
                # Draw acceleration arrow on top of the paddle (if enabled)
                if self.show_acceleration_arrow:
                    self.draw_acceleration_arrow(pos_render, acceleration)
                
                # Defer target marker: must use world_xy_to_output_pixel on the final frame
                # (after vertical rotate), otherwise it misaligns with the paddle in vertical mode.
                if self.show_target_position:
                    sim_target = None
                    if hasattr(self.airhockey_env, "simulator"):
                        sim_target = getattr(self.airhockey_env.simulator, "last_target_position", None)
                    if sim_target is not None:
                        pending_target_positions.append(np.array(sim_target, dtype=float))
                    elif hasattr(self.airhockey_env, "last_action"):
                        pending_target_positions.append(
                            np.array(pos, dtype=float) + np.array(self.airhockey_env.last_action[:2], dtype=float)
                        )
            
        # if self.airhockey_env.paddle[1] is not None: self.draw_circle_with_image(self.airhockey_env.paddle[1], circle_type='paddle')
        if self.orientation == 'vertical':
            self.frame = cv2.rotate(self.frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        for tp in pending_target_positions:
            self.draw_target_marker(tp)
        if len(self.robosuite_view) > 0: self.frame = self.merge_robosuite_frame(self.frame)
        return self.frame

    def render(self):
        """
        Renders the air hockey game.
        """
        frame = self.get_frame()
        cv2.imshow('Air Hockey 2D', frame)
        cv2.waitKey(20)

