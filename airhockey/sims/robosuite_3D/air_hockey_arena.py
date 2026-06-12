"""Air hockey arena with low-friction table and side walls."""

import numpy as np

from robosuite.models.arenas import TableArena
from robosuite.utils.mjcf_utils import array_to_string, new_body, new_geom, new_site


class AirHockeyArena(TableArena):
    """
    Narrow table arena with side walls and goal-line sites.

    Extends robosuite TableArena without modifying core robosuite assets.
    """

    def __init__(
        self,
        table_full_size=(1.2, 0.6, 0.02),
        table_friction=(0.01, 0.0001, 0.00001),
        table_offset=(0, 0, 0.8),
        wall_height=0.08,
        wall_thickness=0.02,
        has_legs=False,
    ):
        super().__init__(
            table_full_size=table_full_size,
            table_friction=table_friction,
            table_offset=table_offset,
            has_legs=has_legs,
        )

        self.wall_height = wall_height
        self.wall_thickness = wall_thickness
        self.goal_line_y = table_full_size[1] / 2 - 0.05
        self.table_bounds_x = (-table_full_size[0] / 2, table_full_size[0] / 2)
        self.table_bounds_y = (-table_full_size[1] / 2, table_full_size[1] / 2)

        self._add_walls()
        self._add_goal_sites()
        self._configure_cameras()

    def _configure_cameras(self):
        """
        Reposition default cameras for the larger air-hockey table.

        The stock TableArena agentview sits close to the tabletop and crops out
        most of the robot. overview pulls back to frame robot + full table.
        """
        # Same downward tilt as robosuite frontview, further from the workspace.
        overview_quat = [0.56, 0.43, 0.43, 0.56]

        self.set_camera(
            camera_name="agentview",
            pos=[1.35, 0.0, 1.55],
            quat=overview_quat,
            camera_attribs={"fovy": "50"},
        )
        self.set_camera(
            camera_name="frontview",
            pos=[1.85, 0.0, 1.65],
            quat=overview_quat,
            camera_attribs={"fovy": "50"},
        )
        self.set_camera(
            camera_name="overview",
            pos=[2.15, 0.0, 1.9],
            quat=overview_quat,
            camera_attribs={"fovy": "58"},
        )

    def _add_walls(self):
        """Add four perimeter walls around the table top."""
        lx, ly, lz = self.table_half_size
        wall_z = lz + self.wall_height / 2
        wall_rgba = "0.2 0.2 0.2 1"

        specs = [
            ("wall_left", [-(lx + self.wall_thickness), 0, wall_z], [self.wall_thickness, ly + self.wall_thickness, self.wall_height / 2]),
            ("wall_right", [lx + self.wall_thickness, 0, wall_z], [self.wall_thickness, ly + self.wall_thickness, self.wall_height / 2]),
            ("wall_near", [0, -(ly + self.wall_thickness), wall_z], [lx + self.wall_thickness, self.wall_thickness, self.wall_height / 2]),
            ("wall_far", [0, ly + self.wall_thickness, wall_z], [lx + self.wall_thickness, self.wall_thickness, self.wall_height / 2]),
        ]

        for name, pos, size in specs:
            body = new_body(name=f"{name}_body", pos=array_to_string(pos))
            geom = new_geom(
                name=name,
                type="box",
                size=array_to_string(size),
                rgba=wall_rgba,
                friction=array_to_string(self.table_friction),
                group="0",
                conaffinity="1",
                contype="1",
            )
            body.append(geom)
            self.table_body.append(body)

    def _add_goal_sites(self):
        """Goal-line sites used for success checks and visualization."""
        lx = self.table_half_size[0] * 0.4
        goal_y = self.goal_line_y
        z = self.table_half_size[2] + 0.005

        for name, y_sign in (("goal_opponent", 1), ("goal_defender", -1)):
            site = new_site(
                name=name,
                pos=array_to_string([0, y_sign * goal_y, z]),
                size="0.02",
                rgba="0 1 0 0.3",
                type="box",
            )
            self.table_body.append(site)
