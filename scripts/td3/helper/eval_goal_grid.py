"""Deterministic 4x5 eval goal grid for goal-position tasks."""
from __future__ import annotations

import numpy as np

GOAL_GRID_TASK_NAMES: tuple[str, ...] = (
    "puck_goal_position",
    "puck_goal_position_velocity",
    "puck_goal_position_obstacles",
    "puck_goal_position_dynamic_negative_regions",
)

GOAL_GRID_ROWS: int = 4
GOAL_GRID_COLS: int = 5
GOAL_GRID_TOP_INSET_FRAC: float = 0.10
GOAL_GRID_BOTTOM_INSET_FRAC: float = 0.30
GOAL_GRID_LEFT_INSET_FRAC: float = 0.15
GOAL_GRID_RIGHT_INSET_FRAC: float = 0.30


def build_eval_goal_grid(
    *,
    table_x_bot: float,
    table_y_right: float,
) -> list[tuple[float, float]]:
    """Row-major (x outer, y inner) grid over the goal region of the table."""
    x_max_abs = float(table_x_bot)
    y_half = float(table_y_right)
    x_top = -x_max_abs * (1.0 - GOAL_GRID_TOP_INSET_FRAC)
    x_bot = -x_max_abs * GOAL_GRID_BOTTOM_INSET_FRAC
    y_left = -y_half * (1.0 - GOAL_GRID_LEFT_INSET_FRAC)
    y_right = +y_half * (1.0 - GOAL_GRID_RIGHT_INSET_FRAC)
    xs = np.linspace(x_top, x_bot, GOAL_GRID_ROWS)
    ys = np.linspace(y_left, y_right, GOAL_GRID_COLS)
    grid: list[tuple[float, float]] = []
    for x in xs:
        for y in ys:
            grid.append((float(x), float(y)))
    return grid


def build_eval_goal_grid_from_env(env) -> list[tuple[float, float]]:
    """Build the eval grid from an env's table geometry."""
    return build_eval_goal_grid(
        table_x_bot=float(getattr(env, "table_x_bot")),
        table_y_right=float(getattr(env, "table_y_right")),
    )
