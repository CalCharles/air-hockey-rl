"""Matplotlib-based paddle trajectory visualizer (headless)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


class TrajectoryVisualizer:
    """Render paddle trajectory diagnostics as RGBA numpy arrays.

    Table coordinate convention (base coords):
        x ∈ [-length/2, +length/2]  — long axis (ego=positive, opponent=negative)
        y ∈ [-width/2,  +width/2]   — short axis

    In all plots the horizontal axis is y and the vertical axis is x so that
    the rink is taller than it is wide (portrait orientation).

    Args:
        table_length: Full table length in metres (default 1.9304).
        table_width:  Full table width in metres (default 0.8636).
        grid_resolution: Number of bins per axis for heatmaps (default 64).
        alpha: Line transparency for trajectory overlays (default 0.3).
        max_traj_lines: Maximum number of trajectory lines per category (default 50).
    """

    def __init__(
        self,
        table_length: float = 1.9304,
        table_width: float = 0.8636,
        grid_resolution: int = 64,
        alpha: float = 0.3,
        max_traj_lines: int = 50,
    ):
        self.table_length = table_length
        self.table_width = table_width
        self.grid_resolution = grid_resolution
        self.alpha = alpha
        self.max_traj_lines = max_traj_lines

        self.x_min = -table_length / 2
        self.x_max = table_length / 2
        self.y_min = -table_width / 2
        self.y_max = table_width / 2

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clip_traj(self, traj: np.ndarray) -> np.ndarray:
        """Clamp trajectory positions to table bounds."""
        out = traj.copy()
        out[:, 0] = np.clip(out[:, 0], self.x_min, self.x_max)
        out[:, 1] = np.clip(out[:, 1], self.y_min, self.y_max)
        return out

    @staticmethod
    def _fig_to_rgba(fig) -> np.ndarray:
        """Convert a matplotlib Figure to an RGBA (H, W, 4) uint8 array."""
        fig.canvas.draw()
        buf = fig.canvas.tostring_argb()
        w, h = fig.canvas.get_width_height()
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        # ARGB -> RGBA: roll channel axis so A moves from index 0 to index 3
        rgba = np.roll(arr, shift=-1, axis=2)
        return rgba

    def _add_table_rect(self, ax):
        """Overlay a white rectangle representing the table boundary."""
        rect = mpatches.Rectangle(
            (self.y_min, self.x_min),
            self.table_width,
            self.table_length,
            linewidth=1.5,
            edgecolor="white",
            facecolor="none",
        )
        ax.add_patch(rect)

    def _setup_axes(self, ax, title: str):
        """Apply common axis configuration (labels, limits, title)."""
        ax.set_xlim(self.y_min, self.y_max)
        ax.set_ylim(self.x_min, self.x_max)
        ax.set_xlabel("y (short axis)")
        ax.set_ylabel("x (long axis)")
        ax.set_title(title)

    def _build_heatmap(self, trajs: list[np.ndarray]):
        """Return a normalised 2-D histogram (grid x grid) or zeros if empty."""
        if not trajs:
            return np.zeros((self.grid_resolution, self.grid_resolution), dtype=np.float32)

        positions = np.concatenate([self._clip_traj(t) for t in trajs], axis=0)
        # np.histogram2d bins: first axis = x, second axis = y
        hist, _, _ = np.histogram2d(
            positions[:, 0],  # x values  -> vertical axis
            positions[:, 1],  # y values  -> horizontal axis
            bins=self.grid_resolution,
            range=[[self.x_min, self.x_max], [self.y_min, self.y_max]],
        )
        max_count = hist.max()
        if max_count > 0:
            hist = hist / max_count
        return hist.astype(np.float32)

    def _render_single_heatmap(self, ax, hist: np.ndarray, title: str):
        """Draw a pre-computed heatmap onto *ax*."""
        # imshow with extent maps array to axis coordinates.
        # hist[i, j] = count at x_bin i, y_bin j.
        # With origin='lower', row 0 is at x_min, row N is at x_max,
        # col 0 is at y_min, col N is at y_max.
        ax.imshow(
            hist,
            origin="lower",
            extent=[self.y_min, self.y_max, self.x_min, self.x_max],
            aspect="auto",
            cmap="hot",
            vmin=0.0,
            vmax=1.0,
        )
        self._add_table_rect(ax)
        self._setup_axes(ax, title)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_heatmap(self, policy_trajs: list[np.ndarray]) -> np.ndarray | None:
        """Render a density heatmap for *policy_trajs*.

        Returns:
            RGBA (H, W, 4) uint8 array, or None if *policy_trajs* is empty.
        """
        if not policy_trajs:
            return None

        hist = self._build_heatmap(policy_trajs)
        fig, ax = plt.subplots(figsize=(6, 8))
        self._render_single_heatmap(ax, hist, "Policy paddle density")
        rgba = self._fig_to_rgba(fig)
        plt.close(fig)
        return rgba

    def render_overlay(
        self,
        policy_trajs: list[np.ndarray],
        expert_trajs: list[np.ndarray],
    ) -> np.ndarray | None:
        """Render policy (blue) and expert (orange) trajectories overlaid.

        Returns:
            RGBA (H, W, 4) uint8 array, or None if both inputs are empty.
        """
        if not policy_trajs and not expert_trajs:
            return None

        fig, ax = plt.subplots(figsize=(6, 8))

        for traj in policy_trajs[: self.max_traj_lines]:
            t = self._clip_traj(traj)
            ax.plot(t[:, 1], t[:, 0], color="blue", alpha=self.alpha, linewidth=0.8)

        for traj in expert_trajs[: self.max_traj_lines]:
            t = self._clip_traj(traj)
            ax.plot(t[:, 1], t[:, 0], color="orange", alpha=self.alpha, linewidth=0.8)

        policy_patch = mpatches.Patch(color="blue", label="policy")
        expert_patch = mpatches.Patch(color="orange", label="expert")
        ax.legend(handles=[policy_patch, expert_patch], loc="upper right")

        self._setup_axes(ax, "Trajectory overlay")
        self._add_table_rect(ax)

        rgba = self._fig_to_rgba(fig)
        plt.close(fig)
        return rgba

    def render_sidebyside(
        self,
        policy_trajs: list[np.ndarray],
        expert_trajs: list[np.ndarray],
    ) -> np.ndarray | None:
        """Render policy and expert heatmaps side-by-side with shared normalisation.

        Returns:
            RGBA (H, W, 4) uint8 array, or None if both inputs are empty.
        """
        if not policy_trajs and not expert_trajs:
            return None

        policy_hist = self._build_heatmap(policy_trajs)
        expert_hist = self._build_heatmap(expert_trajs)

        # Shared normalisation: re-normalise both by the joint maximum.
        joint_max = max(policy_hist.max(), expert_hist.max())
        if joint_max > 0:
            policy_hist = policy_hist / joint_max
            expert_hist = expert_hist / joint_max

        fig, (ax_policy, ax_expert) = plt.subplots(1, 2, figsize=(12, 8))

        for ax, hist, title in [
            (ax_policy, policy_hist, "Policy paddle density"),
            (ax_expert, expert_hist, "Expert paddle density"),
        ]:
            ax.imshow(
                hist,
                origin="lower",
                extent=[self.y_min, self.y_max, self.x_min, self.x_max],
                aspect="auto",
                cmap="hot",
                vmin=0.0,
                vmax=1.0,
            )
            self._add_table_rect(ax)
            self._setup_axes(ax, title)

        fig.tight_layout()
        rgba = self._fig_to_rgba(fig)
        plt.close(fig)
        return rgba

    def render_all(
        self,
        policy_trajs: list[np.ndarray],
        expert_trajs: list[np.ndarray],
    ) -> dict:
        """Render all three visualisations.

        Returns:
            dict with keys "traj/heatmap", "traj/overlay", "traj/sidebyside".
            Values are RGBA arrays or None if the render was skipped.
        """
        return {
            "traj/heatmap": self.render_heatmap(policy_trajs),
            "traj/overlay": self.render_overlay(policy_trajs, expert_trajs),
            "traj/sidebyside": self.render_sidebyside(policy_trajs, expert_trajs),
        }
