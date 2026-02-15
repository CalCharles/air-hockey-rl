"""Rendering utilities for air hockey trajectory visualization.

Provides sprite loading, pixel coordinate conversion, frame annotation,
GIF saving, and side-by-side GIF synchronization.
"""

import os
import cv2
import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont

from data_loading import (
    BOX2D_TABLE_LENGTH,
    BOX2D_TABLE_WIDTH,
    BOX2D_PADDLE_RADIUS,
    BOX2D_PUCK_RADIUS,
)

RENDER_SIZE = 360
ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../assets"))


def load_assets(ppm):
    """Load and prepare table, paddle, and puck images."""
    table = cv2.imread(os.path.join(ASSETS_DIR, "air_hockey_table.png"))
    table = cv2.rotate(table, cv2.ROTATE_90_CLOCKWISE)
    render_w = int(RENDER_SIZE)
    render_l = int(ppm * BOX2D_TABLE_LENGTH)
    table = cv2.resize(table, (render_l, render_w))

    paddle = cv2.imread(os.path.join(ASSETS_DIR, "paddle.png"), cv2.IMREAD_UNCHANGED)
    d = 2 * int(BOX2D_PADDLE_RADIUS * ppm)
    paddle = cv2.resize(paddle, (d, d))

    puck = cv2.imread(os.path.join(ASSETS_DIR, "puck.png"), cv2.IMREAD_UNCHANGED)
    d = 2 * int(BOX2D_PUCK_RADIUS * ppm)
    puck = cv2.resize(puck, (d, d))

    return table, paddle, puck


def overlay_sprite(frame, sprite, center_px):
    """Alpha-blend a BGRA sprite onto a BGR frame at pixel center."""
    h, w = sprite.shape[:2]
    tl_x = int(center_px[0] - w // 2)
    tl_y = int(center_px[1] - h // 2)

    src_x0 = max(0, -tl_x)
    src_y0 = max(0, -tl_y)
    dst_x0 = max(0, tl_x)
    dst_y0 = max(0, tl_y)
    dst_x1 = min(frame.shape[1], tl_x + w)
    dst_y1 = min(frame.shape[0], tl_y + h)
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)

    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return

    region = sprite[src_y0:src_y1, src_x0:src_x1]
    mask = region[:, :, 3] > 0
    frame[dst_y0:dst_y1, dst_x0:dst_x1][mask] = region[:, :, :3][mask]


def pos_to_pixel(pos_base, ppm):
    """Convert base coords (x, y) -> pixel coords on the horizontal frame."""
    rx, ry = pos_base[1], -pos_base[0]
    center = np.array([rx + BOX2D_TABLE_WIDTH / 2, ry + BOX2D_TABLE_LENGTH / 2])
    pixel = np.array([center[1], center[0]]) * ppm
    return pixel


def annotate_loss(frame, step_loss, cumulative_loss):
    """Overlay per-step and cumulative loss text on a frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    h = frame.shape[0]
    cv2.putText(frame, f"loss: {step_loss:.4f}", (10, h - 30),
                font, 0.45, (0, 0, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"cum: {cumulative_loss:.4f}", (10, h - 10),
                font, 0.45, (0, 0, 200), 1, cv2.LINE_AA)


def save_gif(frames, gif_path, fps=20):
    """Save a list of RGB frames as a GIF."""
    if os.path.isdir(gif_path) or not gif_path.endswith(".gif"):
        os.makedirs(gif_path, exist_ok=True)
        gif_path = os.path.join(gif_path, "rollout.gif")
    else:
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    duration = int(1000 / fps)
    imageio.mimsave(gif_path, frames, format="GIF", loop=0, duration=duration)
    print(f"Saved GIF ({len(frames)} frames, {fps} fps): {gif_path}")
    return gif_path


def produce_synchronized_gifs(path_to_source_gif, path_to_target_gif,
                              path_for_synchronization,
                              title_left="Source Env", title_right="Target Env",
                              fps=20):
    """Create a side-by-side GIF from two existing GIFs."""
    gif1 = [np.array(f) for f in imageio.mimread(path_to_source_gif)]
    gif2 = [np.array(f) for f in imageio.mimread(path_to_target_gif)]

    font_size = 26
    title_bar_height = 60
    background_color = (0, 0, 0)
    text_color = "White"

    try:
        font = ImageFont.truetype("arialbd.ttf", font_size)
    except OSError:
        try:
            font = ImageFont.truetype("Arial Bold.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

    combined_frames = []
    for f1, f2 in zip(gif1, gif2):
        combined = np.hstack((f1, f2))
        gif_img = Image.fromarray(combined)
        W, H = gif_img.size
        half_W = W // 2

        new_img = Image.new("RGB", (W, H + title_bar_height), background_color)
        new_img.paste(gif_img, (0, 0))
        draw = ImageDraw.Draw(new_img)

        bbox_left = draw.textbbox((0, 0), title_left, font=font)
        tw_left = bbox_left[2] - bbox_left[0]
        th_left = bbox_left[3] - bbox_left[1]

        bbox_right = draw.textbbox((0, 0), title_right, font=font)
        tw_right = bbox_right[2] - bbox_right[0]

        y_pos = H + (title_bar_height - th_left) // 2
        draw.text(((half_W - tw_left) // 2, y_pos),
                  title_left, fill=text_color, font=font)
        draw.text((half_W + (half_W - tw_right) // 2, y_pos),
                  title_right, fill=text_color, font=font)

        combined_frames.append(np.array(new_img))

    imageio.mimsave(path_for_synchronization, combined_frames, loop=0, fps=fps)
    print(f"Produced synchronized GIF: {path_for_synchronization}")
