"""Local GIF gallery and optional W&B logging for online FB."""
from __future__ import annotations

import html
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import imageio.v2 as imageio

from airhockey.renderers import AirHockeyRenderer


class EpisodeVideo:
    def __init__(self, env, task, args, step, episode):
        self.renderer = AirHockeyRenderer(env, orientation="vertical",
                                         show_target_position=True, show_acceleration_arrow=False)
        self.task, self.args, self.step, self.episode = task, args, step, episode
        self.frames = []

    def capture(self, reward, total):
        frame = self.renderer.get_frame()
        if self.task == "reach":
            x, y = self.renderer.world_xy_to_output_pixel(self.args.paddle_goal_x, self.args.paddle_goal_y)
            cv2.circle(frame, (round(x), round(y)),
                       max(2, round(self.args.paddle_goal_radius * self.renderer.ppm)), (0, 180, 0), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (160, round(frame.shape[0] * 160 / frame.shape[1])))
        cv2.rectangle(frame, (0, 0), (160, 37), (255, 255, 255), -1)
        cv2.putText(frame, f"{self.task} step {self.step}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX,
                    .32, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"R {reward:.2f}  G {total:.2f}", (3, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    .32, (0, 0, 0), 1, cv2.LINE_AA)
        self.frames.append(frame)

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{self.task}_ep{self.episode:02d}.gif"
        imageio.mimsave(path, self.frames, format="GIF", loop=0, duration=50)
        return path


def update_gallery(root):
    """Self-contained HTML with relative GIF links; refresh to see new evaluations."""
    root = Path(root)
    clips = sorted(root.glob("step_*/*.gif"), reverse=True)
    cards = []
    for path in clips:
        rel = path.relative_to(root).as_posix()
        cards.append(f'<figure><img loading="lazy" src="{html.escape(rel)}">'
                     f'<figcaption>{html.escape(rel)}</figcaption></figure>')
        # Stable filenames allow opening current performance directly.
    latest = root / "latest"
    latest.mkdir(exist_ok=True)
    latest_step = max(root.glob("step_*"), default=None)
    if latest_step:
        for path in latest_step.glob("*.gif"):
            shutil.copyfile(path, latest / path.name)
    page = ('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="60">'
            '<title>Online FB evaluation</title><style>body{font:16px sans-serif;margin:24px;'
            'background:#f4f4f4}main{display:flex;flex-wrap:wrap}figure{margin:12px;'
            'padding:12px;background:white}figcaption{max-width:240px}</style>'
            '<h1>Online FB evaluation</h1><p>Fixed evaluation seeds. Newest first. '
            'Updates every evaluation; this page refreshes each minute.</p><main>'
            + ''.join(cards) + '</main>')
    tmp = root / "index.tmp"
    tmp.write_text(page)
    tmp.replace(root / "index.html")


class WandbLogger:
    def __init__(self, args, sim_config, output):
        import wandb
        self.wandb = wandb
        from dataclasses import asdict
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        self.run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                              name=args.wandb_name or Path(output).name, dir=str(output),
                              mode="online", config={**asdict(args), "sim_config": sim_config,
                                                     "git_revision": revision})
        self.run.define_metric("env_step")
        self.run.define_metric("*", step_metric="env_step")
        (Path(output) / "wandb_run.json").write_text(json.dumps(
            {"id": self.run.id, "url": self.run.url, "project": self.run.project,
             "entity": self.run.entity}, indent=2))
        print(f"W&B run: {self.run.url}", flush=True)

    def log(self, values, step):
        # Custom x-axis allows multiple episode/train/eval records at one env step.
        self.run.log({"env_step": step, **values})

    def log_gifs(self, folder, step):
        videos = {f"videos/{path.stem}": self.wandb.Video(str(path), format="gif")
                  for path in sorted(Path(folder).glob("*.gif"))}
        self.log(videos, step)

    def finish(self, success):
        self.run.finish(exit_code=0 if success else 1)
