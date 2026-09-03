"""Episode-GIF recording for training rollouts.

Owns the `watch/` and `samples/` directories under a run's log_parent_dir:
- `watch/` keeps the last N recorded episodes on a FIFO ring (overwritten as
  new recordings arrive — useful for "tail the current behavior").
- `samples/` persists one recorded episode every `sample_gif_interval`
  env-steps, capped at `sample_gif_max_storage_mb` MB total (FIFO eviction).

Reusable from any training script that has a renderer with `.get_frame()`
returning a BGR numpy array (e.g. `AirHockeyRenderer`).
"""

import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import imageio


class GIFEpisodeRecorder:
    def __init__(
        self,
        parent_dir: str,
        *,
        watch_ring_size: int = 10,
        watch_episode_interval: int = 50,
        sample_gif_interval: int = 10000,
        sample_gif_max_storage_mb: float = 50.0,
        frame_width: int = 160,
        frame_duration_ms: int = 50,
    ) -> None:
        self.watch_dir = os.path.join(parent_dir, "watch")
        self.samples_dir = os.path.join(parent_dir, "samples")
        os.makedirs(self.watch_dir, exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)

        self.watch_ring_size = int(watch_ring_size)
        self.watch_episode_interval = int(watch_episode_interval)
        self.sample_gif_interval = int(sample_gif_interval)
        self.sample_gif_max_storage_mb = float(sample_gif_max_storage_mb)
        self.frame_width = int(frame_width)
        self.frame_duration_ms = int(frame_duration_ms)

        self._frames: list = []
        self._recording = False
        self._last_reward = 0.0
        self._cumulative_reward = 0.0
        self._completed_episodes = 0
        self._watch_ring_idx = 0
        self._last_sample_gif_step = 0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gif-encode")
        self._encode_future = None

    def close(self) -> None:
        """Flush any pending GIF encode. Safe to call more than once."""
        if self._encode_future is not None:
            try:
                self._encode_future.result()
            except Exception as exc:  # pragma: no cover - best effort
                print(f"GIF encode failed: {exc}")
            self._encode_future = None
        self._executor.shutdown(wait=True)

    @property
    def recording(self) -> bool:
        return self._recording

    def capture_frame(self, renderer, global_step: int) -> None:
        """Render + overlay reward stats + append. No-op when not recording."""
        if not self._recording:
            return
        frame = renderer.get_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        aspect_ratio = frame.shape[1] / frame.shape[0]
        frame = cv2.resize(frame, (self.frame_width, int(self.frame_width / aspect_ratio)))
        cv2.putText(
            frame, f"R: {self._last_reward:.2f}",
            (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2,
        )
        cv2.putText(
            frame, f"G: {self._cumulative_reward:.2f}",
            (frame.shape[1] - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2,
        )
        cv2.putText(
            frame, f"Step: {global_step}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1,
        )
        self._frames.append(frame)

    def note_reward(self, reward: float) -> None:
        """Track last + cumulative reward for the overlay. No-op when not recording."""
        if not self._recording:
            return
        self._last_reward = float(reward)
        self._cumulative_reward += self._last_reward

    def _encode(self, frames: list, paths: list, evict: bool) -> None:
        """GIF palette quantisation costs ~7 ms/frame in Pillow; run it off the
        training thread (Pillow releases the GIL in its C encoder)."""
        for path in paths:
            imageio.mimsave(
                path, frames, format="GIF",
                loop=0, duration=self.frame_duration_ms,
            )
        if evict:
            self._evict_old_samples()

    def on_episode_end(self, global_step: int) -> None:
        """Call once per terminated/truncated episode.

        If we were recording and have frames, saves the GIF into the watch
        ring and optionally also into samples/. Then bumps the completed-
        episode counter and decides whether to start recording the *next*
        episode based on watch_episode_interval.
        """
        if self._recording and self._frames:
            paths = [os.path.join(self.watch_dir, f"ep_{self._watch_ring_idx}.gif")]
            self._watch_ring_idx = (self._watch_ring_idx + 1) % self.watch_ring_size

            evict = False
            if global_step - self._last_sample_gif_step >= self.sample_gif_interval:
                paths.append(os.path.join(self.samples_dir, f"step_{global_step}.gif"))
                self._last_sample_gif_step = global_step
                evict = True

            frames = self._frames
            self._frames = []
            # Serialise encodes (one worker) so eviction / ring order stays deterministic.
            if self._encode_future is not None:
                self._encode_future.result()
            self._encode_future = self._executor.submit(self._encode, frames, paths, evict)
            self._recording = False
            self._last_reward = 0.0
            self._cumulative_reward = 0.0

        self._completed_episodes += 1
        if self._completed_episodes % self.watch_episode_interval == 0:
            self._recording = True

    def _evict_old_samples(self) -> None:
        files = sorted(
            [
                os.path.join(self.samples_dir, f)
                for f in os.listdir(self.samples_dir)
                if f.endswith(".gif")
            ],
            key=os.path.getmtime,
        )
        total = sum(os.path.getsize(f) for f in files)
        max_bytes = self.sample_gif_max_storage_mb * 1024 * 1024
        while total > max_bytes and files:
            oldest = files.pop(0)
            total -= os.path.getsize(oldest)
            os.remove(oldest)
