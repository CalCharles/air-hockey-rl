import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

from scripts.smooth_policy.amp_history.amp_training.td3.extras.run_td3_motion_weight_staged import (
    build_stage_command,
    build_weight_schedule,
    detect_alternate_log_dir,
    evaluate_convergence,
    find_latest_stage_checkpoint,
    parse_checkpoint_step_from_path,
    terminate_process,
    validate_args,
)


class WeightScheduleTests(unittest.TestCase):
    def test_schedule_includes_start_and_final(self) -> None:
        self.assertEqual(
            build_weight_schedule(start=0.0, final=0.3, increment=0.1),
            [0.0, 0.1, 0.2, 0.3],
        )

    def test_schedule_includes_final_even_if_not_multiple(self) -> None:
        self.assertEqual(
            build_weight_schedule(start=0.0, final=0.25, increment=0.1),
            [0.0, 0.1, 0.2, 0.25],
        )

    def test_schedule_rejects_bad_increment(self) -> None:
        with self.assertRaises(ValueError):
            build_weight_schedule(start=0.0, final=0.2, increment=0.0)


class ConvergenceTests(unittest.TestCase):
    def test_convergence_requires_both_metrics(self) -> None:
        window = 50_000
        q1_task = [
            (25_000, 10.0),
            (75_000, 10.1),
            (125_000, 10.0),
        ]
        # motion has a large shift, so this should not converge.
        q1_motion = [
            (25_000, 1.0),
            (75_000, 1.0),
            (125_000, 2.0),
        ]
        status = evaluate_convergence(
            q1_task_points=q1_task,
            q1_motion_points=q1_motion,
            window_steps=window,
            threshold=0.05,
        )
        self.assertFalse(status.converged)

    def test_convergence_true_when_both_deltas_small(self) -> None:
        window = 50_000
        q1_task = [
            (25_000, 10.0),
            (75_000, 10.0),
            (125_000, 10.2),
        ]
        q1_motion = [
            (25_000, 2.0),
            (75_000, 2.0),
            (125_000, 2.05),
        ]
        status = evaluate_convergence(
            q1_task_points=q1_task,
            q1_motion_points=q1_motion,
            window_steps=window,
            threshold=0.05,
        )
        self.assertTrue(status.converged)
        self.assertIsNotNone(status.rel_delta_task)
        self.assertIsNotNone(status.rel_delta_motion)


class CheckpointSelectionTests(unittest.TestCase):
    def test_parse_checkpoint_step_from_path(self) -> None:
        path = Path("/tmp/run/stage_00_w0p0/checkpoint_345000/training_state.pth")
        self.assertEqual(parse_checkpoint_step_from_path(path), 345000)

    def test_find_latest_checkpoint_prefers_highest_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stage_dir = Path(tmp_dir)
            (stage_dir / "checkpoint_5000").mkdir()
            (stage_dir / "checkpoint_15000").mkdir()
            (stage_dir / "checkpoint_5000" / "training_state.pth").write_text("a")
            (stage_dir / "checkpoint_15000" / "training_state.pth").write_text("b")
            latest = find_latest_stage_checkpoint(stage_dir)
            self.assertEqual(latest, stage_dir / "checkpoint_15000" / "training_state.pth")

    def test_find_latest_checkpoint_falls_back_to_stage_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stage_dir = Path(tmp_dir)
            (stage_dir / "training_state.pth").write_text("root")
            latest = find_latest_stage_checkpoint(stage_dir)
            self.assertEqual(latest, stage_dir / "training_state.pth")


class TerminationTests(unittest.TestCase):
    def test_terminate_process_graceful_then_return(self) -> None:
        proc = mock.Mock()
        proc.poll.side_effect = [None, 0]
        proc.wait.return_value = 0
        terminate_process(proc, soft_timeout=0.1, term_timeout=0.1)
        proc.send_signal.assert_called_once()
        proc.terminate.assert_not_called()
        proc.kill.assert_not_called()


class StageCommandTests(unittest.TestCase):
    def test_stage_command_contains_exploration_overrides(self) -> None:
        args = mock.Mock()
        args.python_executable = "python"
        args.args_file = "/tmp/args.yaml"
        args.device = "cuda:1"
        args.stage_learning_starts = 0
        args.stage_exploration_primitive_chance_pre_learning_starts = 0.0
        args.stage_exploration_primitive_chance_start = 0.08
        args.stage_exploration_primitive_chance_anneal_steps = 50000

        command = build_stage_command(
            args=args,
            train_script_path=Path("/tmp/train.py"),
            stage_dir=Path("/tmp/stage_00"),
            run_name="stage0",
            motion_reward_weight=0.1,
            stage_start_checkpoint=Path("/tmp/training_state.pth"),
            stage_total_timesteps=123456,
        )

        joined = " ".join(command)
        self.assertIn("--learning-starts 0", joined)
        self.assertIn("--exploration-primitive-chance-pre-learning-starts 0.0", joined)
        self.assertIn("--exploration-primitive-chance-start 0.08", joined)
        self.assertIn("--exploration-primitive-chance-anneal-steps 50000", joined)


class ForceStopAliasTests(unittest.TestCase):
    def test_force_stop_after_timesteps_overrides_max_stage_steps(self) -> None:
        args = SimpleNamespace(
            force_stop_after_timesteps=12345,
            max_stage_steps=500000,
            final_motion_reward_weight=0.3,
            start_motion_reward_weight=0.0,
            motion_reward_weight_increment=0.1,
            window_steps=50000,
            convergence_threshold=0.05,
            poll_seconds=5.0,
            stage_learning_starts=0,
            stage_exploration_primitive_chance_anneal_steps=50000,
            stage_exploration_primitive_chance_pre_learning_starts=0.0,
            stage_exploration_primitive_chance_start=0.08,
        )
        validate_args(args)
        self.assertEqual(args.max_stage_steps, 12345)


class AlternateLogDirDetectionTests(unittest.TestCase):
    def test_detect_alternate_log_dir_from_launcher_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "launcher.log"
            alt_dir = Path(tmp_dir) / "stage_00_w0r1"
            log_file.write_text(
                "line1\n"
                f"Log directory exists. Saving to alternate log directory: {alt_dir}\n",
                encoding="utf-8",
            )
            detected = detect_alternate_log_dir(log_file)
            self.assertEqual(detected, alt_dir.resolve())


if __name__ == "__main__":
    unittest.main()
