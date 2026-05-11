import ast
from dataclasses import dataclass, fields
import tempfile
import textwrap
import unittest
from pathlib import Path

import yaml


def _load_mapping_fn():
    source_path = Path(__file__).resolve().parent.parent / "helper" / "real_td3_runtime.py"
    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    args_node = None
    mapping_node = None
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "Args":
            args_node = node
        elif isinstance(node, ast.FunctionDef) and node.name == "_build_args_file_defaults":
            mapping_node = node
    if args_node is None or mapping_node is None:
        raise RuntimeError("Failed to locate Args and _build_args_file_defaults in real_td3_runtime.py")

    future_annotations = ast.ImportFrom(
        module="__future__",
        names=[ast.alias(name="annotations", asname=None)],
        level=0,
    )
    subset_module = ast.Module(body=[future_annotations, args_node, mapping_node], type_ignores=[])
    ast.fix_missing_locations(subset_module)
    namespace = {
        "dataclass": dataclass,
        "fields": fields,
        "yaml": yaml,
    }
    exec(compile(subset_module, str(source_path), "exec"), namespace)
    return namespace["_build_args_file_defaults"]


_build_args_file_defaults = _load_mapping_fn()


class AsyncTd3ArgsFileMappingTests(unittest.TestCase):
    def _write_yaml(self, body: str) -> str:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        path = Path(tmp_dir.name) / "args.yaml"
        path.write_text(textwrap.dedent(body), encoding="utf-8")
        return str(path)

    def test_direct_args_keys_are_applied(self) -> None:
        path = self._write_yaml(
            """
            collector_device: "cuda:1"
            updates_per_second: 3.5
            actor_sync_check_every_episode: false
            collector_log_interval_sec: 1.25
            learner_log_interval_sec: 2.5
            data_root_dir: "/tmp/data_root"
            smoke_test_seconds: 15
            exploration_primitive_chance_start: 0.4
            exploration_primitive_chance: 0.1
            exploration_primitive_steps: 7
            exploration_primitive_weight_stand_still: 0.25
            exploration_primitive_weight_same_direction: 1.5
            exploration_primitive_weight_y_aligned: 0.75
            exploration_primitive_weight_target_position_directional: 2.0
            exploration_target_position_steps: 6
            enable_latency_profiling: true
            latency_profile_output_dir: "/tmp/latency"
            latency_profile_hist_bins: 50
            """
        )

        mapped, applied, ignored = _build_args_file_defaults(path)

        self.assertEqual(mapped["collector_device"], "cuda:1")
        self.assertEqual(mapped["updates_per_second"], 3.5)
        self.assertFalse(mapped["actor_sync_check_every_episode"])
        self.assertEqual(mapped["collector_log_interval_sec"], 1.25)
        self.assertEqual(mapped["learner_log_interval_sec"], 2.5)
        self.assertEqual(mapped["data_root_dir"], "/tmp/data_root")
        self.assertEqual(mapped["smoke_test_seconds"], 15)
        self.assertEqual(mapped["exploration_primitive_chance_start"], 0.4)
        self.assertEqual(mapped["exploration_primitive_chance"], 0.1)
        self.assertEqual(mapped["exploration_primitive_steps"], 7)
        self.assertEqual(mapped["exploration_primitive_weight_stand_still"], 0.25)
        self.assertEqual(mapped["exploration_primitive_weight_same_direction"], 1.5)
        self.assertEqual(mapped["exploration_primitive_weight_y_aligned"], 0.75)
        self.assertEqual(mapped["exploration_primitive_weight_target_position_directional"], 2.0)
        self.assertEqual(mapped["exploration_target_position_steps"], 6)
        self.assertTrue(mapped["enable_latency_profiling"])
        self.assertEqual(mapped["latency_profile_output_dir"], "/tmp/latency")
        self.assertEqual(mapped["latency_profile_hist_bins"], 50)
        self.assertIn("collector_device", applied)
        self.assertIn("data_root_dir", applied)
        self.assertIn("exploration_primitive_chance_start", applied)
        self.assertEqual(ignored, [])

    def test_legacy_per_artifact_dir_keys_are_ignored(self) -> None:
        # The split per-artifact dirs were collapsed into a single `data_root_dir`
        # field. Older configs that still set them must be silently ignored
        # rather than crash, but should be reported as ignored keys.
        path = self._write_yaml(
            """
            episode_artifact_dir: "/tmp/episodes"
            reset_artifact_dir: "/tmp/resets"
            episode_gif_dir: "/tmp/gifs"
            episode_camera_video_dir: "/tmp/camera"
            """
        )

        mapped, applied, ignored = _build_args_file_defaults(path)

        self.assertEqual(mapped, {})
        self.assertEqual(applied, [])
        for legacy_key in (
            "episode_artifact_dir",
            "reset_artifact_dir",
            "episode_gif_dir",
            "episode_camera_video_dir",
        ):
            self.assertIn(legacy_key, ignored)

    def test_deprecated_aliases_are_ignored_not_remapped(self) -> None:
        # Legacy alias fields from older async_td3_real versions (and td3_training.py's
        # deprecated `*_hidden_size` / `learning_starts` / `device` names) must NOT be
        # auto-remapped to their canonical equivalents. They land in `ignored` and the
        # user is expected to rename them in their args.yaml.
        path = self._write_yaml(
            """
            learning_starts: 1234
            device: "cuda:0"
            agent_hidden_size: 128
            q_hidden_size: 256
            """
        )

        mapped, applied, ignored = _build_args_file_defaults(path)

        self.assertNotIn("min_replay_size_before_learning", mapped)
        self.assertNotIn("learner_device", mapped)
        self.assertNotIn("agent_hidden_layer_size", mapped)
        self.assertNotIn("q_hidden_layer_size", mapped)
        self.assertEqual(applied, [])
        self.assertIn("learning_starts", ignored)
        self.assertIn("device", ignored)
        self.assertIn("agent_hidden_size", ignored)
        self.assertIn("q_hidden_size", ignored)

    def test_canonical_online_keys_are_applied(self) -> None:
        # Online behavior fields still live on Args, so they flow from --args-file.
        path = self._write_yaml(
            """
            learner_device: "cuda:0"
            min_replay_size_before_learning: 1234
            """
        )

        mapped, applied, ignored = _build_args_file_defaults(path)

        self.assertEqual(mapped["learner_device"], "cuda:0")
        self.assertEqual(mapped["min_replay_size_before_learning"], 1234)
        self.assertEqual(ignored, [])

    def test_architecture_keys_in_args_file_are_ignored(self) -> None:
        # Architecture fields now live on TrainArgs (sourced from --train-args);
        # if they appear in the --args-file YAML they must be ignored so they
        # cannot accidentally override the TrainArgs values.
        path = self._write_yaml(
            """
            agent_hidden_layer_size: 128
            agent_num_hidden_layers: 3
            q_hidden_layer_size: 256
            q_num_hidden_layers: 3
            action_scale: 0.5
            use_last_action_in_policy_state: true
            """
        )

        mapped, applied, ignored = _build_args_file_defaults(path)

        self.assertEqual(mapped, {})
        self.assertEqual(applied, [])
        for field_name in (
            "agent_hidden_layer_size",
            "agent_num_hidden_layers",
            "q_hidden_layer_size",
            "q_num_hidden_layers",
            "action_scale",
            "use_last_action_in_policy_state",
        ):
            self.assertIn(field_name, ignored)

    def test_unsupported_keys_are_reported(self) -> None:
        path = self._write_yaml(
            """
            totally_unknown_key: 42
            exploration_primitive_chance_pre_learning_starts: 0.5
            exploration_primitive_weight_anneal_same_direction: 1.0
            exploration_policy_takeover_enabled: true
            exploration_pre_contact_hit_variant_chance: 0.25
            """
        )

        mapped, applied, ignored = _build_args_file_defaults(path)

        self.assertIn("totally_unknown_key", ignored)
        self.assertIn("exploration_primitive_chance_pre_learning_starts", ignored)
        self.assertIn("exploration_primitive_weight_anneal_same_direction", ignored)
        self.assertIn("exploration_policy_takeover_enabled", ignored)
        self.assertIn("exploration_pre_contact_hit_variant_chance", ignored)


if __name__ == "__main__":
    unittest.main()
