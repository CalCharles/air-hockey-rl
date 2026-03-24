import ast
from dataclasses import dataclass, fields
import tempfile
import textwrap
import unittest
from pathlib import Path

import yaml


def _load_mapping_fn():
    source_path = Path(__file__).resolve().parent.parent / "extras" / "async_td3_real.py"
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
        raise RuntimeError("Failed to locate Args and _build_args_file_defaults in async_td3_real.py")

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
            episode_artifact_dir: "/tmp/episodes"
            smoke_test_seconds: 15
            episode_camera_video_dir: "/tmp/camera"
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
        self.assertEqual(mapped["episode_artifact_dir"], "/tmp/episodes")
        self.assertEqual(mapped["smoke_test_seconds"], 15)
        self.assertEqual(mapped["episode_camera_video_dir"], "/tmp/camera")
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
        self.assertIn("episode_camera_video_dir", applied)
        self.assertIn("exploration_primitive_chance_start", applied)
        self.assertEqual(ignored, [])

    def test_legacy_alias_applies_when_canonical_absent(self) -> None:
        path = self._write_yaml(
            """
            learning_starts: 1234
            """
        )

        mapped, applied, ignored = _build_args_file_defaults(path)

        self.assertEqual(mapped["min_replay_size_before_learning"], 1234)
        self.assertIn("learning_starts", applied)
        self.assertEqual(ignored, [])

    def test_canonical_value_wins_and_null_alias_is_ignored(self) -> None:
        path = self._write_yaml(
            """
            agent_hidden_layer_size: 128
            agent_hidden_size: null
            q_hidden_layer_size: 256
            q_hidden_size: 64
            """
        )

        mapped, applied, ignored = _build_args_file_defaults(path)

        self.assertEqual(mapped["agent_hidden_layer_size"], 128)
        self.assertEqual(mapped["q_hidden_layer_size"], 256)
        self.assertNotIn("agent_hidden_size", applied)
        self.assertNotIn("q_hidden_size", applied)
        self.assertEqual(ignored, [])

    def test_unsupported_keys_are_reported(self) -> None:
        path = self._write_yaml(
            """
            totally_unknown_key: 42
            device: "cuda:0"
            exploration_primitive_chance_pre_learning_starts: 0.5
            exploration_primitive_weight_anneal_same_direction: 1.0
            exploration_policy_takeover_enabled: true
            exploration_pre_contact_hit_variant_chance: 0.25
            """
        )

        mapped, applied, ignored = _build_args_file_defaults(path)

        self.assertEqual(mapped["learner_device"], "cuda:0")
        self.assertIn("device", applied)
        self.assertIn("totally_unknown_key", ignored)
        self.assertIn("exploration_primitive_chance_pre_learning_starts", ignored)
        self.assertIn("exploration_primitive_weight_anneal_same_direction", ignored)
        self.assertIn("exploration_policy_takeover_enabled", ignored)
        self.assertIn("exploration_pre_contact_hit_variant_chance", ignored)


if __name__ == "__main__":
    unittest.main()
