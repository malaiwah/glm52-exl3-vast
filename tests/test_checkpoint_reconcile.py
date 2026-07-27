#!/usr/bin/env python3
"""Regression tests for checkpoint/config reconciliation."""
import json
import os
import struct
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import reconcile_checkpoint as reconcile  # noqa: E402


def write_json(path, doc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(doc, handle)


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def read_bytes(path):
    with open(path, "rb") as handle:
        return handle.read()


def read_text(path):
    with open(path) as handle:
        return handle.read()


def write_safetensors(path, names):
    header = {name: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}
              for name in names}
    encoded = json.dumps(header).encode()
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        handle.write(b"\x00\x00")


def text_config(moe_end, ignored=True):
    return {
        "architectures": ["GlmForCausalLM"],
        "hybrid_tr3_tail": {"moe_layers": [3, moe_end]},
        "quantization_config": {
            "ignore": (["model.layers.78.mlp.experts.*"] if ignored else [])
        },
    }


class ReconcileTests(unittest.TestCase):
    def run_main(self, model_dir, vision):
        state = os.path.join(model_dir, ".state")
        runtime = os.path.join(model_dir, ".runtime")
        with mock.patch.dict(
                os.environ,
                {"GLM_STATE_DIR": state, "GLM_RUNTIME_DIR": runtime},
                clear=False):
            return reconcile.main([model_dir, "--vision", str(int(vision)), "--quiet"])

    def test_observation_distinguishes_bf16_and_trellis(self):
        with tempfile.TemporaryDirectory() as directory:
            shard = os.path.join(directory, reconcile.LSHARD)
            write_safetensors(
                shard, ["model.layers.78.mlp.experts.0.w2_weight"])
            self.assertEqual(reconcile.observe_layer78(directory), "bf16")
            write_safetensors(
                shard, ["model.layers.78.mlp.experts.w2_weight"])
            self.assertEqual(reconcile.observe_layer78(directory), "trellis")

    def test_bf16_text_only_repairs_stale_graft_and_index(self):
        with tempfile.TemporaryDirectory() as directory:
            # Live config incorrectly describes a grafted layer and a vision
            # wrapper. The on-disk tensors are BF16 and vision is requested off.
            live_text = text_config(78, ignored=False)
            write_json(os.path.join(directory, reconcile.CFG), {
                "architectures": ["Glm5vForConditionalGeneration"],
                "text_config": live_text,
            })
            write_json(
                os.path.join(directory, reconcile.CFG + ".orig"),
                text_config(77, ignored=True))
            write_safetensors(
                os.path.join(directory, reconcile.LSHARD),
                ["model.layers.78.mlp.experts.0.w2_weight"])
            write_safetensors(
                os.path.join(directory, "model-layer-001.safetensors"),
                ["model.layers.1.weight"])
            write_safetensors(
                os.path.join(directory, "vision_tower.safetensors"),
                ["vision.weight"])
            write_safetensors(
                os.path.join(directory, "mm_projector.safetensors"),
                ["projector.weight"])
            write_json(os.path.join(directory, reconcile.IDX), {"weight_map": {
                "vision.weight": "vision_tower.safetensors"}})
            open(os.path.join(directory, ".mtp78-grafted"), "w").close()

            self.assertEqual(self.run_main(directory, vision=False), 0)
            result = read_json(os.path.join(directory, reconcile.CFG))
            self.assertNotIn("text_config", result)
            self.assertEqual(result["hybrid_tr3_tail"]["moe_layers"], [3, 77])
            self.assertIn(
                "model.layers.78.mlp.experts.*",
                result["quantization_config"]["ignore"])
            index = read_json(os.path.join(directory, reconcile.IDX))
            self.assertNotIn("vision.weight", index["weight_map"])
            self.assertFalse(os.path.exists(
                os.path.join(directory, ".mtp78-grafted")))

    def test_trellis_with_vision_rebuilds_wrapper_index_and_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            write_json(
                os.path.join(directory, reconcile.CFG),
                text_config(77, ignored=True))
            write_json(
                os.path.join(directory, reconcile.CFG + ".orig"),
                text_config(77, ignored=True))
            write_safetensors(
                os.path.join(directory, reconcile.LSHARD),
                ["model.layers.78.mlp.experts.w2_weight"])
            write_safetensors(
                os.path.join(directory, "vision_tower.safetensors"),
                ["vision.weight"])
            write_safetensors(
                os.path.join(directory, "mm_projector.safetensors"),
                ["projector.weight"])
            write_json(os.path.join(directory, reconcile.IDX), {"weight_map": {}})
            vision_dir = os.path.join(directory, ".vision")
            write_json(os.path.join(vision_dir, reconcile.CFG), {
                "architectures": ["Glm5vForConditionalGeneration"],
                "vision_config": {"hidden_size": 1},
            })
            with open(os.path.join(directory, "chat_template.jinja"), "w") as handle:
                handle.write("text")
            with open(os.path.join(vision_dir, "chat_template.jinja"), "w") as handle:
                handle.write("vision")

            self.assertEqual(self.run_main(directory, vision=True), 0)
            result = read_json(os.path.join(directory, reconcile.CFG))
            self.assertEqual(
                result["architectures"], ["Glm5vForConditionalGeneration"])
            self.assertEqual(
                result["text_config"]["hybrid_tr3_tail"]["moe_layers"], [3, 78])
            self.assertFalse(any(
                "layers.78" in item
                for item in result["text_config"]["quantization_config"]["ignore"]))
            index = read_json(os.path.join(directory, reconcile.IDX))
            self.assertEqual(
                index["weight_map"]["vision.weight"], "vision_tower.safetensors")
            self.assertEqual(
                read_text(os.path.join(directory, "chat_template.jinja")),
                "vision")
            self.assertTrue(os.path.exists(
                os.path.join(directory, ".mtp78-grafted")))
            self.assertTrue(os.path.exists(
                os.path.join(directory, ".vision-enabled")))

    def test_dry_run_changes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = text_config(78, ignored=False)
            write_json(os.path.join(directory, reconcile.CFG), cfg)
            write_json(
                os.path.join(directory, reconcile.CFG + ".orig"),
                text_config(77, ignored=True))
            write_safetensors(
                os.path.join(directory, reconcile.LSHARD),
                ["model.layers.78.mlp.experts.0.w2_weight"])
            before = read_bytes(os.path.join(directory, reconcile.CFG))
            state = os.path.join(directory, ".state")
            with mock.patch.dict(os.environ, {"GLM_STATE_DIR": state}, clear=False):
                self.assertEqual(reconcile.main(
                    [directory, "--vision", "0", "--dry-run", "--quiet"]), 0)
            self.assertEqual(
                read_bytes(os.path.join(directory, reconcile.CFG)), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
