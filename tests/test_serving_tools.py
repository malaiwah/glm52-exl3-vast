#!/usr/bin/env python3
"""Unit tests for the paid-runtime qualification harnesses."""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import benchmark_serving as bench  # noqa: E402
import needle_matrix  # noqa: E402
import verify_serving as verify  # noqa: E402


class BenchmarkTests(unittest.TestCase):
    def test_parse_metrics_sums_labelled_series(self):
        metrics = bench.parse_metrics("""
# HELP vllm:prompt_tokens_total Total prompt tokens.
vllm:prompt_tokens_total{model_name="a"} 12
vllm:prompt_tokens_total{model_name="b"} 8
vllm:num_preemptions_total 2
not_a_number NaN
""")
        self.assertEqual(metrics["vllm:prompt_tokens_total"], 20)
        self.assertEqual(metrics["vllm:num_preemptions_total"], 2)

    def test_speculative_summary_uses_official_mal_formula(self):
        before = {
            "vllm:spec_decode_num_drafts": 10,
            "vllm:spec_decode_num_draft_tokens": 30,
            "vllm:spec_decode_num_accepted_tokens": 18,
        }
        after = {
            "vllm:spec_decode_num_drafts": 20,
            "vllm:spec_decode_num_draft_tokens": 60,
            "vllm:spec_decode_num_accepted_tokens": 42,
        }
        result = bench.spec_summary(before, after)
        self.assertEqual(result["drafts"], 10)
        self.assertEqual(result["accepted_tokens"], 24)
        self.assertEqual(result["mean_acceptance_length"], 3.4)
        self.assertEqual(result["draft_token_acceptance_rate"], 0.8)

    def test_percentile_interpolates(self):
        self.assertEqual(bench.percentile([1, 2, 3, 4], 50), 2.5)
        self.assertAlmostEqual(bench.percentile([10, 20], 95), 19.5)

    def test_summary_reports_failures_throughput_and_preemptions(self):
        results = [
            {"ok": True, "prompt_tokens": 100, "output_tokens": 20,
             "ttft_ms": 40, "tpot_ms": 5, "mean_itl_ms": 6},
            {"ok": False, "error": "timeout"},
        ]
        before = {"vllm:num_preemptions_total": 2}
        after = {"vllm:num_preemptions_total": 5}
        summary = bench.summarize_requests(results, 2, before, after, 2)
        self.assertEqual(summary["completed"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["prompt_throughput_tok_s"], 50)
        self.assertEqual(summary["output_throughput_tok_s"], 10)
        self.assertEqual(summary["preemptions"], 3)
        self.assertEqual(summary["errors"], ["timeout"])

    def test_result_write_is_atomic_and_supports_bare_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            old = os.getcwd()
            try:
                os.chdir(directory)
                bench.write_result("result.json", {"ok": True})
                with open("result.json") as handle:
                    self.assertEqual(json.load(handle), {"ok": True})
                self.assertFalse(os.path.exists("result.json.tmp"))
            finally:
                os.chdir(old)

    def test_matrix_persists_completed_levels_before_a_later_crash(self):
        row = {"failed": 0}
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "partial.json")
            calls = [
                row,
                ConnectionResetError("engine died"),
            ]
            with mock.patch.object(bench, "run_level", side_effect=calls), \
                 mock.patch.object(bench, "discover_model", return_value="model"):
                with self.assertRaises(ConnectionResetError):
                    bench.main([
                        "--base-url", "http://test",
                        "--prefill-tokens", "1024,8192",
                        "--concurrency", "",
                        "--warmup", "0",
                        "--out", output,
                    ])
            with open(output) as handle:
                partial = json.load(handle)
            self.assertFalse(partial["complete"])
            self.assertFalse(partial["ok"])
            self.assertEqual(partial["prefill"][0]["target_prompt_tokens"], 1024)
            self.assertIn("ConnectionResetError", partial["fatal_error"])


class NeedleTests(unittest.TestCase):
    def test_haystack_is_seeded_unique_and_contains_every_needle(self):
        text_a, needles_a = verify.build_haystack(8192, [0.01, 0.5, 0.99], 7)
        text_b, needles_b = verify.build_haystack(8192, [0.01, 0.5, 0.99], 8)
        self.assertIn("trial 7", text_a.splitlines()[0])
        self.assertNotEqual(text_a, text_b)
        self.assertNotEqual(needles_a, needles_b)
        for city, code in needles_a:
            self.assertIn(f"access code for {city} is {code}", text_a)

    def test_degenerate_detector_catches_known_failure_shapes(self):
        self.assertIn("no word", verify.degenerate("...,,,!!!"))
        self.assertIn("repeats", verify.degenerate("one two three " * 10))
        self.assertEqual(verify.degenerate("Kyoto: ABC-1234"), "")

    def test_probe_records_seed_duration_and_retrieval(self):
        def fake_count(_base, _key, _model, text):
            return len(text.split()), True

        def fake_complete(_base, _key, _model, prompt, **_kwargs):
            codes = []
            for line in prompt.splitlines():
                if line.startswith("IMPORTANT:"):
                    codes.append(line.rsplit(" ", 1)[-1].rstrip("."))
            return " ".join(codes)

        with mock.patch.object(verify, "count_tokens", side_effect=fake_count), \
             mock.patch.object(verify, "complete", side_effect=fake_complete):
            result = verify.needle_probe(
                "http://test", "", "model", 8192, [0.1, 0.9], seed=99)
        self.assertTrue(result["ok"])
        self.assertEqual(result["seed"], 99)
        self.assertEqual(result["found"], 2)
        self.assertGreaterEqual(result["duration_s"], 0)

    def test_probe_calibrates_to_within_one_percent_of_requested_tokens(self):
        def fake_count(_base, _key, _model, text):
            return (text.count("\n") + 1) * 16, True

        def fake_complete(_base, _key, _model, prompt, **_kwargs):
            return " ".join(
                line.rsplit(" ", 1)[-1].rstrip(".")
                for line in prompt.splitlines()
                if line.startswith("IMPORTANT:")
            )

        with mock.patch.object(verify, "count_tokens", side_effect=fake_count), \
             mock.patch.object(verify, "complete", side_effect=fake_complete):
            result = verify.needle_probe(
                "http://test", "", "model", 8192, [0.1, 0.9], seed=100)
        self.assertLessEqual(abs(result["tokens"] - 8192) / 8192, 0.01)
        self.assertTrue(result["ok"])

    def test_matrix_caps_deduplicates_and_sorts_sizes(self):
        self.assertEqual(
            needle_matrix.capped_sizes(
                [490000, 32768, 600000, 32768], 524288, 4096),
            [32768, 490000, 520192])


if __name__ == "__main__":
    unittest.main(verbosity=2)
