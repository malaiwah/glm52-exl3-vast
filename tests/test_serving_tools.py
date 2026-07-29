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
import download_progress  # noqa: E402
import needle_matrix  # noqa: E402
import offload_prefix_benchmark as offload  # noqa: E402
import verify_serving as verify  # noqa: E402


class DownloadProgressTests(unittest.TestCase):
    def test_fresh_download_reports_zero_then_crossed_deciles(self):
        expected = 10 * download_progress.GIB
        self.assertEqual(
            download_progress.crossed_milestones(None, 0, expected), [0])
        self.assertEqual(
            download_progress.crossed_milestones(0, 3.4 * download_progress.GIB,
                                                 expected),
            [10, 20, 30],
        )

    def test_resumed_download_reports_only_current_decile(self):
        expected = 20 * download_progress.GIB
        self.assertEqual(
            download_progress.crossed_milestones(
                None, 17 * download_progress.GIB, expected),
            [80],
        )

    def test_in_progress_report_never_claims_completion(self):
        expected = 21 * download_progress.GIB
        self.assertEqual(
            download_progress.crossed_milestones(
                80, 30 * download_progress.GIB, expected),
            [90],
        )

    def test_directory_size_tolerates_normal_files(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "shard"), "wb") as handle:
                handle.write(b"x" * 17)
            self.assertEqual(download_progress.directory_bytes(directory), 17)

    def test_milestone_is_newline_record_and_boot_note_is_timestamped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "boot.log")
            message = download_progress.format_progress(
                40, 9 * download_progress.GIB, 21)
            download_progress.append_boot_note(path, message)
            with open(path, encoding="utf-8") as handle:
                line = handle.read()
            self.assertRegex(
                line,
                r"^\[20\d\d-\d\d-\d\dT\d\d:\d\d:\d\dZ\] "
                r">>> Download progress: 40% \(9\.0 of ~21 GiB\)\n$",
            )


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

    def test_prompt_seed_makes_ab_request_text_reproducible(self):
        nonces = []

        def fake_prompt(_base, _key, _model, _tokens, nonce, _insecure):
            nonces.append(nonce)
            return f"prompt {nonce}", 32

        ok = {
            "ok": True, "prompt_tokens": 32, "output_tokens": 1,
            "ttft_ms": 1, "tpot_ms": 0, "mean_itl_ms": 0,
        }
        with mock.patch.object(bench, "make_prompt", side_effect=fake_prompt), \
             mock.patch.object(bench, "get_metrics", return_value={}), \
             mock.patch.object(bench, "stream_completion", return_value=ok):
            bench.run_level(
                "http://test", "", "model", 32, 1, 2, 3, 10,
                nonce_prefix="loader-ab-c2")
        self.assertEqual(nonces, [
            "loader-ab-c2-0", "loader-ab-c2-1", "loader-ab-c2-2"])

    def test_temperature_and_seed_reach_each_matched_request(self):
        ok = {
            "ok": True, "prompt_tokens": 32, "output_tokens": 1,
            "ttft_ms": 1, "tpot_ms": 0, "mean_itl_ms": 0,
        }
        with mock.patch.object(
                bench, "make_prompt", return_value=("prompt", 32)), \
             mock.patch.object(bench, "get_metrics", return_value={}), \
             mock.patch.object(
                 bench, "stream_completion", return_value=ok) as completion:
            bench.run_level(
                "http://test", "", "model", 32, 1, 1, 2, 10,
                nonce_prefix="block-ab", temperature=1.0, seed=1776)
        self.assertEqual(completion.call_count, 2)
        for call in completion.call_args_list:
            self.assertEqual(call.args[-2:], (1.0, 1776))


class OffloadBenchmarkTests(unittest.TestCase):
    def test_metric_summary_distinguishes_gpu_and_dram_prefix_hits(self):
        before = {
            "vllm:prefix_cache_hits_total": 10,
            "vllm:external_prefix_cache_hits_total": 20,
            "vllm:kv_offload_load_bytes_total": 100,
            "vllm:kv_offload_total_bytes": 1000,
        }
        after = {
            "vllm:prefix_cache_hits_total": 42,
            "vllm:external_prefix_cache_hits_total": 84,
            "vllm:kv_offload_load_bytes_total": 612,
            "vllm:kv_offload_total_bytes": 2000,
        }
        result = offload.metric_summary(before, after)
        self.assertEqual(result["gpu_prefix_hit_tokens"], 32)
        self.assertEqual(result["external_prefix_hit_tokens"], 64)
        self.assertEqual(result["kv_offload_load_bytes"], 512)
        self.assertEqual(result["kv_offload_total_bytes"], 2000)

    def test_metric_summary_never_reports_negative_counter_deltas(self):
        result = offload.metric_summary(
            {"vllm:kv_offload_load_bytes": 100},
            {"vllm:kv_offload_load_bytes": 1})
        self.assertEqual(result["kv_offload_load_bytes"], 0)


class NeedleTests(unittest.TestCase):
    def test_verifier_discovers_the_served_model_alias(self):
        with mock.patch.object(
                verify, "_req",
                return_value={"data": [{"id": "GLM-5.2"}]}):
            self.assertEqual(
                verify.discover_model("http://test", ""), "GLM-5.2")

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

    def test_structured_probe_requires_exact_schema_and_enables_thinking(self):
        response = {
            "choices": [{
                "message": {
                    "content": json.dumps({"answer": 42}),
                    "reasoning_content": "brief reasoning",
                },
            }],
        }
        with mock.patch.object(verify, "_req", return_value=response) as request:
            result = verify.structured_output_probe(
                "http://test", "key", "model"
            )
        self.assertTrue(result["ok"])
        self.assertTrue(result["reasoning_field"])
        payload = request.call_args.args[1]
        self.assertTrue(
            payload["chat_template_kwargs"]["enable_thinking"]
        )
        self.assertTrue(
            payload["response_format"]["json_schema"]["strict"]
        )

    def test_structured_probe_rejects_extra_schema_fields(self):
        response = {
            "choices": [{
                "message": {
                    "content": json.dumps({"answer": 42, "extra": True}),
                },
            }],
        }
        with mock.patch.object(verify, "_req", return_value=response):
            result = verify.structured_output_probe(
                "http://test", "", "model"
            )
        self.assertFalse(result["ok"])

    def test_structured_probe_retries_transient_failure(self):
        import http.client
        response = {
            "choices": [{
                "message": {
                    "content": json.dumps({"answer": 42}),
                    "reasoning_content": "brief reasoning",
                },
            }],
        }
        calls = {"n": 0}

        def fake_req(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise http.client.IncompleteRead(b"")
            return response

        with mock.patch.object(verify, "_req", side_effect=fake_req), \
             mock.patch.object(verify, "time") as mtime:
            mtime.sleep = lambda _s: None
            result = verify.structured_output_probe(
                "http://test", "key", "model"
            )
        self.assertTrue(result["ok"])
        self.assertEqual(calls["n"], 2)

    def test_structured_probe_does_not_retry_nontransient(self):
        import urllib.error
        calls = {"n": 0}

        def fake_req(*args, **kwargs):
            calls["n"] += 1
            raise urllib.error.HTTPError(
                "url", 400, "bad request", {}, None)

        with mock.patch.object(verify, "_req", side_effect=fake_req):
            result = verify.structured_output_probe(
                "http://test", "key", "model"
            )
        self.assertFalse(result["ok"])
        self.assertEqual(calls["n"], 1)

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
