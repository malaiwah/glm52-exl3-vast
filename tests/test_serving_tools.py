#!/usr/bin/env python3
"""Unit tests for the paid-runtime qualification harnesses."""
import hashlib
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
import evaluate_scorecard as scorecard  # noqa: E402
import needle_matrix  # noqa: E402
import offload_prefix_benchmark as offload  # noqa: E402
import verify_serving as verify  # noqa: E402


class DownloadProgressTests(unittest.TestCase):
    def test_fresh_download_reports_zero_then_crossed_5pct(self):
        expected = 10 * download_progress.GIB
        self.assertEqual(
            download_progress.crossed_milestones(None, 0, expected), [0])
        self.assertEqual(
            download_progress.crossed_milestones(0, 3.4 * download_progress.GIB,
                                                 expected),
            [5, 10, 15, 20, 25, 30],
        )

    def test_resumed_download_reports_only_current_5pct(self):
        expected = 20 * download_progress.GIB
        self.assertEqual(
            download_progress.crossed_milestones(
                None, 17 * download_progress.GIB, expected),
            [85],
        )

    def test_in_progress_report_never_claims_completion(self):
        expected = 21 * download_progress.GIB
        self.assertEqual(
            download_progress.crossed_milestones(
                85, 30 * download_progress.GIB, expected),
            [90, 95],
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
vllm:prompt_tokens_total{model_name="c"} NaN
vllm:num_preemptions_total 2
not_a_number NaN
""")
        # A NaN sample on a real series must not poison its sum, and a purely
        # NaN-valued metric must be absent rather than recorded as NaN.
        self.assertEqual(metrics["vllm:prompt_tokens_total"], 20)
        self.assertEqual(metrics["vllm:num_preemptions_total"], 2)
        self.assertNotIn("not_a_number", metrics)

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
             "ttft_ms": 40, "tpot_ms": 5, "mean_inter_chunk_ms": 6},
            {"ok": False, "error": "timeout"},
        ]
        before = {
            "vllm:num_preemptions_total": 2,
            "vllm:prefix_cache_queries_total": 100,
            "vllm:prefix_cache_hits_total": 10,
            "vllm:external_prefix_cache_queries_total": 40,
            "vllm:external_prefix_cache_hits_total": 4,
        }
        after = {
            "vllm:num_preemptions_total": 5,
            "vllm:prefix_cache_queries_total": 250,
            "vllm:prefix_cache_hits_total": 210,
            "vllm:external_prefix_cache_queries_total": 60,
            "vllm:external_prefix_cache_hits_total": 24,
        }
        summary = bench.summarize_requests(results, 2, before, after, 2)
        self.assertEqual(summary["completed"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["prompt_throughput_tok_s"], 50)
        self.assertEqual(summary["output_throughput_tok_s"], 10)
        self.assertEqual(summary["preemptions"], 3)
        self.assertEqual(summary["errors"], ["timeout"])
        # Prefix-cache deltas expose a prefix-cache-warm matched-seed rerun.
        self.assertEqual(summary["prefix_cache_query_tokens"], 150)
        self.assertEqual(summary["gpu_prefix_cache_hit_tokens"], 200)
        self.assertEqual(summary["external_prefix_cache_query_tokens"], 20)
        self.assertEqual(summary["external_prefix_cache_hit_tokens"], 20)

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
            "ttft_ms": 1, "tpot_ms": 0, "mean_inter_chunk_ms": 0,
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
            "ttft_ms": 1, "tpot_ms": 0, "mean_inter_chunk_ms": 0,
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


class ScorecardTests(unittest.TestCase):
    def test_gsm8k_prefers_last_strict_answer(self):
        text = "The answer is 12. Correction: The answer is 1,234."
        self.assertEqual(scorecard.extract_answer("gsm8k_cot", text), "1234")

    def test_gpqa_requires_explicit_answer_marker(self):
        self.assertIsNone(
            scorecard.extract_answer("gpqa_diamond", "I considered A and B.")
        )
        self.assertEqual(
            scorecard.extract_answer("gpqa_diamond", "Answer: $c"), "C"
        )

    def test_scorecard_defaults_leave_room_for_reasoning(self):
        self.assertEqual(scorecard.default_max_tokens("gsm8k_cot"), 4096)
        self.assertEqual(scorecard.default_max_tokens("gpqa_diamond"), 32768)

    def test_hidden_reasoning_is_not_a_visible_answer(self):
        content = ""
        reasoning = "After checking the work, The answer is 42."
        self.assertIsNone(scorecard.extract_answer("gsm8k_cot", content))
        self.assertEqual(
            scorecard.extract_answer("gsm8k_cot", reasoning), "42"
        )

    def test_dataset_content_digest_is_recorded(self):
        raw = '{"question":"q","answer":"#### 1"}\n'
        with mock.patch.object(scorecard, "fetch_text", return_value=raw):
            rows, digest = scorecard.load_gsm8k()
        self.assertEqual(rows[0]["question"], "q")
        self.assertEqual(digest, hashlib.sha256(raw.encode()).hexdigest())


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

    def test_lmcache_external_hit_does_not_require_native_byte_metrics(self):
        # An external hit with no native byte counter (LMCache served from
        # persistent NVMe L2) is a genuine external hit but NOT a confirmed
        # DRAM hit: dram_hit_observed must stay False without the DRAM-specific
        # native signal, so it actually distinguishes DRAM from generic external.
        evidence = offload.connector_hit_evidence({
            "external_prefix_hit_tokens": 131072,
            "kv_offload_load_bytes": 0,
        })
        self.assertTrue(evidence["external_hit_observed"])
        self.assertFalse(evidence["native_load_bytes_observed"])
        self.assertFalse(evidence["dram_hit_observed"])

    def test_native_hit_retains_both_evidence_signals(self):
        evidence = offload.connector_hit_evidence({
            "external_prefix_hit_tokens": 131072,
            "kv_offload_load_bytes": 2_000_000_000,
        })
        self.assertTrue(evidence["external_hit_observed"])
        self.assertTrue(evidence["native_load_bytes_observed"])
        self.assertTrue(evidence["dram_hit_observed"])

    def test_restart_l2_hit_is_valid_even_when_reload_stage_is_cold(self):
        evidence = offload.connector_hit_summary(
            {"external_prefix_hit_tokens": 131072},
            {"external_prefix_hit_tokens": 0},
        )
        self.assertTrue(evidence["initial_external_hit_observed"])
        self.assertFalse(evidence["external_hit_observed"])
        self.assertTrue(evidence["any_external_hit_observed"])


class NeedleTests(unittest.TestCase):
    def test_verifier_discovers_the_served_model_alias(self):
        with mock.patch.object(
                verify, "_req",
                return_value={"data": [{"id": "GLM-5.2"}]}):
            self.assertEqual(
                verify.discover_model("http://test", ""), "GLM-5.2")

    def test_short_probe_enables_thinking_with_sufficient_output_budget(self):
        answers = iter(("391", "Paris", "BANANA"))

        with mock.patch.object(
                verify, "complete",
                side_effect=lambda *_args, **_kwargs: next(answers)) as complete:
            result = verify.short_probe("http://test", "", "model")

        self.assertTrue(result["ok"])
        self.assertEqual(complete.call_count, 3)
        for call in complete.call_args_list:
            self.assertEqual(call.kwargs["max_tokens"], 256)
            self.assertTrue(call.kwargs["enable_thinking"])

    def test_complete_rejects_truncated_reasoning_as_an_answer(self):
        response = {
            "choices": [{
                "finish_reason": "length",
                "message": {"content": "The answer is Paris"},
            }],
        }
        with mock.patch.object(verify, "_req", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "exhausted"):
                verify.complete(
                    "http://test", "", "model", "prompt",
                    max_tokens=32, enable_thinking=True)

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
        requested = {}

        def fake_count(_base, _key, _model, text):
            return len(text.split()), True

        def fake_complete(_base, _key, _model, prompt, **kwargs):
            requested.update(kwargs)
            codes = []
            for line in prompt.splitlines():
                if line.startswith("IMPORTANT:"):
                    codes.append(line.rsplit(" ", 1)[-1].rstrip("."))
            return " ".join(codes)

        with mock.patch.object(verify, "count_tokens", side_effect=fake_count), \
             mock.patch.object(verify, "complete", side_effect=fake_complete):
            result = verify.needle_probe(
                "http://test", "", "model", 8192, [0.1, 0.9],
                seed=99, max_tokens=1024)
        self.assertTrue(result["ok"])
        self.assertEqual(result["seed"], 99)
        self.assertEqual(result["found"], 2)
        self.assertGreaterEqual(result["duration_s"], 0)
        self.assertEqual(requested["max_tokens"], 1024)

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
