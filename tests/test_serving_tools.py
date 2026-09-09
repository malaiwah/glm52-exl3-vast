#!/usr/bin/env python3
"""Unit tests for the paid-runtime qualification harnesses."""
import hashlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path
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

    def test_metric_timestamp_and_spaced_labels_do_not_change_sample_value(self):
        metrics = bench.parse_metrics(
            'vllm:prompt_tokens_total{model_name="a b"} 12 1788742800000\n'
            'vllm:prompt_tokens_total{model_name="c"} 8 1788742800000\n')
        self.assertEqual(metrics["vllm:prompt_tokens_total"], 20)

    def test_overflowing_metric_sum_fails_instead_of_emitting_infinity(self):
        with self.assertRaises(ValueError):
            bench.parse_metrics('counter{rank="0"} 1e308\ncounter{rank="1"} 1e308')

    def test_nonfinite_result_does_not_replace_previous_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "result.json")
            bench.write_result(path, {"ok": False})
            with self.assertRaises(ValueError):
                bench.write_result(path, {"gauge": float("nan")})
            self.assertEqual(json.loads(Path(path).read_text()), {"ok": False})

    def test_sse_chunks_are_not_used_as_token_counts(self):
        chunk = {"choices": [{"delta": {"content": "one two three"}}]}
        for usage in ({}, {"prompt_tokens": 100, "completion_tokens": 5}):
            with self.subTest(usage=usage):
                stream = io.BytesIO((
                    "data: " + json.dumps(chunk) + "\n"
                    + "data: " + json.dumps({"usage": usage}) + "\n"
                    + "data: [DONE]\n").encode())
                with mock.patch.object(bench, "request", return_value=stream):
                    result = bench.stream_completion(
                        "http://test", "", "model", "prompt", 90, 5)
                self.assertEqual(result["ok"], bool(usage))
                if usage:
                    self.assertEqual(result["output_tokens"], 5)
                    self.assertEqual(result["prompt_tokens"], 100)
                    self.assertEqual(result["text_chunks"], 1)
                else:
                    self.assertNotIn("output_tokens", result)

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
        self.assertNotEqual(text_a, text_b)
        self.assertNotEqual(needles_a, needles_b)
        for city, code in needles_a:
            self.assertIn(f"access code for {city} is {code}", text_a)


    def test_structured_probe_accepts_exact_schema(self):
        response = {
            "choices": [{
                "message": {
                    "content": json.dumps({"answer": 42}),
                    "reasoning_content": "brief reasoning",
                },
            }],
        }
        with mock.patch.object(verify, "_req", return_value=response):
            result = verify.structured_output_probe(
                "http://test", "key", "model"
            )
        self.assertTrue(result["ok"])
        self.assertTrue(result["reasoning_field"])

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


    def test_stochastic_probe_requires_full_output_budget(self):
        response = {
            "choices": [{
                "finish_reason": "length",
                "message": {"content": "coherent output"},
            }],
            "usage": {"completion_tokens": 511},
        }

        def fake_req(url, payload=None, **_kwargs):
            return "healthy" if url.endswith("/health") else response

        with mock.patch.object(
                verify, "count_tokens", return_value=(1024, True)), \
             mock.patch.object(verify, "_req", side_effect=fake_req):
            result = verify.stochastic_sampling_probe(
                "http://test", "", "model")
        self.assertFalse(result["ok"])

    def test_main_fails_before_long_probe_when_stochastic_sampling_fails(self):
        failure = {
            "attempted": True,
            "ok": False,
            "detail": "sampling request failed: timeout",
        }
        with tempfile.TemporaryDirectory() as raw:
            out = os.path.join(raw, "verdict.json")
            with mock.patch.object(
                    verify, "wait_health", return_value=(True, "healthy")), \
                 mock.patch.object(
                    verify, "short_probe",
                    return_value={"ok": True, "checks": []}), \
                 mock.patch.object(
                    verify, "structured_output_probe",
                    return_value={"ok": True, "detail": "passed"}), \
                 mock.patch.object(
                    verify, "stochastic_sampling_probe",
                    return_value=failure), \
                 mock.patch.object(verify, "needle_probe") as needle:
                rc = verify.main([
                    "--base-url", "http://test",
                    "--model", "model",
                    "--out", out,
                ])
            verdict = json.loads(Path(out).read_text())
        self.assertEqual(rc, 1)
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["stochastic_sampling"], failure)
        needle.assert_not_called()

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

    def test_interrupted_matrix_never_publishes_partial_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "needles.json")
            with mock.patch.object(verify, "needle_probe", side_effect=[
                    {"ok": True, "tokens": 8192, "tokens_exact": True},
                    KeyboardInterrupt()]):
                with self.assertRaises(KeyboardInterrupt):
                    needle_matrix.main([
                        "--model", "model", "--sizes", "8192,16384", "--out", path])
            report = json.loads(Path(path).read_text())
            self.assertFalse(report["complete"])
            self.assertFalse(report["ok"])
            self.assertTrue(report["probes"][0]["ok"])

    def test_matrix_gates_exact_length_and_post_stress_and_emits_one_document(self):
        for exact, tokens, post_ok, expected in (
                (True, 8192, True, True), (False, 8192, True, False),
                (True, 4096, True, False), (True, 8192, False, False)):
            with self.subTest(exact=exact, tokens=tokens, post_ok=post_ok):
                output = io.StringIO()
                with mock.patch.object(verify, "needle_probe", return_value={
                        "ok": True, "tokens": tokens, "tokens_exact": exact}), \
                     mock.patch.object(verify, "short_probe", return_value={"ok": True}), \
                     mock.patch.object(verify, "stochastic_sampling_probe",
                                       return_value={"ok": post_ok}), \
                     mock.patch("sys.stdout", output):
                    status = needle_matrix.main(["--model", "model", "--sizes", "8192"])
                report = json.loads(output.getvalue())
                self.assertTrue(report["complete"])
                self.assertEqual(report["ok"], expected)
                self.assertEqual(status, 0 if expected else 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
