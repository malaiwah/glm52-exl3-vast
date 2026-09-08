"""Maintenance qualification and rollback boundaries; no GPU or Podman daemon."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'maintenance' / 'glm53-aibeast-500k'))
import candidate
import run_qualification


class MaintenanceSafetyTests(unittest.TestCase):
    def test_rollback_revokes_boot_even_without_systemd_and_starts_production(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = candidate.SPEC['default_name']
            production = candidate.SPEC['production_container']
            (root / 'stage.json').write_text(json.dumps({
                'environment': {'NAME': name, 'MAINTENANCE_TRIAL': 'parity'},
                'production_container': production}))
            (root / 'qualification.json').write_text('{"qualified":true}')
            (root / ('container-' + name + '.service')).write_text('[Service]\n')
            actions = []

            def operation(*args, **kwargs):
                self.assertTrue((root / 'boot-disabled').exists())
                self.assertFalse((root / 'qualification.json').exists())
                actions.append(args)

            with mock.patch.dict(os.environ, {}, clear=True), \
                 mock.patch.object(candidate, 'settings', side_effect=ValueError('broken checkpoint')), \
                 mock.patch.object(candidate, 'location', return_value=(root, name)), \
                 mock.patch.object(candidate, 'exists', return_value=True), \
                 mock.patch.object(candidate, 'candidate'), \
                 mock.patch.object(candidate, 'inspect', return_value={'State': {'Running': False}}), \
                 mock.patch.object(candidate, 'podman', side_effect=operation), \
                 mock.patch.object(candidate.subprocess, 'run', side_effect=FileNotFoundError('systemctl')), \
                 mock.patch.object(sys, 'argv', ['candidate.py', 'rollback']):
                candidate.main()
            self.assertEqual(actions, [('stop', '-t', '120', name), ('start', production)])
            with mock.patch.dict(os.environ, {}, clear=True), \
                 mock.patch.object(candidate, 'location', return_value=(root, name)), \
                 mock.patch.object(sys, 'argv', ['candidate.py', 'boot-check']):
                with self.assertRaisesRegex(ValueError, 'revoked'):
                    candidate.main()

    def test_warmed_retry_or_oom_logs_cannot_certify_cold_start(self):
        self.assertTrue(run_qualification.cold_log_clean('Loading weights\nCUDA graphs captured\n'))
        self.assertFalse(run_qualification.cold_log_clean(
            'CUDA out of memory\nrestarting vLLM — attempt 1/2\nApplication startup complete'))
        self.assertFalse(run_qualification.cold_log_clean(
            'vLLM exited (attempt 0)\nApplication startup complete'))

    def test_missing_mixed_workload_receipt_is_required_not_optional(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = {'SERVED_MODEL_NAME': 'candidate', 'PORT': '8001', 'MAX_MODEL_LEN': '520192'}
            (root / 'start.json').write_text(json.dumps({'stage_sha256': 'stage', 'container_id': 'container'}))
            for filename, doc in {
                'needles': {'complete': True, 'ok': True, 'model': 'candidate', 'base_url': 'http://127.0.0.1:8001',
                            'max_model_len': 520192, 'prompt_identity_policy': 'fresh-uuid',
                            'probes': [{'ok': True, 'tokens_exact': True, 'tokens': 516096}],
                            'post_stress': {'short_prompt': {'ok': True}, 'stochastic_sampling': {'ok': True}}},
                'features': {'complete': True, 'ok': True, 'model': 'candidate', 'base_url': 'http://127.0.0.1:8001'},
                'offload': {'complete': True, 'ok': True, 'requests_ok': True, 'model': 'candidate', 'base_url': 'http://127.0.0.1:8001'},
            }.items():
                (root / (filename + '.json')).write_text(json.dumps(doc))
            (root / 'proof').write_text('simulated evidence for gate rejection only')
            proof = {'path': 'proof', 'sha256': candidate.digest(root / 'proof')}
            for kind in ['cache', 'load']:
                checks = {key: {'ok': True, 'artifact': proof} for key in candidate.SPEC['required_' + kind + '_checks']
                          if key != 'long_prefill_decode_overlap'}
                (root / (kind + '.json')).write_text(json.dumps({'stage_sha256': 'stage', 'complete': True, 'checks': checks}))
            evidence = {'stage_sha256': 'stage', 'container_id': 'container',
                        'diagnostics': {key: proof for key in ['container', 'gpu_initial', 'gpu_final']},
                        'reports': {key: {'path': key + '.json', 'sha256': candidate.digest(root / (key + '.json'))}
                                    for key in ['needles', 'features', 'offload', 'cache', 'load']}}
            path = root / 'evidence.json'
            path.write_text(json.dumps(evidence))
            with mock.patch.object(candidate, 'staged', return_value='stage'), \
                 mock.patch.object(candidate, 'verify_model'):
                with self.assertRaisesRegex(ValueError, 'long_prefill_decode_overlap'):
                    candidate.qualify(root, env, path)
            self.assertFalse((root / 'qualification.json').exists())


class MaintenanceTrialTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        model_root = self.root / ('models--' + candidate.SPEC['model_repository'].replace('/', '--'))
        model = model_root / 'snapshots' / candidate.SPEC['model_revision']
        model.mkdir(parents=True)
        (model_root / 'blobs').mkdir()
        config = model / 'config.json'
        config.write_text(json.dumps({
            'model_type': 'glm_moe_dsa', 'num_hidden_layers': 78, 'hidden_size': 6144,
            'n_routed_experts': 256, 'max_position_embeddings': 1048576}))
        marker = self.root / 'download-marker'
        marker.write_text('prepared fixture')
        for patcher in [
            mock.patch.dict(os.environ, {
                'IMAGE': 'example.invalid/candidate@sha256:' + 'a' * 64,
                'MODEL_DIR_HOST': str(model), 'DOWNLOAD_MARKER_HOST': str(marker),
            }, clear=True),
            mock.patch.object(candidate, 'MODEL_FILES', {
                'files': [{'path': 'config.json', 'size': config.stat().st_size}]}),
        ]:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_default_parity_and_explicit_floor_preserve_context_and_isolate_storage(self):
        parity_root, parity = candidate.settings()
        with mock.patch.dict(os.environ, {'MAINTENANCE_TRIAL': 'rental-floor'}):
            floor_root, floor = candidate.settings()
        self.assertEqual(parity['MAINTENANCE_TRIAL'], 'parity')
        self.assertEqual(parity['NAME'], candidate.SPEC['default_name'])
        self.assertEqual(floor['NAME'], parity['NAME'] + '-rental-floor')
        self.assertNotEqual(parity_root, floor_root)
        for key in ('CACHE_VOLUME', 'STATE_VOLUME', 'LMCACHE_DISK_HOST'):
            self.assertTrue(Path(parity[key]).is_relative_to(parity_root))
            self.assertTrue(Path(floor[key]).is_relative_to(floor_root))
            self.assertNotEqual(parity[key], floor[key])
        for key, expected in {
            'MAX_NUM_SEQS': ('12', '8'), 'MAX_NUM_BATCHED_TOKENS': ('3072', '2048'),
            'VLLM_EXL3_PREFILL_CAPACITY': ('3072', '1024'),
            'GPU_MEMORY_UTILIZATION': ('0.95', '0.93'),
            'MAX_CUDAGRAPH_CAPTURE_SIZE': ('48', '32'),
            'VLLM_EXL3_TRELLIS_MAX_M': ('48', '32'), 'LMCACHE_L1_INIT_GB': ('125', '20'),
        }.items():
            self.assertEqual((parity[key], floor[key]), expected)
        for key, expected in {
            'MAX_MODEL_LEN': '520192', 'KV_CACHE_MEMORY_BYTES': '4518907904',
            'TENSOR_PARALLEL_SIZE': '4', 'DCP': '4', 'MTP_TOKENS': '3',
            'MTP_DRAFT': 'native', 'MTP_DRAFT_SAMPLE_METHOD': 'probabilistic',
            'LMCACHE_L1_MAX_GB': '125', 'PREFIX_CACHE_DISK_GB': '384',
            'MODEL_PROFILE': 'glm53-3.42bpw-500k',
        }.items():
            self.assertEqual(parity[key], expected)
            self.assertEqual(floor[key], expected)

    def test_unknown_selector_rejected_before_lifecycle_side_effects(self):
        with mock.patch.dict(os.environ, {'MAINTENANCE_TRIAL': 'automatic'}), \
             mock.patch.object(candidate, 'podman') as podman, \
             mock.patch.object(candidate, 'launch') as launch:
            for command in ('stage', 'verify-model', 'preflight', 'config-smoke', 'start',
                            'qualification-commands', 'run-qualification', 'qualify',
                            'rollback', 'prepare-reboot', 'boot-check'):
                with self.subTest(command=command), \
                     mock.patch.object(sys, 'argv', ['candidate.py', command]):
                    with self.assertRaisesRegex(ValueError, 'MAINTENANCE_TRIAL'):
                        candidate.main()
            with mock.patch.dict(os.environ, {'RUN_GPU_QUALIFICATION': '1'}), \
                 mock.patch.object(sys, 'argv', ['run_qualification.py']):
                with self.assertRaisesRegex(ValueError, 'MAINTENANCE_TRIAL'):
                    run_qualification.main()
            with self.assertRaisesRegex(ValueError, 'MAINTENANCE_TRIAL'):
                candidate.settings()
            podman.assert_not_called()
            launch.assert_not_called()

    def test_existing_stage_cannot_change_trial_even_with_same_explicit_name(self):
        _, parity = candidate.settings()
        path = self.root / 'stage.json'
        path.write_text(json.dumps(candidate.manifest(parity)))
        before = path.read_bytes()
        candidate.staged(self.root, parity)
        with mock.patch.dict(os.environ, {
            'MAINTENANCE_TRIAL': 'rental-floor', 'NAME': parity['NAME'],
        }):
            _, floor = candidate.settings()
            with self.assertRaisesRegex(ValueError, 'identity/configuration changed'):
                candidate.staged(self.root, floor)
            with mock.patch.object(candidate, 'location', return_value=(self.root, parity['NAME'])), \
                 mock.patch.object(candidate, 'podman') as podman:
                for command in ('boot-check', 'rollback'):
                    with self.subTest(command=command), \
                         mock.patch.object(sys, 'argv', ['candidate.py', command]):
                        with self.assertRaisesRegex(ValueError, 'maintenance trial differs'):
                            candidate.main()
                podman.assert_not_called()
        self.assertEqual(path.read_bytes(), before)

    def test_qualification_child_resolves_the_selected_stage(self):
        with mock.patch.dict(os.environ, {'MAINTENANCE_TRIAL': 'rental-floor'}):
            root, env = candidate.settings()

            def child(command, *, env, check):
                with mock.patch.dict(os.environ, env, clear=True):
                    child_root, child_env = run_qualification.deployment.settings()
                self.assertEqual((child_root, child_env), (root, expected))

            expected = env
            with mock.patch.object(candidate, 'staged', return_value='stage'), \
                 mock.patch.object(candidate.subprocess, 'run', side_effect=child), \
                 mock.patch.object(sys, 'argv', ['candidate.py', 'run-qualification']):
                candidate.main()

    def test_floor_boot_uses_saved_stage_without_requiring_model_settings(self):
        with mock.patch.dict(os.environ, {'MAINTENANCE_TRIAL': 'rental-floor'}):
            _, env = candidate.settings()
            path = self.root / 'stage.json'
            path.write_text(json.dumps(candidate.manifest(env)))
            evidence = self.root / 'evidence.json'
            evidence.write_text('{}')
            (self.root / 'qualification.json').write_text(json.dumps({
                'qualified': True, 'stage_sha256': candidate.digest(path),
                'evidence_path': str(evidence), 'evidence_sha256': candidate.digest(evidence)}))
            with mock.patch.object(candidate, 'location', return_value=(self.root, env['NAME'])), \
                 mock.patch.object(candidate, 'settings', side_effect=ValueError('no launch settings at boot')), \
                 mock.patch.object(candidate, 'qualify'), \
                 mock.patch.object(candidate, 'inspect', return_value={'State': {'Running': True}}), \
                 mock.patch.object(sys, 'argv', ['candidate.py', 'boot-check']):
                with self.assertRaisesRegex(ValueError, 'production is already running'):
                    candidate.main()


if __name__ == '__main__':
    unittest.main()
