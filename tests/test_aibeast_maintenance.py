"""Maintenance qualification and rollback boundaries; no GPU or Podman daemon."""
import json
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
                'environment': {'NAME': name}, 'production_container': production}))
            (root / 'qualification.json').write_text('{"qualified":true}')
            (root / ('container-' + name + '.service')).write_text('[Service]\n')
            actions = []

            def operation(*args, **kwargs):
                self.assertTrue((root / 'boot-disabled').exists())
                self.assertFalse((root / 'qualification.json').exists())
                actions.append(args)

            with mock.patch.object(candidate, 'location', return_value=(root, name)), \
                 mock.patch.object(candidate, 'exists', return_value=True), \
                 mock.patch.object(candidate, 'candidate'), \
                 mock.patch.object(candidate, 'inspect', return_value={'State': {'Running': False}}), \
                 mock.patch.object(candidate, 'podman', side_effect=operation), \
                 mock.patch.object(candidate.subprocess, 'run', side_effect=FileNotFoundError('systemctl')), \
                 mock.patch.object(sys, 'argv', ['candidate.py', 'rollback']):
                candidate.main()
            self.assertEqual(actions, [('stop', '-t', '120', name), ('start', production)])
            with mock.patch.object(candidate, 'location', return_value=(root, name)), \
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


if __name__ == '__main__':
    unittest.main()
