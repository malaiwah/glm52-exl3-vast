"""Provenance must follow the encoder loading contract, not import its frontend."""
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import write_runtime_provenance as provenance


class RuntimeProvenanceTests(unittest.TestCase):
    def test_configured_encoder_is_hashed_without_importing_serving_frontend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('vllm', 'b12x', 'torch', 'triton', 'lmcache', 'torchao', 'tilelang'):
                package = root / name
                package.mkdir()
                (package / '__init__.py').write_text('raise AssertionError("do not import GPU runtime")\n')
            encoder = root / 'configured-encoder'
            quantize = encoder / 'modules/quant/exl3_lib/quantize.py'
            quantize.parent.mkdir(parents=True)
            (encoder / '__init__.py').write_text('raise AssertionError("do not import serving frontend")\n')
            content = b'def quantize_exl3():\n    return 1\n'
            quantize.write_bytes(content)
            with mock.patch.object(sys, 'path', [str(root), *sys.path]), \
                 mock.patch.dict(os.environ, {'VLLM_EXL3_ENCODER_SOURCE': str(encoder)}):
                record = provenance.module_records()['exllamav3']
                self.assertEqual(record['encoder']['sha256'], hashlib.sha256(content).hexdigest())
                self.assertEqual(record['package_roots'], [str(encoder)])
                quantize.unlink()
                with self.assertRaises(FileNotFoundError):
                    provenance.module_records()


if __name__ == '__main__':
    unittest.main()
