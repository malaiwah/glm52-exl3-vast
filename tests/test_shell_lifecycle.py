#!/usr/bin/env python3
"""Exercise entrypoint transitions without GPUs, services, or downloads."""
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def shell_functions(*names):
    source = (ROOT / "entrypoint.sh").read_text()
    functions = []
    for name in names:
        start = re.search(r"^" + re.escape(name) + r"\(\) \{\n", source, re.M)
        if start is None:
            raise ValueError(f"shell function not found: {name}")
        body = ""
        for line in source[start.start():].splitlines(keepends=True):
            body += line
            if line.rstrip() == "}":
                parsed = subprocess.run(
                    ["bash", "-n"], input=body, text=True, capture_output=True)
                if parsed.returncode == 0:
                    functions.append(body)
                    break
        else:
            raise ValueError(f"unterminated shell function: {name}")
    return "\n".join(functions)


class LifecycleTests(unittest.TestCase):
    def run_shell(self, code, **env):
        return subprocess.run(["bash", "-c", code], text=True, capture_output=True,
                              env={**os.environ, **env}, timeout=15)

    def test_checkpoint_mismatch_returns_to_supervisor_without_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp)
            (model / ".download-repo").write_text("owner/original")
            (model / "config.json").write_text("{}")
            before = {p.name: p.read_bytes() for p in model.iterdir()}
            result = self.run_shell(shell_functions("fetch_weights") + '''
if fetch_weights; then exit 9; fi
printf 'supervisor-survived\\n'
''', MODEL_DIR=temp, MODEL_REPO="owner/different")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("supervisor-survived", result.stdout)
            self.assertEqual(before, {p.name: p.read_bytes() for p in model.iterdir()})

    def test_ambiguous_lock_selects_new_cache_without_touching_old_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp) / "cache"
            ext = cache / "sparkinfer_pcie_dma_ext"
            ext.mkdir(parents=True)
            lock = ext / "lock"
            lock.write_bytes(b"foreign-live-lock")
            result = self.run_shell(shell_functions("recover_sparkinfer_extension_lock") + '''
boot_note() { :; }
recover_sparkinfer_extension_lock || exit 9
printf '%s\\n' "$TORCH_EXTENSIONS_DIR"
''', TORCH_EXTENSIONS_DIR=str(cache), SCRIPTS_DIR=str(ROOT / "scripts"))
            self.assertEqual(result.returncode, 0, result.stderr)
            selected = Path(result.stdout.strip())
            self.assertNotEqual(selected, cache)
            self.assertTrue(selected.is_dir())
            self.assertEqual(lock.read_bytes(), b"foreign-live-lock")
            self.assertEqual(list(ext.iterdir()), [lock])

    def test_disabled_soul_is_idle_but_event_restarts_level_resolution(self):
        with tempfile.TemporaryDirectory() as temp:
            result = self.run_shell(shell_functions("soul_level_cached", "reconcile_soul") + '''
SOUL_LEVEL_CACHE=""; SOUL_LEVEL_CHECK_AT=0; SOUL_CONFIG_MTIME=""
SOUL_IDLE_STOPPED=0; SOUL_PID=""; SOUL_NEXT_START=0
soul_level() { echo read >> "$SOUL_STATE_DIR/reads"; cat "$SOUL_STATE_DIR/level"; }
stop_soul() { echo stop >> "$SOUL_STATE_DIR/stops"; }
soul_prepare_permissions() { :; }
start_soul() { echo start >> "$SOUL_STATE_DIR/starts"; }
echo 0 > "$SOUL_STATE_DIR/level"
for tick in 1 2 3 4 5; do reconcile_soul; done
echo 1 > "$SOUL_STATE_DIR/level"
touch "$SOUL_RUNTIME_DIR/reconcile"
reconcile_soul
''', SOUL_STATE_DIR=temp, SOUL_RUNTIME_DIR=temp)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((Path(temp) / "reads").read_text().splitlines(), ["read", "read"])
            self.assertEqual((Path(temp) / "stops").read_text().splitlines(), ["stop"])
            self.assertEqual((Path(temp) / "starts").read_text().splitlines(), ["start"])

    def test_overlay_partial_download_retries_before_committing_readiness(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            module = root / "huggingface_hub.py"
            module.write_text('''import os
from pathlib import Path
def snapshot_download(*args, **kwargs):
    target = Path(kwargs['local_dir']) / '3bpw-keep0'
    target.mkdir(parents=True, exist_ok=True)
    (target / 'part.safetensors').write_bytes(b'partial')
    if not (target / 'allow-success').exists():
        raise RuntimeError('interrupted download')
''')
            result = self.run_shell(shell_functions("fetch_mtp78_overlay") + '''
if fetch_mtp78_overlay; then exit 9; fi
test ! -e "$MODEL_DIR/.mtp78-overlay/3bpw-keep0/.download-complete" || exit 10
touch "$MODEL_DIR/.mtp78-overlay/3bpw-keep0/allow-success"
fetch_mtp78_overlay
''', MODEL_DIR=str(root / "model"), PYTHONPATH=temp)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / "model/.mtp78-overlay/3bpw-keep0/.download-complete").is_file())

    def test_nvfp4_source_path_is_data_and_failed_download_never_builds(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = str(root / "quote'\nsource")
            (root / "huggingface_hub.py").write_text('''import os
from pathlib import Path
def snapshot_download(*args, **kwargs):
    Path(os.environ['OBSERVED']).write_text(kwargs['local_dir'])
    raise RuntimeError('download failed')
''')
            (root / "build_nvfp4_mtp_draft.py").write_text("raise SystemExit('builder must not run')")
            result = self.run_shell(shell_functions("prepare_nvfp4_mtp78_draft") + '''
if prepare_nvfp4_mtp78_draft; then exit 9; fi
printf 'recovered\\n'
''', MTP_DRAFT="nvfp4", DRAFT_SOURCE_DIR=source, DRAFT_MODEL=str(root / "draft"),
                SCRIPTS_DIR=temp, PYTHONPATH=temp, OBSERVED=str(root / "observed"))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((root / "observed").read_text(), source)
            self.assertNotIn("builder must not run", result.stderr)
            self.assertIn("recovered", result.stdout)

    def test_failed_vision_install_restores_template_and_does_not_serve(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model"
            vision = model / ".vision"
            vision.mkdir(parents=True)
            (model / "config.json").write_text('{"architectures":["TextModel"]}')
            (model / "chat_template.jinja").write_text("original text template")
            (vision / "chat_template.jinja").write_text("vision template")
            for filename in ("vision_tower.safetensors", "mm_projector.safetensors"):
                (vision / filename).write_bytes(b"payload")
            (root / "huggingface_hub.py").write_text("def snapshot_download(*a, **kw): pass\n")
            (root / "safetensors.py").write_text("def safe_open(*a, **kw): raise AssertionError('not indexing')\n")
            result = self.run_shell(shell_functions("restore_text_vision_state", "prepare_vision") + '''
status_update() { :; }
install_vision_plugin() { return 1; }
if prepare_vision; then exit 9; fi
printf 'supervisor-retry:%s\\n' "$VISION"
''', MODEL_DIR=str(model), VISION="1", SCRIPTS_DIR=str(ROOT / "scripts"), PYTHONPATH=temp)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("supervisor-retry:1", result.stdout)
            self.assertEqual((model / "chat_template.jinja").read_text(), "original text template")
            self.assertEqual(json.loads((model / "config.json").read_text()), {"architectures": ["TextModel"]})
            self.assertFalse((model / ".vision-enabled").exists())

    def test_failed_draft_never_reconciles_or_prepares_vision(self):
        with tempfile.TemporaryDirectory() as temp:
            result = self.run_shell(shell_functions("prepare_checkpoint") + '''
revert_mtp78_before_fetch() { return 0; }
fetch_weights() { return 0; }
prepare_mtp78() { return 1; }
prepare_vision() { touch "$MODEL_DIR/unwanted-vision"; }
if prepare_checkpoint; then exit 9; fi
printf 'supervisor-retry\\n'
''', MODEL_DIR=temp, MODEL_FAMILY="glm52", MODEL_READ_ONLY="0", TERMINATE_FLAG=temp + "/terminate")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((Path(temp) / "unwanted-vision").exists())
            self.assertIn("supervisor-retry", result.stdout)

    def test_bootstrap_refresh_preserves_model_and_disarmed_credential(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = root / "workspace"
            secrets = workspace / ".secrets"
            secrets.mkdir(parents=True)
            envfile = secrets / "appliance.env"
            envfile.write_text("MODEL_PROFILE=glm52-exl3\nMODEL_FAMILY=glm52\n"
                               "MODEL_VARIANT=exl3-tr3\nTERMINATE_ENABLED=1\n"
                               "JARVISLABS_TERMINATE_API_KEY=persisted-secret\n")
            bindir = root / "bin"
            bindir.mkdir()
            for name, script in {
                "docker": "#!/bin/bash\nexit 0\n",
                "sudo": '#!/bin/bash\n[ "${1:-}" != "-n" ] || shift\nexec "$@"\n',
            }.items():
                target = bindir / name
                target.write_text(script)
                target.chmod(0o755)
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith(("MODEL_", "JARVISLABS_", "TERMINATE_", "TURNKEY_"))}
            env.update(PATH=str(bindir) + ":" + os.environ["PATH"],
                       TURNKEY_WORKSPACE=str(workspace), JARVISLABS_MACHINE_ID="123",
                       JARVISLABS_REGION="EU1", PUBLIC_IPADDR="127.0.0.1",
                       TERMINATE_ENABLED="0", TURNKEY_WAIT_FOR_SERVING="0")
            result = subprocess.run(["bash", str(ROOT / "scripts/jarvislabs_vm_bootstrap.sh")],
                                    env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            persisted = dict(line.split("=", 1) for line in envfile.read_text().splitlines())
            self.assertEqual(persisted["MODEL_PROFILE"], "glm52-exl3")
            self.assertEqual(persisted["MODEL_VARIANT"], "exl3-tr3")
            self.assertEqual(persisted["TERMINATE_ENABLED"], "0")
            self.assertEqual(persisted["JARVISLABS_TERMINATE_API_KEY"], "persisted-secret")
            self.assertEqual(envfile.stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
