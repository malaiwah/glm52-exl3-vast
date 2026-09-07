#!/usr/bin/env python3
"""Explicit local-only candidate lifecycle; never stops production for a launch.

Inputs: IMAGE (required repository@sha256), MODEL_DIR_HOST (pinned HF snapshot),
DOWNLOAD_MARKER_HOST (prepared completion marker). Optional NAME and STAGE_ROOT.
Invoke with Python 3; stage/preflight/config-smoke do not run GPU workloads.
Qualification gate consumes --evidence JSON with stage_sha256 and reports:
needles, features, offload, cache, load. Each report reference has path and sha256.
Cache/load reports require complete=true, stage_sha256 and checks named by
candidate.json, each with ok=true and an artifact reference (path and sha256).
Those operational fault/concurrency receipts must come from actual candidate
experiments, not the config smoke or a healthy HTTP endpoint.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SPEC = json.loads((HERE / "candidate.json").read_text())
LAUNCHER = REPO / "scripts/run-local-podman.sh"
MODEL_FILES = json.loads((HERE / "model-files.json").read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def podman(*args, capture=False):
    return subprocess.run(["podman", *args], check=True, text=True,
                          stdout=subprocess.PIPE if capture else None).stdout


def exists(name):
    code = subprocess.run(["podman", "container", "exists", name]).returncode
    require(code in (0, 1), f"Podman container lookup failed ({code}): {name}")
    return code == 0


def inspect(name):
    return json.loads(podman("inspect", name, capture=True))[0]


def location():
    name = os.environ.get("NAME", SPEC["default_name"])
    require(re.fullmatch(re.escape(SPEC["default_name"]) + r"(?:-[a-zA-Z0-9_-]+)?", name),
            "NAME must use the isolated candidate prefix")
    root = (Path(os.environ.get("STAGE_ROOT", SPEC["default_root"])) / name).resolve()
    require(root.is_relative_to(Path("/mnt/fast")), "STAGE_ROOT must resolve beneath /mnt/fast")
    return root, name


def settings():
    image = os.environ.get("IMAGE", "")
    require(re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image),
            "IMAGE must be an explicit repository@sha256:<64 hex>; no tag/default/pull")
    root, name = location()
    default_model = Path(SPEC["default_model_root"]) / "snapshots" / SPEC["model_revision"]
    model = Path(os.environ.get("MODEL_DIR_HOST", str(default_model)))
    require(model.is_absolute() and model.is_dir(), "MODEL_DIR_HOST must be a prepared absolute HF snapshot")
    require(model.parent.name == "snapshots" and model.name == SPEC["model_revision"],
            "checkpoint must be the exact pinned HF snapshot (no mutable main)")
    model_root = model.parent.parent
    require(model_root.name == "models--" + SPEC["model_repository"].replace("/", "--"),
            "HF model repository directory does not match the pinned full GLM-5.3 model")
    require((model_root / "blobs").is_dir(), "HF model repository is missing blobs/")
    marker = Path(os.environ.get("DOWNLOAD_MARKER_HOST", ""))
    require(marker.is_absolute() and marker.is_file(), "DOWNLOAD_MARKER_HOST must be a prepared absolute file")
    cfg = json.loads((model / "config.json").read_text())
    for field, expected in {"model_type": "glm_moe_dsa", "num_hidden_layers": 78,
                            "hidden_size": 6144, "n_routed_experts": 256}.items():
        require(cfg.get(field) == expected, f"not the FULL GLM-5.3 architecture: {field}")
    for path in model.rglob("*"):
        require(not path.is_symlink() or path.exists(), f"dangling checkpoint symlink: {path}")
    # Use the pinned file inventory rather than reparsing the 82 MB tensor
    # index on every lifecycle command. verify-model checks every object hash.
    for item in MODEL_FILES["files"]:
        path = model / item["path"]
        require(path.is_file() and path.stat().st_size == item["size"],
                f"missing/incomplete pinned runtime file: {item['path']}")
    env = dict(SPEC["environment"])
    require(int(env["MAX_MODEL_LEN"]) >= 520192, "refusing to shrink AIBeast's 520192-token envelope")
    require(cfg.get("max_position_embeddings", 0) >= int(env["MAX_MODEL_LEN"]),
            "checkpoint positional limit is smaller than the candidate envelope")
    env.update(IMAGE=image, NAME=name, MODEL_DIR_HOST=str(model),
               DOWNLOAD_MARKER_HOST=str(marker), CACHE_VOLUME=str(root / "compile-cache"),
               STATE_VOLUME=str(root / "state"), LMCACHE_DISK_HOST=str(root / "lmcache"))
    return root, env


def manifest(env):
    return {"schema": 1, "environment": env,
            "model_revision": SPEC["model_revision"], "model_repository": SPEC["model_repository"],
            "production_container": SPEC["production_container"],
            "sources": {str(path.relative_to(REPO)): digest(path)
                        for path in [HERE / "candidate.json", HERE / "candidate.py",
                                     HERE / "model-files.json", HERE / "memory-comparison.json",
                                     HERE / "run_qualification.py", LAUNCHER,
                                     REPO / "tests/test_scopedlmcache_retrieve.py",
                                     REPO / "scripts/benchmark_serving.py", REPO / "scripts/verify_serving.py",
                                     REPO / "scripts/needle_matrix.py", REPO / "scripts/feature_suite.py",
                                     REPO / "scripts/offload_prefix_benchmark.py"]},
            "checkpoint_config_sha256": digest(Path(env["MODEL_DIR_HOST"]) / "config.json"),
            "marker_sha256": digest(env["DOWNLOAD_MARKER_HOST"])}


def staged(root, env):
    path = root / "stage.json"
    require(path.is_file(), "run stage first; no implicit staging during start")
    require(json.loads(path.read_text()) == manifest(env),
            "stage identity/configuration changed; use a fresh NAME/STAGE_ROOT")
    return digest(path)


def file_state(path):
    stat = path.stat()
    return [str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def verify_model(root, env, stage_hash, full=False):
    model = Path(env["MODEL_DIR_HOST"])
    require(MODEL_FILES["revision"] == SPEC["model_revision"], "model manifest revision mismatch")
    receipt_path = root / "model-integrity.json"
    if full:
        verified = {}
        for item in MODEL_FILES["files"]:
            path = model / item["path"]
            before = file_state(path)
            require(path.stat().st_size == item["size"], f"model file size mismatch: {path}")
            if "sha256" in item:
                actual = digest(path)
                expected = item["sha256"]
            else:
                content = path.read_bytes()
                actual = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
                expected = item["git_blob_sha1"]
            require(actual == expected, f"pinned HF object hash mismatch: {path}")
            require(file_state(path) == before, f"model file changed during verification: {path}")
            verified[item["path"]] = before
            print(f"Verified {item['path']}", flush=True)
        receipt_path.write_text(json.dumps({"stage_sha256": stage_hash, "files": verified}, indent=2) + "\n")
    else:
        require(receipt_path.is_file(), "run verify-model first (local full-file hashes; no downloads)")
        receipt = json.loads(receipt_path.read_text())
        require(receipt["stage_sha256"] == stage_hash, "model integrity receipt belongs to a different stage")
        require(set(receipt["files"]) == {item["path"] for item in MODEL_FILES["files"]},
                "model integrity receipt is incomplete")
        for name, state in receipt["files"].items():
            require(file_state(model / name) == state, f"model file changed since hash verification: {name}")


def launch(env, **overrides):
    # Do not inherit accidental tuning/profile/cache overrides from the shell.
    clean = {key: value for key, value in os.environ.items()
             if key in {"PATH", "HOME", "USER", "LOGNAME", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS"}}
    subprocess.run(["bash", str(LAUNCHER)], env={**clean, **env, **overrides}, check=True)


def candidate(root, env):
    doc = inspect(env["NAME"])
    require(doc["Config"]["Image"] == env["IMAGE"], "candidate name now belongs to a different image")
    mounts = {mount["Destination"]: mount["Source"] for mount in doc.get("Mounts", [])}
    require(mounts.get("/state") == str(root / "state"), "candidate state mount does not match stage")
    return doc


def artifact(reference, base):
    path = (base / reference["path"]).resolve()
    require(path.is_file() and path.stat().st_size > 0, f"missing/empty evidence artifact: {path}")
    require(digest(path) == reference["sha256"], f"evidence digest mismatch: {path}")
    return path


def qualify(root, env, evidence):
    stage_hash = staged(root, env)
    verify_model(root, env, stage_hash)
    source = Path(evidence).resolve()
    doc = json.loads(source.read_text())
    require(doc["stage_sha256"] == stage_hash, "evidence is not bound to this pinned stage")
    start = json.loads((root / "start.json").read_text())
    require(start["stage_sha256"] == stage_hash and doc["container_id"] == start["container_id"],
            "evidence does not identify this candidate launch")
    for name in ("container", "gpu_initial", "gpu_final"):
        artifact(doc["diagnostics"][name], source.parent)
    reports = {name: json.loads(artifact(doc["reports"][name], source.parent).read_text())
               for name in ("needles", "features", "offload", "cache", "load")}
    for name in ("needles", "features", "offload"):
        report = reports[name]
        require(report.get("complete") is True and report.get("ok") is True,
                f"{name} did not complete successfully")
        require(report.get("model") == env["SERVED_MODEL_NAME"], f"{name} tested a different model alias")
        require(report.get("base_url") == f"http://127.0.0.1:{env['PORT']}",
                f"{name} tested a different endpoint")
    needles = reports["needles"]
    require(needles.get("max_model_len") == int(env["MAX_MODEL_LEN"]), "needle envelope differs from stage")
    require(needles.get("prompt_identity_policy") == "fresh-uuid", "cold context gate forbids reused prompt identity")
    probes = needles.get("probes", [])
    require(probes and all(p.get("ok") is True and p.get("tokens_exact") is True for p in probes),
            "context probes must all succeed with tokenizer-measured counts")
    require(max(p.get("tokens", 0) for p in probes) >= 500000,
            "no successful exact-token >=500K prompt; configured capacity is not evidence")
    for name in ("short_prompt", "stochastic_sampling"):
        require(needles.get("post_stress", {}).get(name, {}).get("ok") is True,
                f"post-stress {name} failed/missing")
    require(reports["offload"].get("requests_ok") is True, "offload requests failed")
    for kind in ("cache", "load"):
        report = reports[kind]
        require(report.get("stage_sha256") == stage_hash and report.get("complete") is True,
                f"{kind} evidence must be complete and bound to this stage")
        for name in SPEC[f"required_{kind}_checks"]:
            check = report.get("checks", {}).get(name, {})
            require(check.get("ok") is True, f"missing/failed {kind} gate: {name}")
            artifact(check["artifact"], source.parent)
    receipt = {"qualified": True, "stage_sha256": stage_hash,
               "evidence_path": str(source), "evidence_sha256": digest(source),
               "note": "Qualification only; no production cutover or reboot policy change performed"}
    (root / "qualification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["stage", "verify-model", "preflight", "config-smoke", "start",
                                           "qualification-commands", "run-qualification", "qualify", "rollback",
                                           "prepare-reboot", "boot-check"])
    parser.add_argument("--evidence", help="actual qualification evidence manifest JSON")
    args = parser.parse_args()
    if args.command == "boot-check":
        root, name = location()
        require(not (root / "boot-disabled").exists(),
                "candidate boot eligibility was revoked by rollback")
        env = json.loads((root / "stage.json").read_text())["environment"]
        require(env["NAME"] == name, "boot container identity differs from stage")
        stage_hash = staged(root, env)
        receipt = json.loads((root / "qualification.json").read_text())
        require(receipt.get("qualified") is True and receipt.get("stage_sha256") == stage_hash,
                "boot requires qualification for this exact stage")
        require(digest(receipt["evidence_path"]) == receipt["evidence_sha256"], "qualification evidence changed")
        qualify(root, env, receipt["evidence_path"])
        require(not inspect(SPEC["production_container"])["State"]["Running"], "production is already running")
        candidate(root, env)
        return
    if args.command == "rollback":
        # Rollback must work even when candidate weights/config are broken.
        root, name = location()
        saved = json.loads((root / "stage.json").read_text())
        env = saved["environment"]
        require(env["NAME"] == name, "stage container identity does not match requested rollback")
        production = SPEC["production_container"]
        require(saved["production_container"] == production, "unexpected production rollback identity")
        require(exists(production), "preserved production rollback container is missing")
        # Revoke eligibility before any stop: an enabled systemd unit must not
        # resurrect this candidate on a later boot or during rollback.
        (root / "boot-disabled").write_text("revoked by explicit rollback\n")
        (root / "qualification.json").unlink(missing_ok=True)
        unit_path = root / ("container-" + name + ".service")
        if unit_path.is_file():
            control = ["systemctl"] + ([] if os.geteuid() == 0 else ["--user"])
            try:
                stopped_unit = subprocess.run(
                    [*control, "disable", "--now", unit_path.name],
                    text=True, capture_output=True, timeout=30)
                unit_error = stopped_unit.stderr.strip() if stopped_unit.returncode else ""
            except (OSError, subprocess.TimeoutExpired) as error:
                unit_error = str(error)
            if unit_error:
                print("Warning: systemd unit disable failed; persistent boot revocation remains active: "
                      + unit_error, file=sys.stderr)
        if exists(name):
            candidate(root, env)
            podman("stop", "-t", "120", name)
        if not inspect(production)["State"]["Running"]:
            podman("start", production)
        print("Production rollback running; candidate container/image/state retained, nothing removed")
        return
    root, env = settings()
    if args.command == "stage":
        require(not root.exists(), f"stage already exists: {root}; choose a fresh NAME")
        root.mkdir(parents=True)
        for directory in ("state", "compile-cache", "lmcache", "smoke-state", "smoke-cache", "evidence"):
            (root / directory).mkdir()
        (root / "stage.json").write_text(json.dumps(manifest(env), indent=2, sort_keys=True) + "\n")
        print(f"Staged {root}; no container, image pull, download, or GPU operation performed")
        return
    stage_hash = staged(root, env)
    if args.command == "verify-model":
        verify_model(root, env, stage_hash, full=True)
    elif args.command == "config-smoke":
        smoke = dict(env, NAME=env["NAME"] + "-config-smoke", CONFIG_SMOKE="1",
                     CACHE_VOLUME=str(root / "smoke-cache"), STATE_VOLUME=str(root / "smoke-state"))
        launch(smoke)
        code = podman("wait", smoke["NAME"], capture=True).strip()
        podman("logs", smoke["NAME"])
        require(code == "0", f"config smoke exited {code}; retained container for inspection")
    elif args.command in ("preflight", "start"):
        verify_model(root, env, stage_hash)
        require(shutil.disk_usage(root).free >= int(env["PREFIX_CACHE_DISK_GB"]) * 1024**3,
                "insufficient /mnt/fast free space for the configured L2 cache cap")
        launch(env, LAUNCH_PREFLIGHT="1")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", int(env["PORT"])))
        if args.command == "start":
            production = SPEC["production_container"]
            require(exists(production), "preserved production rollback container is missing")
            require(not inspect(production)["State"]["Running"],
                    "production is still running: no GPU co-residency; schedule an explicit maintenance window")
            cold_cache = not any((root / "compile-cache").iterdir())
            launch(env)
            started = candidate(root, env)
            (root / "start.json").write_text(json.dumps({
                "stage_sha256": stage_hash, "container_id": started["Id"],
                "compile_cache_empty": cold_cache,
            }, indent=2) + "\n")
        else:
            print("Preflight passed; production was not stopped. Start requires a separate maintenance decision.")
    elif args.command == "qualification-commands":
        print("RUN_GPU_QUALIFICATION=1 " + shlex.join([
            sys.executable, str(HERE / "candidate.py"), "run-qualification"]))
        print("# Runs real GPU context/features/C1C4C8/cache pressure/restarts/service fault injection.")
        print("# Produces all receipts automatically and runs the promotion gate; no cutover.")
        print("# Deadline ownership is tested on the installed adapter with CPU-controlled futures.")
        print("# This does not inject or claim a 180-second GPU DMA hang.")
    elif args.command == "run-qualification":
        subprocess.run([sys.executable, str(HERE / "run_qualification.py")], check=True)
    elif args.command == "qualify":
        require(args.evidence, "qualify requires --evidence; health/config smoke cannot promote")
        qualify(root, env, args.evidence)
    elif args.command == "prepare-reboot":
        require(os.environ.get("ENABLE_REBOOT") == "1", "requires explicit ENABLE_REBOOT=1")
        receipt = json.loads((root / "qualification.json").read_text())
        require(receipt.get("qualified") is True and receipt.get("stage_sha256") == stage_hash,
                "candidate is not qualified for this stage")
        require(digest(receipt["evidence_path"]) == receipt["evidence_sha256"], "qualification evidence changed")
        qualify(root, env, receipt["evidence_path"])
        require(not inspect(SPEC["production_container"])["State"]["Running"], "production is still running")
        candidate(root, env)
        # Podman 4.9 cannot update a container's restart policy. Generate its
        # supported systemd lifecycle instead, without enabling/starting it.
        unit = podman("generate", "systemd", "--name", "--restart-policy=on-failure",
                      env["NAME"], capture=True)
        require("[Service]\n" in unit and "[Unit]\n" in unit, "unexpected generated systemd unit")
        unit = unit.replace("[Unit]\n", "[Unit]\nRequiresMountsFor=" +
                            shlex.join([str(root), env["MODEL_DIR_HOST"], env["DOWNLOAD_MARKER_HOST"]]) + "\n", 1)
        boot_env = " ".join(json.dumps(value) for value in
                            ("NAME=" + env["NAME"], "STAGE_ROOT=" + str(root.parent)))
        unit = unit.replace("[Service]\n", "[Service]\nEnvironment=" + boot_env + "\nExecStartPre=" +
                            shlex.join([sys.executable, str(HERE / "candidate.py"), "boot-check"]) + "\n", 1)
        unit_path = root / ("container-" + env["NAME"] + ".service")
        unit_path.write_text(unit)
        (root / "boot-disabled").unlink(missing_ok=True)
        control = ["systemctl"] + ([] if os.geteuid() == 0 else ["--user"])
        print("Prepared qualified boot unit; keep this checkout at its recorded path.")
        print("Explicit opt-in command (not executed): " + shlex.join([*control, "enable", str(unit_path)]))
        print("No --now, container restart-policy change, production cutover, or host service mutation performed.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        sys.exit(4)
