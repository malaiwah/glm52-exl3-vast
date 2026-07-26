#!/usr/bin/env python3
"""Session erase: destroy the evidence that YOU were on this hardware, before
the instance is handed back to the provider and to whoever rents it next.

WHAT THIS DOES NOT DO — and why that is the right call, not an omission.
It does not touch the model weights. The checkpoint is public: it comes from
Hugging Face, anyone can download the identical bytes, and finding it on a
recycled disk tells the next tenant nothing except that someone ran a popular
open-weights model. Overwriting ~332 GB to hide a public fact would take the
better part of an hour on NVMe (longer on network storage), which is why the
earlier version of this feature was impractical. Dropping it makes the
remaining surface small, fast and complete: keys, certificates, configuration,
logs, prompts, history, and anything YOU added.

WHAT IT CANNOT GUARANTEE. Overwriting a file is a request to the storage
stack, not a physical act:
  * SSDs with wear levelling and over-provisioning may retain the old physical
    pages after a logical overwrite. There is no way for a container to reach
    them, and TRIM/sanitize is not available to us.
  * Copy-on-write and overlay filesystems (Docker's overlay2, btrfs, ZFS) write
    the new bytes elsewhere and unlink the old extent, which stays on the media.
  * Network-backed volumes (RunPod network volumes, NFS) are overwritten on a
    server we do not control, with its own snapshotting and replication.
  * Provider snapshots and backups taken before the erase are untouched.
  * Anything already sent off the box — the instance CONSOLE LOG in the
    provider dashboard, which is where every one of this container's stdout
    lines went, including prompts if request logging was on — cannot be reached
    from in here at all.
So: this reliably removes the data from the filesystem and makes casual and
most forensic-tool recovery fail. It is not a media sanitisation. If your
threat model includes an adversary with physical access to the flash, do not
put the secret on rented hardware in the first place.

Everything is path-injectable so the test suite runs against a temporary tree:
no test ever touches /root, /tmp or a real model directory.
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm_config as gc  # noqa: E402

CHUNK = 1 << 20
# Without a download manifest we cannot tell a user's 4 GiB adapter from a
# shard of the public checkpoint, so files at or above this size are REPORTED
# rather than erased. Small unknown files are erased: a stray notes.txt is far
# more likely to be yours than to be part of a safetensors repo.
LARGE_FILE_MB = 64

# Files the template itself creates under the model dir from PUBLIC inputs.
# They are derived artifacts, not user data, and not sensitive: compile caches,
# the vision/MTP overlays, and the config backups the graft/vision scripts keep.
DERIVED_UNDER_MODEL = (
    ".vllm-cache", ".mtp78-overlay", ".mtp78-draft", ".vision", ".cache",
    ".download-complete", ".mtp78-grafted", ".vision-enabled",
)
DERIVED_SUFFIXES = (".orig", ".text-only", ".pre-qa-ignore-backup")


class Paths:
    """Every root this module may touch. Overridable so tests stay in a sandbox."""

    def __init__(self, env=None):
        # effective_env, not os.environ: on RunPod the HF cache variables (and
        # everything else) are injected into PID 1 only, so an erase started
        # from an SSH session would otherwise miss the network-volume paths
        # entirely. MEASURED on a live pod.
        env = gc.effective_env(env)
        self.env = env
        self.model_dir = env.get("MODEL_DIR", "/workspace/GLM-5.2-EXL3-TR3-3.0bpw")
        self.workspace = env.get("ERASE_WORKSPACE", env.get("MODEL_ROOT", "/workspace"))
        self.home = env.get("ERASE_HOME", "/root")
        self.tmp = env.get("ERASE_TMP", "/tmp")
        self.var_log = env.get("ERASE_VARLOG", "/var/log")
        self.state_dir = gc.state_dir()
        self.runtime_dir = gc.runtime_dir()
        self.status_file = env.get("STATUS_FILE", "/tmp/glm-boot-status.json")
        self.keyfile = os.path.join(os.path.dirname(self.model_dir.rstrip("/")) or "/workspace",
                                    ".vllm-api-key")
        # Extra roots that are not under any of the above. On RunPod a network
        # volume mounts at /runpod-volume and PID 1's environment points the
        # whole Hugging Face cache at it by default (VERIFIED: HF_HOME=
        # /runpod-volume/.cache/huggingface/), so the HF token lands on storage
        # that SURVIVES termination and keeps billing.
        extra = env.get("ERASE_EXTRA_ROOTS", "/runpod-volume")
        self.extra_roots = [r for r in extra.split(":") if r and os.path.isdir(r)]

    def hf_cache_roots(self):
        """Everywhere a Hugging Face token can land, from the environment that
        the container actually started with."""
        roots = []
        for var in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "HF_DATASETS_CACHE",
                    "TRANSFORMERS_CACHE", "HF_HUB_CACHE"):
            v = self.env.get(var)
            if v:
                roots.append(v)
        xdg = self.env.get("XDG_CACHE_HOME")
        roots.append(os.path.join(xdg, "huggingface") if xdg
                     else os.path.join(self.home, ".cache", "huggingface"))
        for r in self.extra_roots:
            roots.append(os.path.join(r, ".cache", "huggingface"))
        seen, out = set(), []
        for r in roots:
            r = r.rstrip("/")
            if r and r not in seen:
                seen.add(r)
                out.append(r)
        return out


# --------------------------------------------------------------------------
# what counts as the public checkpoint
# --------------------------------------------------------------------------

def public_manifest(model_dir):
    """-> set of repo-relative paths that came from the Hub, or None.

    huggingface_hub's snapshot_download(local_dir=...) "will create a
    .cache/huggingface/ folder at the root of local_dir to store some metadata
    related to the downloaded files"
    (https://huggingface.co/docs/huggingface_hub/en/guides/download). Each
    downloaded file leaves a sidecar whose name ends in .metadata, mirroring the
    repo layout. We accept ANY layout under that folder that ends in .metadata,
    because the exact nesting has moved between hub versions and we would rather
    be version-agnostic than precise.

    UNVERIFIED: the exact sidecar naming was not exercised against a real
    download here (no network, no 332 GB). When the folder is absent or empty
    this returns None and the caller degrades to the size rule, which is the
    behaviour a user gets on any hub version we guessed wrong about.
    """
    root = os.path.join(model_dir, ".cache", "huggingface")
    if not os.path.isdir(root):
        return None
    names = set()
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if not f.endswith(".metadata"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, f), root)
            rel = rel[: -len(".metadata")]
            parts = rel.split(os.sep)
            if parts and parts[0] in ("download", "downloads"):
                parts = parts[1:]
            if parts:
                names.add(os.sep.join(parts))
    return names or None


def _is_derived(rel):
    top = rel.split(os.sep)[0]
    return top in DERIVED_UNDER_MODEL or rel.endswith(DERIVED_SUFFIXES)


def user_files_under_model(model_dir):
    """-> (targets, unknown_large, manifest_used)

    Anything under the model dir that is neither a file the Hub put there nor a
    derived artifact of this template is, by elimination, something the user
    added: an adapter, a dataset, notes, a script."""
    if not os.path.isdir(model_dir):
        return [], [], False
    manifest = public_manifest(model_dir)
    targets, unknown_large = [], []
    for dirpath, dirs, files in os.walk(model_dir):
        dirs[:] = [d for d in dirs
                   if not _is_derived(os.path.relpath(os.path.join(dirpath, d), model_dir))]
        for f in files:
            path = os.path.join(dirpath, f)
            rel = os.path.relpath(path, model_dir)
            if _is_derived(rel):
                continue
            if manifest is not None and rel in manifest:
                continue          # public checkpoint file: deliberately kept
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if manifest is None and size >= LARGE_FILE_MB * (1 << 20):
                unknown_large.append({"path": path, "size": size})
                continue
            targets.append({"path": path, "size": size, "group": "user-files",
                            "why": ("under the model dir but not part of the public "
                                    "checkpoint")})
    return targets, unknown_large, manifest is not None


# --------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------

def _add(targets, path, group, why, seen):
    try:
        if not os.path.isfile(path) or os.path.islink(path):
            if os.path.islink(path):
                targets.append({"path": path, "size": 0, "group": group, "why": why})
            return
        real = os.path.realpath(path)
        if real in seen:
            return
        seen.add(real)
        targets.append({"path": path, "size": os.path.getsize(path),
                        "group": group, "why": why})
    except OSError:
        pass


def _add_tree(targets, root, group, why, seen, skip=()):
    if not os.path.isdir(root):
        return
    skip = tuple(os.path.realpath(s) for s in skip if s)
    for dirpath, dirs, files in os.walk(root):
        if os.path.realpath(dirpath).startswith(skip):
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs
                   if not os.path.realpath(os.path.join(dirpath, d)).startswith(skip)]
        for f in files:
            _add(targets, os.path.join(dirpath, f), group, why, seen)


def plan(paths=None, keep=()):
    """Everything that says a particular person used this box.

    `keep` is the worker's own progress file and anything else that must
    survive until the terminate call has been made."""
    p = paths or Paths()
    targets, seen = [], set()
    keep = tuple(os.path.realpath(k) for k in keep if k)

    # 1. credentials -------------------------------------------------------
    _add(targets, p.keyfile, "credentials",
         "the persisted vLLM API key — grants use of your endpoint", seen)
    _add(targets, p.status_file, "credentials",
         "boot status file; it carries the API key in plaintext for the landing page", seen)
    # ~/.runpod/config.toml is where runpodctl persists the account API key —
    # VERIFIED on a real install, and it writes it 0644 (world-readable), so it
    # is exactly the kind of file that must not be left on a recycled disk.
    for rel in (".cache/huggingface/token", ".huggingface/token",
                ".cache/huggingface/stored_tokens", ".vast_api_key", ".runpod.yaml",
                ".runpod/config.toml", ".config/vastai/vast_api_key",
                ".config/runpod/config.toml"):
        _add(targets, os.path.join(p.home, rel), "credentials",
             "provider or Hugging Face credential", seen)
    _add_tree(targets, os.path.join(p.home, ".runpod"), "credentials",
              "runpodctl configuration (holds the account API key, written 0644)", seen)
    # The Hugging Face TOKEN, wherever the cache lives — including a network
    # volume that termination does not clear. The cached model BLOBS are left
    # alone for the same reason the weights are: they are public bytes.
    for root in p.hf_cache_roots():
        for name in ("token", "stored_tokens"):
            _add(targets, os.path.join(root, name), "credentials",
                 f"Hugging Face token in {root}", seen)
    _add_tree(targets, os.path.join(p.home, ".ssh"), "credentials",
              "SSH keys / authorized_keys / known_hosts — identifies you and your hosts",
              seen)

    # 2. TLS material ------------------------------------------------------
    _add_tree(targets, os.path.join(p.workspace, ".lego"), "tls",
              "Let's Encrypt account key, certificate and PRIVATE KEY for your domain",
              seen)

    # 3. configuration state ----------------------------------------------
    for name in ("config.json", "known-good.json", "apply-state.json",
                 "verify-last.json", "checkpoint-baseline.json",
                 "terminate-switches.json"):
        _add(targets, os.path.join(p.state_dir, name), "config-state",
             "your saved configuration and its history", seen)
    _add_tree(targets, os.path.join(p.state_dir, "failures"), "config-state",
              "preserved failed configurations, their boot logs and the model's "
              "written analysis of them", seen)

    # 4. logs — where prompts end up ---------------------------------------
    _add_tree(targets, os.path.join(p.state_dir, "logs"), "logs",
              "engine logs; with request/trace logging on these contain prompt text",
              seen)
    for root in (p.workspace, p.model_dir):
        if not os.path.isdir(root):
            continue
        for f in os.listdir(root):
            if f.endswith((".log", ".jsonl", ".ndjson")):
                _add(targets, os.path.join(root, f), "logs",
                     "log file at the volume root", seen)
    _add_tree(targets, p.var_log, "logs", "system logs", seen)
    for root in p.extra_roots:
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                if f.endswith((".log", ".jsonl", ".ndjson")):
                    _add(targets, os.path.join(dirpath, f), "logs",
                         "log file on a mounted volume that outlives this instance", seen)

    # 5. runtime scratch ---------------------------------------------------
    _add_tree(targets, p.runtime_dir, "runtime",
              "resolved config, env snapshot, verification verdicts", seen,
              skip=keep)
    _add_tree(targets, p.tmp, "runtime", "temporary files", seen,
              skip=tuple(keep) + (p.runtime_dir,))

    # 6. shell / editor history -------------------------------------------
    for rel in (".bash_history", ".python_history", ".viminfo", ".zsh_history",
                ".lesshst", ".wget-hsts", ".nano_history",
                ".ipython/profile_default/history.sqlite",
                ".jupyter/lab/workspaces", ".local/share/jupyter/runtime"):
        _add(targets, os.path.join(p.home, rel), "history",
             "what you typed on this box", seen)

    # 7. anything you added under the model dir ----------------------------
    user, unknown_large, manifest_used = user_files_under_model(p.model_dir)
    for t in user:
        real = os.path.realpath(t["path"])
        if real in seen:
            continue
        seen.add(real)
        targets.append(t)

    targets = [t for t in targets if not os.path.realpath(t["path"]).startswith(keep)]
    return {"targets": targets,
            "unknown_large": unknown_large,
            "manifest_used": manifest_used,
            "total_bytes": sum(t["size"] for t in targets),
            "count": len(targets),
            "groups": sorted({t["group"] for t in targets})}


# --------------------------------------------------------------------------
# doing it
# --------------------------------------------------------------------------

def overwrite_file(path, passes=1):
    """One pass of random bytes, flushed to the device, then unlink under a
    renamed inode. More passes are theatre on any storage made after 1995 and
    cost time we may not have before the instance goes away."""
    size = os.path.getsize(path)
    if size:
        with open(path, "r+b", buffering=0) as f:
            for _ in range(max(1, passes)):
                f.seek(0)
                remaining = size
                while remaining > 0:
                    n = min(CHUNK, remaining)
                    f.write(os.urandom(n))
                    remaining -= n
                f.flush()
                os.fsync(f.fileno())
        with open(path, "r+b", buffering=0) as f:
            f.truncate(0)
            os.fsync(f.fileno())
    d = os.path.dirname(path)
    tmp = os.path.join(d, "." + os.urandom(8).hex())
    try:
        os.rename(path, tmp)
        os.unlink(tmp)
    except OSError:
        os.unlink(path)
    return size


def erase(plan_doc, dry_run=False, progress=None):
    done, failed, bytes_done = 0, [], 0
    for t in plan_doc["targets"]:
        path = t["path"]
        try:
            if dry_run:
                done += 1
                bytes_done += t["size"]
            elif os.path.islink(path):
                os.unlink(path)
                done += 1
            elif os.path.isfile(path):
                bytes_done += overwrite_file(path)
                done += 1
        except OSError as e:
            failed.append({"path": path, "error": str(e)})
        if progress and done % 25 == 0:
            progress(done, len(plan_doc["targets"]))
    # empty directories left behind (failures/, logs/, .lego/, .ssh/ contents)
    if not dry_run:
        for root in {os.path.dirname(t["path"]) for t in plan_doc["targets"]}:
            try:
                if os.path.isdir(root) and not os.listdir(root):
                    os.rmdir(root)
            except OSError:
                pass
    return {"erased": done, "bytes": bytes_done, "failed": failed,
            "dry_run": bool(dry_run)}


def erase_ram(max_gib=None, dry_run=False):
    """Drop the page cache, then overwrite free RAM.

    The page cache is where file contents (prompts, keys) linger after the file
    is gone. Overwriting free memory additionally displaces anything the KV
    cache and the allocator left behind. Bounded, because an unbounded
    allocation on a box whose RAM the provider may have oversubscribed is how
    you get OOM-killed before the terminate call goes out."""
    out = {"drop_caches": "not attempted", "overwritten_bytes": 0}
    if dry_run:
        out["drop_caches"] = "skipped (dry run)"
        return out
    try:
        subprocess.run(["sync"], check=False, timeout=60)
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
        out["drop_caches"] = "ok"
    except OSError as e:
        out["drop_caches"] = f"unavailable in this container ({e})"

    avail = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) * 1024
                    break
    except OSError:
        pass
    cap = int((max_gib or 32) * (1 << 30))
    budget = min(int(avail * 0.6), cap)
    if budget <= 0:
        out["note"] = "no free RAM budget to overwrite"
        return out
    block = 256 << 20
    written = 0
    try:
        buf = os.urandom(1 << 20) * 256          # 256 MiB of non-zero bytes
        holder = []
        while written + block <= budget:
            holder.append(bytearray(buf))
            written += block
        del holder
        out["overwritten_bytes"] = written
    except MemoryError:
        out["overwritten_bytes"] = written
        out["note"] = "stopped early at MemoryError"
    return out


def erase_vram(dry_run=False):
    """Best-effort: fill every visible GPU with zeros, then release.

    Only meaningful once vLLM has exited — the worker stops the engine first.
    Never runs in tests; there is no torch and no GPU on a build machine, and
    the caller must ask for it explicitly."""
    out = {"devices": [], "status": "not attempted"}
    if dry_run:
        out["status"] = "skipped (dry run)"
        return out
    try:
        import torch
    except Exception as e:
        out["status"] = f"torch unavailable ({e})"
        return out
    try:
        if not torch.cuda.is_available():
            out["status"] = "no CUDA devices visible"
            return out
        for i in range(torch.cuda.device_count()):
            got = 0
            try:
                torch.cuda.set_device(i)
                free, _total = torch.cuda.mem_get_info(i)
                blocks, block = [], 512 << 20
                while free - got > block:
                    b = torch.empty(block, dtype=torch.uint8, device=f"cuda:{i}")
                    b.zero_()
                    blocks.append(b)
                    got += block
                del blocks
                torch.cuda.synchronize(i)
                torch.cuda.empty_cache()
            except Exception as e:                       # OOM is the expected end
                out.setdefault("notes", []).append(f"cuda:{i}: {e}")
            out["devices"].append({"device": i, "zeroed_bytes": got})
        out["status"] = "ok"
    except Exception as e:
        out["status"] = f"failed ({e})"
    return out


GUARANTEES = {
    "does": [
        "overwrites and unlinks every file listed in the plan, one pass of random "
        "bytes, flushed to the device with fsync",
        "removes the API key, the TLS private key, the config state, every log this "
        "template writes (including preserved failure logs and the model's analyses), "
        "shell history, SSH material, provider and Hugging Face credentials, and "
        "anything you added under the model dir",
        "drops the page cache and overwrites free RAM, and can zero VRAM once the "
        "engine has stopped",
    ],
    "does_not": [
        "does NOT erase the public model weights — they are downloadable by anyone "
        "and hiding them buys nothing while costing ~332 GB of writes",
        "does NOT guarantee the old bytes are gone from the physical flash: SSD wear "
        "levelling, over-provisioning, copy-on-write and overlay filesystems all keep "
        "superseded extents out of our reach",
        "does NOT touch network-backed volumes' server-side copies, provider "
        "snapshots, or backups taken before now",
        "does NOT reach the instance console log in the provider dashboard, which has "
        "every line this container printed to stdout",
        "does NOT erase PID 1's environment, which on RunPod holds PUBLIC_KEY (your "
        "account's SSH public keys) and the pod-scoped API key. The copy written into "
        "~/.ssh/authorized_keys IS erased; the copy in /proc/1/environ dies with the "
        "container, not with this erase",
        "is NOT media sanitisation: it defeats recovery tools, not an adversary with "
        "the flash in hand",
    ],
}


def main(argv):
    ap = argparse.ArgumentParser(description="plan or run the session erase")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--run", action="store_true",
                    help="actually erase (default is to print the plan only)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    doc = plan()
    if args.run:
        doc["result"] = erase(doc, dry_run=args.dry_run)
    if args.json:
        print(json.dumps(doc, indent=1))
    else:
        for g in doc["groups"]:
            items = [t for t in doc["targets"] if t["group"] == g]
            print(f"{g}: {len(items)} file(s), "
                  f"{sum(i['size'] for i in items) / 2**20:.1f} MiB")
        print(f"total {doc['count']} files, {doc['total_bytes'] / 2**20:.1f} MiB")
        if doc["unknown_large"]:
            print(f"NOT erased ({len(doc['unknown_large'])} large files could not be "
                  "distinguished from the public checkpoint):")
            for u in doc["unknown_large"][:20]:
                print(f"  {u['path']} ({u['size'] / 2**30:.1f} GiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
