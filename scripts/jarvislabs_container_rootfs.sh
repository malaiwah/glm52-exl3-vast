#!/usr/bin/env bash
# Run the turnkey appliance inside a container-only GPU host (JarvisLabs
# container instances, including spot) where no Docker or Podman daemon exists.
#
# JarvisLabs' framework containers cannot run a container runtime, so the
# appliance image is fetched with skopeo, unpacked with umoci - which is the
# part that gets whiteouts and opaque directories right - and entered with
# chroot. The unpacked tree is the exact published image content: every
# critical source is re-verified against the image's own build-time
# /opt/runtime-provenance.json before anything is launched.
#
# Subcommands: prepare | verify | smoke | run | stop
#
# The whole script executes through main() at the bottom so a truncated
# `curl | bash` transfer cannot execute a destructive prefix.
set -Eeuo pipefail

UMOCI_VERSION=0.6.0
UMOCI_SHA256=b51c267ec394499e42c6fde47f240b7b7dba57ea49df0b5acd304378b82a3b71

log() { printf '>>> %s\n' "$*"; }
fatal() { printf 'FATAL: %s\n' "$*" >&2; exit 2; }

require_root() {
  [[ "$(id -u)" == 0 ]] || fatal "run this as root inside the GPU container."
}

require_digest_pin() {
  # A tag would let the served bytes change between the verification and the
  # launch, which is the whole point of the pinned lineage.
  [[ "$IMAGE" =~ ^[^[:space:]@]+@sha256:[0-9a-f]{64}$ ]] ||
    fatal "TURNKEY_IMAGE must be an immutable repository@sha256:digest reference."
}

ensure_tools() {
  local missing=()
  command -v skopeo >/dev/null 2>&1 || missing+=(skopeo)
  command -v tar >/dev/null 2>&1 || missing+=(tar)
  command -v curl >/dev/null 2>&1 || missing+=(curl)
  command -v jq >/dev/null 2>&1 || missing+=(jq)
  command -v findmnt >/dev/null 2>&1 || missing+=(util-linux)
  command -v rsync >/dev/null 2>&1 || missing+=(rsync)
  if [[ "${#missing[@]}" -gt 0 ]]; then
    log "installing: ${missing[*]}"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq "${missing[@]}"
  fi
  if [[ ! -x "$UMOCI_BIN" ]]; then
    log "fetching umoci $UMOCI_VERSION"
    curl --fail --show-error --location --retry 4 --retry-all-errors \
      --connect-timeout 15 --max-time 300 -o "$UMOCI_BIN.tmp" \
      "https://github.com/opencontainers/umoci/releases/download/v${UMOCI_VERSION}/umoci.linux.amd64"
    printf '%s  %s\n' "$UMOCI_SHA256" "$UMOCI_BIN.tmp" | sha256sum -c - ||
      fatal "umoci download does not match its pinned hash."
    chmod +x "$UMOCI_BIN.tmp"
    mv -f "$UMOCI_BIN.tmp" "$UMOCI_BIN"
  fi
}

purge() {
  # The image deliberately ships read-only directories (/opt/soul and the
  # pinned overlay payloads), and umoci reproduces those modes, so a plain
  # rm -rf of a previous unpack stops halfway with ENOTEMPTY.
  local path
  for path in "$@"; do
    [[ -e "$path" ]] || continue
    chmod -R u+rwX "$path"
    rm -rf "$path"
  done
}

fetch_and_unpack() {
  local oci="$ROOT/oci" bundle="$ROOT/bundle"
  if [[ -f "$ROOTFS/.unpacked" ]] &&
     [[ "$(cat "$ROOTFS/.unpacked")" == "$IMAGE" ]]; then
    log "rootfs for $IMAGE is already unpacked; reusing it"
    return 0
  fi
  # A partial previous unpack is not a base to build on.
  purge "$bundle" "$oci"
  log "copying $IMAGE into an OCI layout"
  skopeo copy --override-os linux --override-arch amd64 \
    "docker://$IMAGE" "oci:$oci:candidate"
  log "unpacking the image rootfs (whiteouts and opaque dirs included)"
  "$UMOCI_BIN" unpack --image "$oci:candidate" "$bundle"
  [[ -x "$ROOTFS/usr/local/bin/model-turnkey-entry.sh" ]] ||
    fatal "unpacked tree has no appliance entrypoint; wrong image?"
  printf '%s\n' "$IMAGE" >"$ROOTFS/.unpacked"
  # The layout blobs are a second full copy of the image; the rootfs is the
  # artifact, and spot storage is not free.
  purge "$oci"
}

inject_driver() {
  # The image ships its own CUDA user space but never the host driver. The
  # container runtime injected the driver into THIS namespace, so copy exactly
  # the driver libraries - never the CUDA toolkit runtimes, whose versions the
  # image pins itself - into the directory the image already has on its
  # LD_LIBRARY_PATH.
  local target="$ROOTFS/usr/local/nvidia/lib64" lib base tool
  local -a libs=()
  install -d "$target" "$ROOTFS/usr/local/nvidia/bin"
  while read -r lib; do
    base="${lib##*/}"
    # Driver components carry the driver version as their soname suffix.
    [[ "$base" =~ ^lib(cuda|nvidia-[^/]+|nvcuvid|nvoptix|nvidia-ml)\.so\.[0-9] ]] || continue
    [[ -e "$lib" ]] && libs+=("$lib")
  done < <(ldconfig -p 2>/dev/null | sed -n 's/.*=> \(\/.*\)$/\1/p' | sort -u)
  [[ "${#libs[@]}" -gt 0 ]] ||
    fatal "no host NVIDIA driver libraries found; is this a GPU instance?"
  cp -a --dereference --no-clobber "${libs[@]}" "$target/" || true
  ( cd "$target" && for lib in libcuda.so.[0-9]*.[0-9]*; do
      [[ -e "$lib" ]] || continue
      ln -sf "$lib" libcuda.so.1
      ln -sf libcuda.so.1 libcuda.so
    done
    for lib in libnvidia-ml.so.[0-9]*.[0-9]*; do
      [[ -e "$lib" ]] || continue
      ln -sf "$lib" libnvidia-ml.so.1
    done )
  [[ -e "$target/libcuda.so.1" ]] ||
    fatal "no host libcuda driver library was injected; CUDA cannot initialize."
  for tool in nvidia-smi nvidia-debugdump; do
    if command -v "$tool" >/dev/null 2>&1; then
      cp -a --dereference "$(command -v "$tool")" "$ROOTFS/usr/local/nvidia/bin/"
      ln -sf "/usr/local/nvidia/bin/$tool" "$ROOTFS/usr/bin/$tool"
    fi
  done
  log "injected ${#libs[@]} host driver libraries"
}

mount_runtime() {
  local path
  install -d "$WORKSPACE" "$ROOTFS/workspace" "$ROOTFS/cache" "$ROOTFS/state"
  # Bind the host's /proc, /sys and /dev rather than mounting fresh instances:
  # a fresh procfs needs its own PID namespace and a fresh sysfs needs to own
  # the network namespace, and the appliance needs the host network and the
  # injected /dev/nvidia* devices as they are.
  for path in proc sys dev; do
    mountpoint -q "$ROOTFS/$path" && continue
    mount --rbind "/$path" "$ROOTFS/$path"
    mount --make-rslave "$ROOTFS/$path"
  done
  # vLLM's workers exchange tensors through /dev/shm; a 64 MiB default
  # deadlocks multi-GPU startup rather than failing cleanly.
  if [[ "$(findmnt -no SIZE --bytes "$ROOTFS/dev/shm" 2>/dev/null || echo 0)" -lt $((16 * 1024 ** 3)) ]]; then
    mount -t tmpfs -o "size=${SHM_SIZE:-64g},mode=1777" shm "$ROOTFS/dev/shm"
  fi
  mountpoint -q "$ROOTFS/workspace" || mount --bind "$WORKSPACE" "$ROOTFS/workspace"
  cp -f /etc/resolv.conf "$ROOTFS/etc/resolv.conf"
}

unmount_runtime() {
  local path
  for path in "$ROOTFS/workspace" "$ROOTFS/dev/shm" "$ROOTFS/dev" \
              "$ROOTFS/sys" "$ROOTFS/proc"; do
    mountpoint -q "$path" && umount -R "$path"
  done
  return 0
}

image_env() {
  # umoci writes the image's own configuration into the OCI bundle. Reuse it
  # verbatim: the appliance depends on the base image's NCCL selection, EXL3
  # encoder paths and JIT cache namespaces, and inventing a PATH here would
  # silently serve with a different runtime contract than the built image.
  local config="$ROOT/bundle/config.json"
  [[ -f "$config" ]] || fatal "missing $config; re-run prepare."
  jq -r '.process.env[]' "$config"
}

verify_program() {
  # Compare bytes, not filesystem layout: a graft exposes the appliance's
  # trees through symlinks, so resolved paths legitimately differ while every
  # installed source must still hash exactly as the build recorded.
  cat <<'PY'
import json, os, sys
sys.path.insert(0, "/opt/scripts")
import write_runtime_provenance as provenance
recorded = json.load(open("/opt/runtime-provenance.json"))
fresh = provenance.collect(recorded["parent_image"],
                           recorded["appliance_source_revision"])
PREFIX = os.environ.get("TURNKEY_ROOTFS_PREFIX", "")


def normalize(path):
    # A graft reaches the appliance's trees through symlinks, so importlib
    # reports the unpacked location; compare image-relative paths.
    if PREFIX and path.startswith(PREFIX):
        return path[len(PREFIX):]
    return path


def sources(report):
    return {normalize(row["path"]): row["sha256"]
            for row in report["critical_sources"]}


def modules(report):
    return {name: info["source"]["sha256"]
            for name, info in report["modules"].items()}


for name, extract in (("critical_sources", sources), ("modules", modules)):
    before, after = extract(recorded), extract(fresh)
    if before != after:
        differing = sorted(k for k in before.keys() | after.keys()
                           if before.get(k) != after.get(k))
        raise SystemExit(
            f"runtime differs from the built image: {name}: {differing[:5]}")
print(json.dumps({"verified": True,
                  "critical_sources": len(fresh["critical_sources"]),
                  "modules": len(fresh["modules"]),
                  "source_commit": recorded["appliance_source_revision"]}))
PY
}

verify_rootfs() {
  # Re-derive the appliance's own provenance inside the unpacked tree and
  # compare it to what the build recorded. This catches a truncated unpack, a
  # wrong architecture, and a tampered registry copy, without a GPU.
  local -a env_args=()
  mount_runtime
  mapfile -t env_args < <(image_env)
  log "verifying installed sources against the image's build-time provenance"
  verify_program | chroot "$ROOTFS" env -i "${env_args[@]}" /opt/venv/bin/python -
}

appliance_env() {
  # The image environment first, then this launch's overrides: env applies
  # duplicates in order, so the later assignment wins.
  local key value
  image_env
  printf 'MODEL_PROFILE=%s\n' "$PROFILE"
  printf 'LANDING_PAGE=%s\n' "${LANDING_PAGE:-1}"
  printf 'OPEN_BUTTON_PORT=%s\n' "${OPEN_BUTTON_PORT:-1111}"
  printf 'HOME=/root\n'
  for key in HF_TOKEN VLLM_API_KEY AUTH MODEL_FAMILY MODEL_VARIANT MODEL_ID \
             MODEL_DIR SERVED_MODEL_NAME TENSOR_PARALLEL_SIZE MAX_MODEL_LEN \
             MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS GPU_MEMORY_UTILIZATION \
             KV_CACHE_MEMORY_BYTES OFFLOAD_FRACTION PREFIX_CACHE_BACKEND \
             PREFIX_CACHE_DISK_GB LMCACHE_L1_MAX_GB LMCACHE_L1_READ_TTL \
             LMCACHE_SESSION_TTL_SECONDS LMCACHE_RETRIEVE_TIMEOUT_SECONDS \
             VLLM_EXL3_PREFILL_CAPACITY FEATURE_TEST_LEVEL SSHD OPEN_BUTTON_TOKEN \
             GLM_STATE_DIR TERMINATE_ENABLED CONFIG_SMOKE GLM_GPU_COUNT; do
    value="${!key:-}"
    [[ -z "$value" ]] && continue
    [[ "$value" == *$'\n'* ]] && fatal "$key contains a newline."
    printf '%s=%s\n' "$key" "$value"
  done
}

mounts_available() {
  local probe rc=0
  probe="$(mktemp -d)"
  mount --bind "$probe" "$probe" 2>/dev/null || rc=1
  [[ "$rc" == 0 ]] && umount "$probe"
  rmdir "$probe"
  return "$rc"
}

native_verify() {
  # Same fail-closed check as the chroot path, run in place after a graft.
  local -a env_args=()
  mapfile -t env_args < <(image_env)
  log "verifying grafted sources against the image's build-time provenance"
  verify_program | env -i "${env_args[@]}" "TURNKEY_ROOTFS_PREFIX=$ROOTFS" \
    /opt/venv/bin/python -
}

graft() {
  # Containers on managed GPU clouds keep CAP_SYS_CHROOT but drop
  # CAP_SYS_ADMIN and block user namespaces, so neither a chroot (no /proc,
  # no /dev) nor an unprivileged mount namespace can host the appliance.
  # Install the image over the rented container instead and run it in the
  # host namespace, where /proc, /sys, /dev/nvidia* and /dev/shm already are.
  # This mutates the rented container: it is one-way and only ever correct on
  # a disposable instance.
  local dir entry name
  [[ "${TURNKEY_GRAFT:-0}" == 1 ]] ||
    fatal "this host cannot mount; re-run with TURNKEY_GRAFT=1 to install the image over this disposable container."
  if [[ -f /.turnkey-grafted ]] && [[ "$(cat /.turnkey-grafted)" == "$IMAGE" ]]; then
    log "container already carries $IMAGE"
    return 0
  fi
  log "grafting the image over this container's filesystem"
  # The appliance's own trees are large and self-contained: expose them by
  # symlink at the absolute paths the image bakes in, instead of copying
  # tens of GiB twice onto the same instance.
  for entry in "$ROOTFS"/opt/*; do
    [[ -e "$entry" ]] || continue
    name="${entry##*/}"
    if [[ -L "/opt/$name" ]]; then
      rm -f "/opt/$name"
    elif [[ -e "/opt/$name" ]]; then
      purge "/opt/$name"
    fi
    ln -sfn "$entry" "/opt/$name"
  done
  # Everything else is the OS layer the appliance was built against, notably a
  # newer glibc than a managed container image ships. rsync stages each file
  # and renames it into place, so an already-mapped library keeps its old
  # inode and running processes (this shell, sshd) survive the swap. The host
  # driver files the container runtime bind-mounted must win and cannot be
  # unlinked at all, so they are excluded by name.
  rsync -aHAX --numeric-ids --force \
    --exclude '/usr/bin/nvidia-*' \
    --exclude '/usr/lib/x86_64-linux-gnu/libnvidia*' \
    --exclude '/usr/lib/x86_64-linux-gnu/libcuda.so*' \
    --exclude '/usr/lib/x86_64-linux-gnu/libnvcuvid*' \
    --exclude '/usr/lib/x86_64-linux-gnu/libnvoptix*' \
    --exclude '/usr/share/doc/**' --exclude '/usr/share/man/**' \
    "$ROOTFS"/usr "$ROOTFS"/lib "$ROOTFS"/lib64 "$ROOTFS"/bin \
    "$ROOTFS"/sbin "$ROOTFS"/var "$ROOTFS"/root /
  # The host's /etc stays in place - it owns DNS, hosts and the provider's
  # own accounts - but two directories in it are part of the image's runtime
  # contract: the alternatives symlinks that make /usr/bin/python resolve to
  # the interpreter the venv was built against, and the loader search paths.
  rsync -aHAX --numeric-ids --force \
    "$ROOTFS"/etc/alternatives "$ROOTFS"/etc/ld.so.conf.d /etc/
  ldconfig
  # Keep the checkpoint, caches and state on the instance's persistent volume.
  for dir in workspace cache state; do
    install -d "$ROOT/$dir"
    if [[ -L "/${dir:?}" ]]; then
      rm -f "/${dir:?}"
    elif [[ -e "/${dir:?}" ]]; then
      purge "/${dir:?}"
    fi
    ln -sfn "$ROOT/$dir" "/$dir"
  done
  printf '%s\n' "$IMAGE" >/.turnkey-grafted
}

enter() {
  local -a env_args=()
  if mounts_available; then
    mount_runtime
    mapfile -t env_args < <(appliance_env)
    log "entering the unpacked appliance rootfs"
    exec chroot "$ROOTFS" env -i "${env_args[@]}" \
      /usr/local/bin/model-turnkey-entry.sh
  fi
  graft
  mapfile -t env_args < <(appliance_env)
  log "launching the grafted appliance in the container namespace"
  exec env -i "${env_args[@]}" /usr/local/bin/model-turnkey-entry.sh
}

main() {
  IMAGE="${TURNKEY_IMAGE:-}"
  ROOT="${TURNKEY_ROOT:-/home/turnkey}"
  ROOTFS="$ROOT/bundle/rootfs"
  WORKSPACE="${TURNKEY_WORKSPACE:-$ROOT/workspace}"
  UMOCI_BIN="${TURNKEY_UMOCI:-/usr/local/bin/umoci}"
  PROFILE="${MODEL_PROFILE:-glm53-3.42bpw-500k}"
  local command="${1:-run}"

  case "$command" in
    prepare|verify|smoke|run|graft)
      IMAGE="${IMAGE:?set TURNKEY_IMAGE to the published repo@sha256:digest}"
      require_digest_pin
      require_root
      install -d "$ROOT"
      ;;
    stop) require_root ;;
    *) fatal "usage: $0 [prepare|verify|smoke|run|graft|stop]" ;;
  esac

  case "$command" in
    prepare)
      ensure_tools
      fetch_and_unpack
      inject_driver
      if mounts_available; then
        verify_rootfs
        unmount_runtime
      else
        log "this host cannot mount: verification happens after 'graft'"
      fi
      log "prepared $ROOTFS"
      ;;
    graft)
      graft
      native_verify
      ;;
    verify)
      if mounts_available; then
        verify_rootfs
        unmount_runtime
      else
        native_verify
      fi
      ;;
    smoke)
      CONFIG_SMOKE=1
      export CONFIG_SMOKE
      enter
      ;;
    run) enter ;;
    stop)
      unmount_runtime
      log "unmounted the appliance rootfs"
      ;;
  esac
}

main "$@"
