#!/usr/bin/env bash
# Run the turnkey appliance inside a container-only GPU host (JarvisLabs
# container instances, including spot) where no Docker or Podman daemon exists.
#
# JarvisLabs' framework containers cannot run a container runtime, so the
# appliance image is fetched with skopeo, unpacked with umoci - which is the
# part that gets whiteouts and opaque directories right - and entered with
# chroot where mounts are available. A graft instead replaces selected runtime
# paths in a disposable provider container; it is not OCI isolation or full
# filesystem equivalence. Recorded source and native-library hashes are checked.
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

validate_root() {
  local canonical
  [[ "$ROOT" =~ ^/[A-Za-z0-9_./-]+$ ]] ||
    fatal "TURNKEY_ROOT must be absolute and contain only letters, digits, slash, dot, underscore or hyphen."
  canonical="$(realpath -m -- "$ROOT")"
  [[ "$ROOT" == "$canonical" ]] ||
    fatal "TURNKEY_ROOT must be canonical and have no symlink components."
  case "$ROOT" in
    /home/*|/mnt/*|/srv/*|/tmp/*|/var/tmp/*) ;;
    *) fatal "TURNKEY_ROOT must be a dedicated directory below /home, /mnt, /srv, /tmp or /var/tmp." ;;
  esac
  # /var/tmp itself is not a dedicated appliance directory.
  [[ "$ROOT" != /var/tmp ]] || fatal "unsafe TURNKEY_ROOT."
  local path
  for path in "$ROOT/bundle" "$ROOT/bundle/rootfs" "$ROOT/oci"; do
    [[ ! -L "$path" ]] || fatal "refusing symlink tree: $path"
  done
}

container_marker_present() {
  [[ -f /.dockerenv || -f /run/.containerenv ]]
}

require_container_graft() {
  [[ "${TURNKEY_GRAFT:-0}" == 1 ]] ||
    fatal "graft requires TURNKEY_GRAFT=1 on a disposable container."
  container_marker_present ||
    fatal "graft requires a container marker; refusing to replace a bare host."
}

graft_present() {
  [[ -e /.turnkey-grafted || -L /.turnkey-grafted ||
     -e /.turnkey-graft-rootfs || -L /.turnkey-graft-rootfs ]]
}

require_unmounted_tree() {
  local path="$1" mounts mounted
  mounts="$(findmnt -rn -o TARGET)" || fatal "cannot inspect mounts safely."
  while IFS= read -r mounted; do
    [[ "$mounted" != "$path" && "$mounted" != "$path/"* ]] ||
      fatal "refusing mounted tree: $path ($mounted)"
  done <<<"$mounts"
}

require_matching_link() {
  local link="$1" target="$2"
  if [[ -e "$link" || -L "$link" ]]; then
    [[ -L "$link" && "$(realpath -m -- "$link")" == "$(realpath -m -- "$target")" ]] ||
      fatal "conflicting path $link; preserve or relocate it explicitly before grafting."
  fi
}

require_image_tree() {
  [[ -f "$ROOTFS/.unpacked" && ! -L "$ROOTFS/.unpacked" ]] ||
    fatal "missing unpacked image marker; run prepare first."
  [[ "$(cat "$ROOTFS/.unpacked")" == "$IMAGE" ]] ||
    fatal "rootfs belongs to a different image; use a fresh container and root."
}

validate_workspace() {
  [[ "$WORKSPACE" == "$(realpath -m -- "$WORKSPACE")" ]] ||
    fatal "TURNKEY_WORKSPACE must be an absolute canonical directory, not a symlink."
  case "$WORKSPACE" in
    /home/*|/mnt/*|/srv/*|/tmp/*|/var/tmp/*) ;;
    *) fatal "TURNKEY_WORKSPACE must be a dedicated storage directory, not a runtime path." ;;
  esac
  [[ "$WORKSPACE" != /var/tmp && "$WORKSPACE" != "$ROOT/bundle" &&
     "$WORKSPACE" != "$ROOT/bundle/"* && "$WORKSPACE" != "$ROOT/oci" &&
     "$WORKSPACE" != "$ROOT/oci/"* ]] ||
    fatal "TURNKEY_WORKSPACE cannot overlap an unpack tree."
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
  local path entry
  validate_root
  for path in "$@"; do
    [[ "$path" == "$ROOT/bundle" || "$path" == "$ROOT/oci" ]] ||
      fatal "refusing purge outside the appliance unpack trees: $path"
    [[ ! -L "$path" ]] || fatal "refusing symlink tree: $path"
    require_unmounted_tree "$path"
    if [[ "$path" == "$ROOT/bundle" ]]; then
      ! graft_present ||
        fatal "refusing to purge a grafted or partially grafted rootfs."
      for entry in /opt/* /workspace /cache /state; do
        if [[ -L "$entry" && "$(realpath -m -- "$entry")" == "$ROOTFS/"* ]]; then
          fatal "refusing to purge rootfs linked from $entry."
        fi
      done
      if [[ -e "$ROOTFS/.unpacked" ]]; then
        fatal "refusing to purge an unpacked image; use a fresh root."
      fi
    fi
    [[ -e "$path" ]] || continue
    chmod -R u+rwX "$path"
    rm -rf --one-file-system -- "$path"
  done
}

fetch_and_unpack() {
  local oci="$ROOT/oci" bundle="$ROOT/bundle"
  validate_root
  if [[ -e "$ROOTFS/.unpacked" || -L "$ROOTFS/.unpacked" ]]; then
    require_image_tree
  fi
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
  local target="$ROOTFS/usr/local/nvidia/lib64" lib base tool listing
  local -a libs=()
  local -A basenames=()
  listing="$(ldconfig -p | sed -n 's/.*=> \(\/.*\)$/\1/p' | LC_ALL=C sort -u)" ||
    fatal "cannot enumerate host driver libraries."
  require_unmounted_tree "$ROOTFS"
  while read -r lib; do
    base="${lib##*/}"
    # Driver components carry the driver version as their soname suffix.
    [[ "$base" =~ ^lib(cuda|nvidia-[^/]+|nvcuvid|nvoptix|nvidia-ml)\.so\.[0-9] ]] || continue
    [[ -e "$lib" ]] || fatal "missing host driver library: $lib"
    if [[ -n "${basenames[$base]:-}" ]]; then
      cmp -s -- "${basenames[$base]}" "$lib" ||
        fatal "conflicting driver basename $base: ${basenames[$base]} and $lib"
      continue
    fi
    basenames["$base"]="$lib"
    libs+=("$lib")
  done <<<"$listing"
  [[ "${#libs[@]}" -gt 0 ]] ||
    fatal "no host NVIDIA driver libraries found; is this a GPU instance?"
  install -d "$target" "$ROOTFS/usr/local/nvidia/bin"
  for lib in "${libs[@]}"; do
    base="${lib##*/}"
    if [[ -e "$target/$base" || -L "$target/$base" ]]; then
      cmp -s -- "$lib" "$target/$base" ||
        fatal "existing injected driver differs: $base; use a fresh root."
    else
      cp -a --dereference -- "$lib" "$target/$base"
    fi
  done
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
  validate_workspace
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
  jq -er '.process.env |
    if type == "array" and length > 0 and
       all(.[]; type == "string" and test("^[A-Za-z_][A-Za-z0-9_]*=") and
           (contains("\n") | not))
    then .[] else error("invalid image environment") end' "$config"
}

load_env() {
  # Do not hide a failed producer behind mapfile's process substitution.
  # env_args is local to the caller (Bash dynamic scope).
  local serialized
  serialized="$("$1")" || fatal "cannot load appliance environment."
  mapfile -t env_args <<<"$serialized"
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
    if PREFIX and path.startswith(PREFIX.rstrip("/") + "/"):
        return path[len(PREFIX):]
    return path


def sources(report):
    return {normalize(row["path"]): row["sha256"]
            for row in report["critical_sources"]}


def modules(report):
    return {name: info["source"]["sha256"]
            for name, info in report["modules"].items()}


def native(report):
    libraries = report["native_libraries"]
    result = {"nccl_selected": normalize(libraries["nccl_selected"])}
    for family in ("nccl", "exllama"):
        for row in libraries[family]:
            key = family + ":" + normalize(row["path"])
            digest = row["sha256"]
            if key in result and result[key] != digest:
                raise SystemExit(f"conflicting normalized native library: {key}")
            result[key] = digest
    return result


extra_native = {}
for name, extract in (("critical_sources", sources), ("modules", modules),
                      ("native_libraries", native)):
    before, after = extract(recorded), extract(fresh)
    # A graft retains provider files, unlike an isolated unpack. Verify every
    # recorded native path and the selected NCCL path; disclose additional
    # provider libraries rather than claiming complete filesystem equivalence.
    compared_keys = before.keys() if PREFIX and name == "native_libraries" else before.keys() | after.keys()
    differing = sorted(k for k in compared_keys if before.get(k) != after.get(k))
    if differing:
        raise SystemExit(
            f"runtime differs from the built image: {name}: {differing[:5]}")
    if PREFIX and name == "native_libraries":
        extra_native = {k: after[k] for k in after.keys() - before.keys()}
print(json.dumps({"verified": True,
                  "critical_sources": len(fresh["critical_sources"]),
                  "modules": len(fresh["modules"]),
                  "native_libraries": len(native(fresh)) - 1,
                  "additional_provider_native_libraries": extra_native,
                  "source_commit": recorded["appliance_source_revision"]}))
PY
}

verify_rootfs() {
  # Re-derive the appliance's own provenance inside the unpacked tree and
  # compare it to what the build recorded. This catches a truncated unpack, a
  # wrong architecture, and a tampered registry copy, without a GPU.
  local -a env_args=()
  mount_runtime
  load_env image_env
  log "verifying installed sources against the image's build-time provenance"
  verify_program | chroot "$ROOTFS" env -i "${env_args[@]}" /opt/venv/bin/python -
}

appliance_env() {
  # The image environment first, then this launch's overrides: env applies
  # duplicates in order, so the later assignment wins.
  local key value
  image_env || return
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
             VLLM_EXL3_PREFILL_CAPACITY VLLM_EXL3_ONLINE_CACHE_DIR \
             VLLM_EXL3_ONLINE_CACHE_MODE FEATURE_TEST_LEVEL SSHD OPEN_BUTTON_TOKEN \
             GLM_STATE_DIR TERMINATE_ENABLED CONFIG_SMOKE GLM_GPU_COUNT \
             SUPERVISOR VERIFY SOUL_AUTONOMY_LEVEL PORT; do
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
  load_env image_env
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
  local dir entry name target
  require_container_graft
  require_image_tree
  validate_workspace
  [[ -d /opt && ! -L /opt ]] || fatal "/opt must be a real directory."
  if [[ -e /.turnkey-grafted || -L /.turnkey-grafted ]]; then
    [[ -f /.turnkey-grafted && ! -L /.turnkey-grafted &&
       "$(cat /.turnkey-grafted)" == "$IMAGE" ]] ||
      fatal "container carries a different or invalid graft marker; use a fresh container."
  fi
  if [[ -e /.turnkey-graft-rootfs || -L /.turnkey-graft-rootfs ]]; then
    [[ -f /.turnkey-graft-rootfs && ! -L /.turnkey-graft-rootfs &&
       "$(cat /.turnkey-graft-rootfs)" == "$ROOTFS" ]] ||
      fatal "container is tied to a different graft rootfs."
  fi
  # Preflight every link before changing any runtime paths. Never remove a
  # provider directory or an unrelated link, even when it appears empty.
  for entry in "$ROOTFS"/opt/*; do
    [[ -e "$entry" || -L "$entry" ]] || continue
    if [[ -f /.turnkey-grafted ]]; then
      [[ -L "/opt/${entry##*/}" ]] || fatal "existing graft is missing /opt/${entry##*/}."
    fi
    require_matching_link "/opt/${entry##*/}" "$entry"
  done
  for dir in workspace cache state; do
    target="$ROOT/$dir"
    [[ "$dir" != workspace ]] || target="$WORKSPACE"
    require_matching_link "/$dir" "$target"
    if [[ -f /.turnkey-grafted ]]; then
      [[ -L "/$dir" ]] || fatal "existing graft is missing /$dir."
    fi
  done
  if [[ -f /.turnkey-grafted ]]; then
    native_verify || fatal "existing graft failed provenance verification."
    log "verified existing graft of $IMAGE (selected provenance, not filesystem equivalence)"
    return 0
  fi
  require_unmounted_tree "$ROOTFS"
  printf '%s\n' "$ROOTFS" >/.turnkey-graft-rootfs
  log "grafting the image over this container's filesystem"
  # The appliance's own trees are large and self-contained: expose them by
  # symlink at the absolute paths the image bakes in, instead of copying
  # tens of GiB twice onto the same instance.
  for entry in "$ROOTFS"/opt/*; do
    [[ -e "$entry" || -L "$entry" ]] || continue
    name="${entry##*/}"
    [[ -L "/opt/$name" ]] || ln -s "$entry" "/opt/$name"
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
    --exclude '/usr/lib/xorg/modules/drivers/nvidia_drv.so' \
    --exclude '/usr/share/egl/egl_external_platform.d/15_nvidia_gbm.json' \
    --exclude '/usr/share/glvnd/egl_vendor.d/10_nvidia.json' \
    --exclude '/usr/share/nvidia/nvoptix.bin' \
    --exclude '/usr/share/doc/**' --exclude '/usr/share/man/**' \
    "$ROOTFS"/usr "$ROOTFS"/lib "$ROOTFS"/lib64 "$ROOTFS"/bin \
    "$ROOTFS"/sbin /
  # The host's /etc stays in place - it owns DNS, hosts and the provider's
  # own accounts - but two directories in it are part of the image's runtime
  # contract: the alternatives symlinks that make /usr/bin/python resolve to
  # the interpreter the venv was built against, and the loader search paths.
  rsync -aHAX --numeric-ids --force \
    "$ROOTFS"/etc/alternatives "$ROOTFS"/etc/ld.so.conf.d /etc/
  ldconfig
  # Keep the checkpoint, caches and state on the instance's persistent volume.
  for dir in workspace cache state; do
    target="$ROOT/$dir"
    [[ "$dir" != workspace ]] || target="$WORKSPACE"
    install -d "$target"
    [[ -L "/$dir" ]] || ln -s "$target" "/$dir"
  done
  native_verify || fatal "graft failed provenance verification; completion was not recorded."
  printf '%s\n' "$IMAGE" >/.turnkey-grafted
  log "verified graft of $IMAGE (selected provenance, not filesystem equivalence)"
}

enter() {
  local -a env_args=()
  require_image_tree
  if ! graft_present && mounts_available; then
    verify_rootfs
    load_env appliance_env
    log "entering the unpacked appliance rootfs"
    exec chroot "$ROOTFS" env -i "${env_args[@]}" \
      /usr/local/bin/model-turnkey-entry.sh
  fi
  graft
  load_env appliance_env
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
      validate_root
      [[ "$command" != graft ]] || require_container_graft
      require_root
      install -d "$ROOT"
      ;;
    stop) require_root; validate_root ;;
    *) fatal "usage: $0 [prepare|verify|smoke|run|graft|stop]" ;;
  esac

  case "$command" in
    prepare)
      ensure_tools
      fetch_and_unpack
      if ! graft_present; then
        inject_driver
      fi
      if graft_present; then
        graft
      elif mounts_available; then
        verify_rootfs
        unmount_runtime
      else
        log "this host cannot mount: verification happens after 'graft'"
      fi
      log "prepared $ROOTFS"
      ;;
    graft)
      graft
      ;;
    verify)
      require_image_tree
      if graft_present; then
        graft
      elif mounts_available; then
        verify_rootfs
        unmount_runtime
      else
        fatal "this container cannot mount; run graft with TURNKEY_GRAFT=1 before verification."
      fi
      ;;
    smoke)
      CONFIG_SMOKE=1
      export CONFIG_SMOKE
      enter
      ;;
    run) enter ;;
    stop)
      ! graft_present ||
        fatal "stop cannot terminate graft processes; stop the launch supervisor/process group explicitly or terminate the disposable instance."
      unmount_runtime
      log "unmounted the appliance rootfs"
      ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
