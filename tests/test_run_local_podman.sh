#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/bin" "$tmp/model"
printf '{}\n' >"$tmp/model/config.json"
printf 'complete\n' >"$tmp/download-complete"

cat >"$tmp/bin/podman" <<'SH'
#!/usr/bin/env bash
if [[ "${1:-}" == container && "${2:-}" == exists ]]; then
  exit 1
fi
if [[ "${1:-}" == run ]]; then
  printf '%s\n' "$@" >"$PODMAN_TEST_ARGS"
  printf 'fake-container-id\n'
  exit 0
fi
printf 'unexpected podman invocation: %s\n' "$*" >&2
exit 2
SH
chmod +x "$tmp/bin/podman"

run_profile() {
  local profile="$1" output="$2"
  PATH="$tmp/bin:$PATH" CONFIG_SMOKE=1 MODEL_PROFILE="$profile" \
    MODEL_DIR_HOST="$tmp/model" DOWNLOAD_MARKER_HOST="$tmp/download-complete" \
    PODMAN_TEST_ARGS="$output" \
    bash "$repo/scripts/run-local-podman.sh" >/dev/null
}

run_profile glm53-3.42bpw "$tmp/glm53.args"
grep -Fxq 'MODEL_PROFILE=glm53-3.42bpw' "$tmp/glm53.args"
grep -Fxq '90m' "$tmp/glm53.args"
if grep -Eq '^MODEL_VARIANT=|^SERVED_MODEL_NAME=' "$tmp/glm53.args"; then
  echo 'public GLM-5.3 profile received an internal identity override' >&2
  exit 1
fi

run_profile qwen36-27b-nvfp4 "$tmp/qwen.args"
grep -Fxq 'MODEL_PROFILE=qwen36-27b-nvfp4' "$tmp/qwen.args"
grep -Fxq '45m' "$tmp/qwen.args"
if grep -Eq '^MODEL_VARIANT=|^SERVED_MODEL_NAME=' "$tmp/qwen.args"; then
  echo 'public Qwen profile received an internal identity override' >&2
  exit 1
fi

PATH="$tmp/bin:$PATH" CONFIG_SMOKE=1 MODEL_PROFILE=custom \
  MODEL_VARIANT=custom MODEL_ID=org/custom QUANTIZATION=awq \
  REASONING_PARSER=custom_reasoning TOOL_CALL_PARSER=custom_tools \
  MULTIMODAL=1 MM_MAX_PIXELS=1048576 SERVED_MODEL_NAME='custom alias' \
  MODEL_DIR_HOST="$tmp/model" DOWNLOAD_MARKER_HOST="$tmp/download-complete" \
  PODMAN_TEST_ARGS="$tmp/explicit.args" \
  bash "$repo/scripts/run-local-podman.sh" >/dev/null
grep -Fxq 'MODEL_VARIANT=custom' "$tmp/explicit.args"
grep -Fxq 'SERVED_MODEL_NAME=custom alias' "$tmp/explicit.args"
grep -Fxq 'MODEL_ID=org/custom' "$tmp/explicit.args"
grep -Fxq 'QUANTIZATION=awq' "$tmp/explicit.args"
grep -Fxq 'REASONING_PARSER=custom_reasoning' "$tmp/explicit.args"
grep -Fxq 'TOOL_CALL_PARSER=custom_tools' "$tmp/explicit.args"
grep -Fxq 'MULTIMODAL=1' "$tmp/explicit.args"
grep -Fxq 'MM_MAX_PIXELS=1048576' "$tmp/explicit.args"

echo 'turnkey local Podman profile forwarding: PASS'
