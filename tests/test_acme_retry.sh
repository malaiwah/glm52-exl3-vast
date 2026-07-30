#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/bin" "$tmp/model/.lego/certificates"

cat >"$tmp/bin/openssl" <<'SH'
#!/usr/bin/env bash
crt="${@: -1}"
[[ -s "$crt" ]]
SH
cat >"$tmp/bin/lego" <<'SH'
#!/usr/bin/env bash
count=0
[[ ! -f "$ACME_TEST_COUNT" ]] || count="$(cat "$ACME_TEST_COUNT")"
count=$((count + 1))
printf '%s\n' "$count" >"$ACME_TEST_COUNT"
if [[ "$count" -lt "${ACME_TEST_SUCCEED_ON:-2}" ]]; then
  exit 1
fi
mkdir -p "$MODEL_ROOT/.lego/certificates"
printf 'certificate\n' >"$MODEL_ROOT/.lego/certificates/$ACME_DOMAIN.crt"
printf 'private-key\n' >"$MODEL_ROOT/.lego/certificates/$ACME_DOMAIN.key"
SH
cat >"$tmp/bin/timeout" <<'SH'
#!/usr/bin/env bash
shift 3
exec "$@"
SH
chmod +x "$tmp/bin/openssl" "$tmp/bin/lego" "$tmp/bin/timeout"

common=(
  PATH="$tmp/bin:$PATH"
  MODEL_ROOT="$tmp/model"
  ACME_DOMAIN="model.example.test"
  ACME_DNS_PROVIDER="cloudflare"
  ACME_ATTEMPT_TIMEOUT_S="2"
  ACME_BACKGROUND_RETRY_S="0.1"
  ACME_TEST_COUNT="$tmp/count"
)

if env "${common[@]}" ACME_TEST_SUCCEED_ON=2 \
  "$BASH" "$repo/scripts/acme_retry.sh" --once; then
  echo "the deliberately failed first ACME attempt unexpectedly passed" >&2
  exit 1
fi
[[ "$(cat "$tmp/count")" == 1 ]]

env "${common[@]}" ACME_TEST_SUCCEED_ON=2 \
  "$BASH" "$repo/scripts/acme_retry.sh" --retry >"$tmp/retry.log"
[[ "$(cat "$tmp/count")" == 2 ]]
test -f "$tmp/model/.lego/restart-required"
grep -Fq 'background retry succeeded' "$tmp/retry.log"

echo "turnkey bounded ACME retry: PASS"
