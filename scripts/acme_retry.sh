#!/usr/bin/env bash
# Bounded ACME/deSEC attempt with an optional background retry loop.
set -u

mode="${1:---once}"
case "$mode" in
  --once|--retry) ;;
  *) echo "usage: acme_retry.sh [--once|--retry]" >&2; exit 2 ;;
esac

: "${ACME_DOMAIN:?ACME_DOMAIN is required}"
: "${ACME_DNS_PROVIDER:?ACME_DNS_PROVIDER is required}"
model_root="${MODEL_ROOT:-/workspace}"
lego_path="${LEGO_PATH:-$model_root/.lego}"
scripts_dir="${SCRIPTS_DIR:-/opt/scripts}"
attempt_timeout="${ACME_ATTEMPT_TIMEOUT_S:-150}"
retry_delay="${ACME_BACKGROUND_RETRY_S:-300}"
crt="$lego_path/certificates/${ACME_DOMAIN}.crt"
key="$lego_path/certificates/${ACME_DOMAIN}.key"

cert_valid() {
  [[ -f "$crt" && -f "$key" ]] &&
    openssl x509 -checkend 604800 -noout -in "$crt" >/dev/null 2>&1
}

# The token travels in a 0600 header file, never argv: this helper re-runs
# every ~300 s for the life of the container, and /proc/*/cmdline is world-
# readable inside it (including by the sandboxed soul account).
#
# Created ONCE here, in the parent shell — a lazy creator called as
# $(desec_header_file) would assign its cache inside the command-substitution
# subshell, so every call would mint (and leak) another token-bearing file for
# the whole life of the retry loop. Removed on exit.
DESEC_HDR=""
if [[ -n "${DESEC_TOKEN:-}" ]]; then
  if DESEC_HDR="$(umask 077 && mktemp "${GLM_RUNTIME_DIR:-/tmp}/acme-desec-hdr.XXXXXX")"; then
    printf 'Authorization: Token %s\n' "$DESEC_TOKEN" > "$DESEC_HDR" || DESEC_HDR=""
  else
    DESEC_HDR=""
  fi
  [[ -n "$DESEC_HDR" ]] ||
    echo "!!! ACME retry: could not stage the deSEC auth header file"
fi
trap '[[ -z "$DESEC_HDR" ]] || rm -f "$DESEC_HDR"' EXIT

# The deSEC subname for ACME_DOMAIN under DESEC_DOMAIN: the label(s) below the
# zone, or "" for the apex (deSEC's apex representation). Returns non-zero when
# ACME_DOMAIN is neither the apex nor within the zone, so callers refuse to
# write a record rather than construct the doubled name example.com.example.com
# (a bare suffix-strip leaves sub==ACME_DOMAIN in exactly that case).
desec_subname() {
  if [[ "$ACME_DOMAIN" == "$DESEC_DOMAIN" ]]; then
    printf ''
    return 0
  fi
  case "$ACME_DOMAIN" in
    *."$DESEC_DOMAIN") printf '%s' "${ACME_DOMAIN%."$DESEC_DOMAIN"}" ;;
    *) return 1 ;;
  esac
}

cleanup_challenge() {
  [[ "$ACME_DNS_PROVIDER" == desec && -n "${DESEC_DOMAIN:-}" &&
     -n "${DESEC_TOKEN:-}" ]] || return 0
  [[ -n "$DESEC_HDR" ]] || return 0
  local sub challenge
  sub="$(desec_subname)" || return 0
  if [[ -n "$sub" ]]; then challenge="_acme-challenge.${sub}"; else challenge="_acme-challenge"; fi
  curl -sf --max-time 15 -X DELETE \
    "https://desec.io/api/v1/domains/${DESEC_DOMAIN}/rrsets/${challenge}/TXT/" \
    -H @"$DESEC_HDR" >/dev/null 2>&1 || true
}

register_desec() {
  [[ "$ACME_DNS_PROVIDER" == desec && -n "${DESEC_DOMAIN:-}" &&
     -n "${DESEC_TOKEN:-}" ]] || return 0
  local sub
  if ! sub="$(desec_subname)"; then
    echo "!!! ACME retry: ACME_DOMAIN=$ACME_DOMAIN is not the apex of, or within, the deSEC zone $DESEC_DOMAIN; refusing to register a bogus record" >&2
    return 1
  fi
  local ip="${ACME_PUBLIC_IP:-${PUBLIC_IPADDR:-${RUNPOD_PUBLIC_IP:-}}}"
  [[ -n "$ip" ]] || {
    echo "!!! ACME retry: public IP is not available for deSEC registration"
    return 1
  }
  [[ -n "$DESEC_HDR" ]] || {
    echo "!!! ACME retry: the deSEC auth header file is unavailable"
    return 1
  }
  curl -sf --max-time 20 -X PUT \
    "https://desec.io/api/v1/domains/${DESEC_DOMAIN}/rrsets/" \
    -H @"$DESEC_HDR" -H "Content-Type: application/json" \
    -d "[{\"subname\":\"${sub}\",\"type\":\"A\",\"ttl\":3600,\"records\":[\"${ip}\"]}]" \
    >/dev/null
}

attempt() {
  cert_valid && return 0
  if ! register_desec; then
    echo "!!! ACME retry: DNS registration failed; the appliance will try again"
    return 1
  fi

  local guard_pid="" rc=1
  local -a propagation=()
  if [[ "$ACME_DNS_PROVIDER" == desec && -n "${DESEC_DOMAIN:-}" &&
        -f "$scripts_dir/desec_acme_guard.py" ]]; then
    /opt/venv/bin/python "$scripts_dir/desec_acme_guard.py" \
      --zone "$DESEC_DOMAIN" --domain "$ACME_DOMAIN" \
      --timeout "${DESEC_PROPAGATION_TIMEOUT:-300}" &
    guard_pid=$!
    propagation=(--dns.propagation-wait "${DESEC_LEGO_PROPAGATION_WAIT:-45s}")
  fi
  echo ">>> ACME attempt: $ACME_DOMAIN via $ACME_DNS_PROVIDER (bounded to ${attempt_timeout}s)"
  local -a lego_args=(--accept-tos
    --email "${ACME_EMAIL:-admin@$ACME_DOMAIN}"
    --dns "$ACME_DNS_PROVIDER" --domains "$ACME_DOMAIN")
  if (( ${#propagation[@]} )); then
    lego_args+=("${propagation[@]}")
  fi
  lego_args+=(--path "$lego_path" run)
  timeout --signal=TERM --kill-after=10 "$attempt_timeout" \
    lego "${lego_args[@]}" && rc=0
  [[ -z "$guard_pid" ]] ||
    { kill "$guard_pid" 2>/dev/null || true; wait "$guard_pid" 2>/dev/null || true; }
  cleanup_challenge
  return "$rc"
}

if [[ "$mode" == --once ]]; then
  attempt
  exit $?
fi

# Exponential backoff with a ceiling and jitter. A persistent failure (bad DNS,
# firewalled validation, wrong creds) otherwise re-orders every 300s forever,
# pinning the account against Let's Encrypt's 5-failed-validations-per-hostname-
# per-hour limit — the same rate-limit hazard the entrypoint guards elsewhere by
# persisting the DNS suffix and reusing valid certs. An optional attempt cap
# stops entirely after N consecutive failures.
max_delay="${ACME_BACKGROUND_RETRY_MAX_S:-3600}"
max_attempts="${ACME_BACKGROUND_MAX_ATTEMPTS:-0}"   # 0 = unlimited
delay="$retry_delay"
attempts=0
while ! cert_valid; do
  if attempt; then
    echo ">>> ACME background retry succeeded for $ACME_DOMAIN."
    echo ">>> Certificate is persisted; restart/apply once to enable endpoint TLS."
    touch "$lego_path/restart-required"
    exit 0
  fi
  attempts=$((attempts + 1))
  if [[ "$max_attempts" -gt 0 && "$attempts" -ge "$max_attempts" ]]; then
    echo "!!! ACME background retry gave up after $attempts attempts; issue TLS manually or relaunch the instance." >&2
    exit 1
  fi
  jitter=$(( (RANDOM % 30) + 1 ))
  echo "!!! ACME background retry failed (attempt $attempts); retrying in $((delay + jitter))s without blocking serving"
  sleep "$((delay + jitter))"
  delay=$(( delay * 2 ))
  [[ "$delay" -le "$max_delay" ]] || delay="$max_delay"
done
