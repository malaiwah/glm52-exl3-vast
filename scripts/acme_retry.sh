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

cleanup_challenge() {
  [[ "$ACME_DNS_PROVIDER" == desec && -n "${DESEC_DOMAIN:-}" &&
     -n "${DESEC_TOKEN:-}" ]] || return 0
  case "$ACME_DOMAIN" in
    *."$DESEC_DOMAIN")
      local sub="${ACME_DOMAIN%."$DESEC_DOMAIN"}"
      curl -sf --max-time 15 -X DELETE \
        "https://desec.io/api/v1/domains/${DESEC_DOMAIN}/rrsets/_acme-challenge.${sub}/TXT/" \
        -H "Authorization: Token ${DESEC_TOKEN}" >/dev/null 2>&1 || true
      ;;
  esac
}

register_desec() {
  [[ "$ACME_DNS_PROVIDER" == desec && -n "${DESEC_DOMAIN:-}" &&
     -n "${DESEC_TOKEN:-}" ]] || return 0
  local sub="${ACME_DOMAIN%."$DESEC_DOMAIN"}"
  local ip="${ACME_PUBLIC_IP:-${PUBLIC_IPADDR:-${RUNPOD_PUBLIC_IP:-}}}"
  [[ -n "$ip" ]] || {
    echo "!!! ACME retry: public IP is not available for deSEC registration"
    return 1
  }
  curl -sf --max-time 20 -X PUT \
    "https://desec.io/api/v1/domains/${DESEC_DOMAIN}/rrsets/" \
    -H "Authorization: Token ${DESEC_TOKEN}" -H "Content-Type: application/json" \
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
  timeout --signal=TERM --kill-after=10 "$attempt_timeout" \
    lego --accept-tos --email "${ACME_EMAIL:-admin@$ACME_DOMAIN}" \
      --dns "$ACME_DNS_PROVIDER" --domains "$ACME_DOMAIN" \
      "${propagation[@]}" --path "$lego_path" run && rc=0
  [[ -z "$guard_pid" ]] ||
    { kill "$guard_pid" 2>/dev/null || true; wait "$guard_pid" 2>/dev/null || true; }
  cleanup_challenge
  return "$rc"
}

if [[ "$mode" == --once ]]; then
  attempt
  exit $?
fi

while ! cert_valid; do
  if attempt; then
    echo ">>> ACME background retry succeeded for $ACME_DOMAIN."
    echo ">>> Certificate is persisted; restart/apply once to enable endpoint TLS."
    touch "$lego_path/restart-required"
    exit 0
  fi
  echo "!!! ACME background retry failed; retrying in ${retry_delay}s without blocking serving"
  sleep "$retry_delay"
done
