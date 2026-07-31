#!/usr/bin/env python3
"""Repair a transient deSEC authoritative split during lego DNS-01 issuance.

lego creates the challenge RRset and then waits for public recursive resolvers.
In one live Vast qualification, only one of deSEC's two authoritative servers
received that first write; the API also returned 404 for the RRset. Repeating
the same idempotent bulk PUT converged both authorities immediately.

This helper runs beside lego, discovers the public challenge value from the
authoritative servers, and republishes it only when those servers disagree.
It never prints the challenge value or the API token.
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request

import dns.exception
import dns.resolver


def _txt_values(answer) -> set[str]:
    values = set()
    for record in answer:
        # dnspython already separates multi-string TXT chunks.
        values.add(b"".join(record.strings).decode())
    return values


def authoritative_txt(zone: str, name: str) -> dict[str, set[str]]:
    default = dns.resolver.Resolver()
    nameservers = [str(r.target).rstrip(".") for r in default.resolve(zone, "NS")]
    observed = {}
    for server in nameservers:
        addresses = [str(r) for r in default.resolve(server, "A")]
        resolver = dns.resolver.Resolver(configure=False)
        resolver.nameservers = addresses
        resolver.timeout = 2
        resolver.lifetime = 4
        try:
            observed[server] = _txt_values(resolver.resolve(name, "TXT"))
        except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer,
                dns.resolver.NoNameservers, dns.resolver.LifetimeTimeout):
            observed[server] = set()
    return observed


def needs_republish(observed: dict[str, set[str]]) -> bool:
    nonempty = [values for values in observed.values() if values]
    return bool(nonempty) and any(values != nonempty[0]
                                  for values in observed.values())


def republish(zone: str, subname: str, values: set[str], token: str) -> None:
    # deSEC represents TXT RDATA as a quoted presentation-format string.
    records = [json.dumps(value) for value in sorted(values)]
    body = json.dumps([{
        "subname": subname,
        "type": "TXT",
        "ttl": 3600,
        "records": records,
    }]).encode()
    request = urllib.request.Request(
        f"https://desec.io/api/v1/domains/{zone}/rrsets/",
        data=body,
        method="PUT",
        headers={
            "Authorization": f"Token {token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        if response.status not in (200, 201, 204):
            raise RuntimeError(f"unexpected deSEC HTTP {response.status}")


def guard(zone: str, fqdn: str, token: str, timeout: int) -> int:
    rel = fqdn.removesuffix("." + zone)
    if rel == zone:
        # Apex issuance: the challenge is _acme-challenge.<zone> itself, with
        # an empty relative part — the naive f-string doubled the zone and
        # made the guard poll a name that never exists.
        subname = "_acme-challenge"
    else:
        subname = f"_acme-challenge.{rel}"
    challenge = f"{subname}.{zone}"
    deadline = time.monotonic() + timeout
    repaired = False
    while time.monotonic() < deadline:
        try:
            observed = authoritative_txt(zone, challenge)
            all_values = set().union(*observed.values()) if observed else set()
            if needs_republish(observed):
                republish(zone, subname, all_values, token)
                repaired = True
                print(">>> deSEC ACME guard: authoritative TXT split detected; "
                      "re-published the same challenge RRset", flush=True)
            elif all_values and len(set(map(frozenset, observed.values()))) == 1:
                if repaired:
                    print(">>> deSEC ACME guard: authoritative TXT servers converged",
                          flush=True)
                return 0
        except (OSError, RuntimeError, urllib.error.URLError,
                dns.exception.DNSException) as exc:
            # DNSException covers the NS/A discovery lookups too: one resolver
            # timeout must degrade to a retry, not kill the guard for the rest
            # of the lego attempt.
            print(f"!!! deSEC ACME guard: transient check failed: "
                  f"{type(exc).__name__}", flush=True)
        time.sleep(3)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    token = os.environ.get("DESEC_TOKEN", "")
    if not token:
        return 0
    return guard(args.zone.rstrip("."), args.domain.rstrip("."), token,
                 max(10, args.timeout))


if __name__ == "__main__":
    raise SystemExit(main())
