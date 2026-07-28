#!/usr/bin/env python3
"""Unit tests for the deSEC DNS-01 authoritative convergence guard."""

import importlib.util
import json
import os
import sys
import types
import unittest
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CI tests the pure decision and API payload without installing dnspython.
# The container pins dnspython 2.8.0 and the live Vast qualification exercises
# the actual authoritative queries.
dns_module = types.ModuleType("dns")
dns_module.__path__ = []
resolver_module = types.ModuleType("dns.resolver")
for name in ("NXDOMAIN", "NoAnswer", "NoNameservers", "LifetimeTimeout"):
    setattr(resolver_module, name, type(name, (Exception,), {}))
resolver_module.Resolver = object
dns_module.resolver = resolver_module
sys.modules.setdefault("dns", dns_module)
sys.modules.setdefault("dns.resolver", resolver_module)

spec = importlib.util.spec_from_file_location(
    "desec_acme_guard",
    os.path.join(REPO, "scripts", "desec_acme_guard.py"),
)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class DecisionTests(unittest.TestCase):
    def test_only_republishes_an_observed_authoritative_split(self):
        self.assertFalse(guard.needs_republish({"ns1": set(), "ns2": set()}))
        self.assertFalse(guard.needs_republish(
            {"ns1": {"challenge"}, "ns2": {"challenge"}}))
        self.assertTrue(guard.needs_republish(
            {"ns1": set(), "ns2": {"challenge"}}))
        self.assertTrue(guard.needs_republish(
            {"ns1": {"old"}, "ns2": {"new"}}))

    def test_republish_uses_idempotent_bulk_put_and_quoted_txt_rdata(self):
        response = mock.MagicMock()
        response.status = 204
        response.__enter__.return_value = response
        with mock.patch.object(guard.urllib.request, "urlopen",
                               return_value=response) as urlopen:
            guard.republish(
                "example.dedyn.io",
                "_acme-challenge.model-123",
                {"second", "first"},
                "not-logged",
            )

        request = urlopen.call_args.args[0]
        self.assertEqual(request.method, "PUT")
        self.assertEqual(
            request.full_url,
            "https://desec.io/api/v1/domains/example.dedyn.io/rrsets/",
        )
        payload = json.loads(request.data)
        self.assertEqual(payload[0]["subname"], "_acme-challenge.model-123")
        self.assertEqual(payload[0]["type"], "TXT")
        self.assertEqual(payload[0]["records"], ['"first"', '"second"'])
        self.assertEqual(
            request.headers["Authorization"], "Token not-logged")


if __name__ == "__main__":
    unittest.main(verbosity=2)
