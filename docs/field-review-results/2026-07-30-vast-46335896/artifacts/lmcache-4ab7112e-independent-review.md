# LMCache `4ab7112e` independent review

Verdict: **REQUEST CHANGES**

- Candidate: `4ab7112e2a442e05928ba4c78e4cf09076b419b9`
- Exact r14 composition base: `0056361275815f7b283366d993a9fa8b069ecd8f`
- Review was read-only; no source, branch, PR or issue was mutated.

## Release blockers

1. `process_outbound_task` stores a caught exception object in its future.
   The traceback retains the outbound frame, client and request payload; raising
   the same stored exception from `MessagingFuture.result()` also attaches the
   caller frame and caller locals. The successor must store traceback-free
   templates and raise fresh exceptions, preserving the fields of
   `RemoteHandlerError`.
2. Requests without CUDA IPC exports still receive an empty
   `_IPCTransportOwnership`. `_fail_outstanding()` permanently quarantines the
   empty records. A deterministic 1,000-NOOP reproduction produced 1,000 empty
   quarantine entries. Export ownership tracking must be conditional on a
   nonempty export snapshot; event-resource quarantine is separate.
3. `MessageQueueServer.close()` shuts down both worker pools with
   `wait=False`, then closes the notifier/socket. A blocked-handler reproduction
   reached the closed notifier twice with `EBADF`. Intake must stop and queued
   or active handlers must drain/cancel under a bounded policy before transport
   and cache-engine teardown. The successor must also integrate upstream
   `b20e6151c9ff54d77bf48b4eb2fc79512ac06bc4` (PR #4197), which adds the
   missing `engine.close()` and SIGTERM hook.
4. Response UID/type decoding occurs outside malformed-body containment. A
   valid pending UID plus invalid `RequestType` left the future incomplete and
   retained in both pending and inflight maps. A recoverable UID must be
   terminalized; an unparseable/truncated header must retire the session and
   quarantine all ambiguous sent work.

## Accepted parts of the design

The reviewed one-export/one-import state machine is sound on the exercised
paths. Constructor and partial-batch rollback, immutable export snapshots,
async-forward leases, strict transfer validation, malformed-body completion,
recovery-NOOP release and unknown-request daemon survival are meaningful
repairs. Exact-SHA GPU gates passed 172/172 cold and 172/172 warm with clean
warning gates. A native-GPU registration plus three STORE/RETRIEVE cycles
matched checksums and explicitly unregistered without warnings.

## Successor gates

- deterministic CPU regressions for all four blockers;
- real CUDA initial REGISTER, fresh same-ID re-REGISTER, STORE/RETRIEVE and
  UNREGISTER with warning/reference checks;
- full vLLM + LMCache shutdown while registration and active/queued operations
  remain live; and
- fresh-cache cold/warm GPU union and native round trip, with no warning,
  residual process or retained CUDA context.
