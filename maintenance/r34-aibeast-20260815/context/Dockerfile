FROM docker.io/voipmonitor/vllm@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b

LABEL org.opencontainers.image.title="AIBeast GLM-5.2 r34 read-lease candidate v2" \
      org.opencontainers.image.description="Exact GG v20 r34 plus vLLM PR277 compatibility, hybrid invalid-block recovery, bounded LMCache MP retrieve, expired read-lease recovery, and bounded L2 LRU" \
      org.opencontainers.image.revision="vllm-a25483af35-lmcache-67561538" \
      ai.malaiwah.base="voipmonitor/vllm@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b" \
      ai.malaiwah.vllm.source="a25483af35" \
      ai.malaiwah.lmcache.source="67561538"

# Fail closed if the registry tag ever resolves to bytes other than the exact
# r34 source composition qualified for this maintenance window.
RUN printf '%s  %s\n' \
      42c0e9150a065c48e3780eebc8b3c89ea410d82610e2c14bc97546dee6866214 /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/exl3.py \
      d540f800e948d4e94e72fb19e91ebb37f163e37a2a64d07051dfd7b5d87dfe2f /opt/venv/lib/python3.12/site-packages/vllm/v1/core/kv_cache_manager.py \
      1ea341f4cc28d282452597c25d97eea84be8b5f984d2e1a6b548356c8417fdce /opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py \
      ea580badefb9a0fad5fa2ac1bdcff4f2b36fc85138d95a5be4b180f07dd2c874 /opt/venv/lib/python3.12/site-packages/lmcache/integration/vllm/vllm_multi_process_adapter.py \
    | sha256sum -c -

COPY requirements-soul.lock /opt/requirements-soul.lock
RUN apt-get update -qq \
 && apt-get install -y -qq nvtop htop openssh-server socat python3-venv util-linux patch \
 && rm -rf /var/lib/apt/lists/* \
 && rm -f /etc/ssh/ssh_host_* \
 && curl -sSL -o /tmp/lego.tgz https://github.com/go-acme/lego/releases/download/v4.35.2/lego_v4.35.2_linux_amd64.tar.gz \
 && echo "ee5be4bf457de8e3efa86a51651c75c87f0ee0e4e9f3ae14f6034d68365770f3  /tmp/lego.tgz" | sha256sum -c - \
 && tar xzf /tmp/lego.tgz -C /usr/local/bin lego \
 && rm /tmp/lego.tgz \
 && chmod +x /usr/local/bin/lego \
 && python3 -m venv /opt/nanobot-venv \
 && /opt/nanobot-venv/bin/pip install --no-cache-dir --require-hashes -r /opt/requirements-soul.lock \
 && /opt/nanobot-venv/bin/python -c "from nanobot import Nanobot; import importlib.metadata as m; assert m.version('nanobot-ai') == '0.3.0'" \
 && { getent group soul >/dev/null || groupadd --system soul; } \
 && { id -u soul >/dev/null 2>&1 || useradd --system --gid soul --home-dir /nonexistent --shell /bin/bash soul; }

# Preserve the exact deployed turnkey control plane; only the wrapper below is
# intentionally changed to pass the retrieve deadline and per-adapter L2 LRU.
COPY assets/etc/ssh/sshd_config.d/99-model-turnkey.conf /etc/ssh/sshd_config.d/99-model-turnkey.conf
COPY assets/opt/landing.py /opt/landing.py
COPY assets/opt/scripts/ /opt/scripts/
COPY assets/opt/soul/ /opt/soul/
COPY assets/usr/local/bin/model-turnkey-entry.sh /usr/local/bin/model-turnkey-entry.sh
COPY assets/opt/scripts/glm52_lmcache_wrapper.sh /usr/local/bin/glm52-lmcache-wrapper.sh

# Backport the reviewed expired-read-lease recovery onto the exact r34 LMCache
# composition. This only changes the next immutable image build; it never
# mutates the running container.
COPY patches/0001-fix-mp-recover-expired-L1-read-leases.patch /tmp/lmcache-read-lease-recovery.patch
RUN patch --batch --forward -p1 -d /opt/venv/lib/python3.12/site-packages \
      < /tmp/lmcache-read-lease-recovery.patch \
 && rm /tmp/lmcache-read-lease-recovery.patch

# Install the reviewed runtime changes into both the importable environment and
# the source tree used by r34's tests/debug tooling.
COPY runtime/vllm/vllm/model_executor/layers/quantization/exl3.py /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/exl3.py
COPY runtime/vllm/vllm/model_executor/layers/quantization/exl3.py /opt/vllm/vllm/model_executor/layers/quantization/exl3.py
COPY runtime/vllm/vllm/v1/core/kv_cache_manager.py /opt/venv/lib/python3.12/site-packages/vllm/v1/core/kv_cache_manager.py
COPY runtime/vllm/vllm/v1/core/kv_cache_manager.py /opt/vllm/vllm/v1/core/kv_cache_manager.py
COPY runtime/vllm/vllm/v1/core/sched/scheduler.py /opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py
COPY runtime/vllm/vllm/v1/core/sched/scheduler.py /opt/vllm/vllm/v1/core/sched/scheduler.py
COPY runtime/lmcache/lmcache/integration/vllm/vllm_multi_process_adapter.py /opt/venv/lib/python3.12/site-packages/lmcache/integration/vllm/vllm_multi_process_adapter.py

RUN printf '%s  %s\n' \
      78a732362077a715228ea096eb07fc9c074ddb712fc5e76e4f289bf4244f4918 /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/exl3.py \
      78a732362077a715228ea096eb07fc9c074ddb712fc5e76e4f289bf4244f4918 /opt/vllm/vllm/model_executor/layers/quantization/exl3.py \
      e7b8bb464c5f4741f482e03bb28148cbec9d1d4438d8d70774eb95bf5f9cca78 /opt/venv/lib/python3.12/site-packages/vllm/v1/core/kv_cache_manager.py \
      e7b8bb464c5f4741f482e03bb28148cbec9d1d4438d8d70774eb95bf5f9cca78 /opt/vllm/vllm/v1/core/kv_cache_manager.py \
      23a0f2ce9dbf2c36f5f240b3aa45d2f25cc31329bd0ac3cfd50847f8a0bd74d6 /opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py \
      23a0f2ce9dbf2c36f5f240b3aa45d2f25cc31329bd0ac3cfd50847f8a0bd74d6 /opt/vllm/vllm/v1/core/sched/scheduler.py \
      0781f930e304992c75ebf596030b6a3f6d0bf697558de14168168531ee51fe21 /opt/venv/lib/python3.12/site-packages/lmcache/integration/vllm/vllm_multi_process_adapter.py \
      58cc7f828c2e65edbfe0d4720a849c931f9b217274e1dad511fcf2aa480aaa3a /opt/venv/lib/python3.12/site-packages/lmcache/cli/commands/trace/_dispatch.py \
      bc98d1db99236847b5b5f88dc77c1a105dc796d206cf3d51e7940be642836e57 /opt/venv/lib/python3.12/site-packages/lmcache/v1/distributed/l1_manager.py \
      9472313897a53bf1c7af68c278921222e01cdfbf5e55198c0022bdf42e22872c /opt/venv/lib/python3.12/site-packages/lmcache/v1/distributed/storage_manager.py \
      04ef2d9e1be5f986c1a93b0ab5acc7c87e47655eb9a239ca5d185ef809b1bed5 /opt/venv/lib/python3.12/site-packages/lmcache/v1/multiprocess/config.py \
      89cfd0dc03e0e38fe6d29fc6b2775710d87fce46e800cfad6fe9f6b4b7f2cef0 /opt/venv/lib/python3.12/site-packages/lmcache/v1/multiprocess/engine_context.py \
      d62c23985d0b1f38ff9b7429b17b29931188512ca1c4e5c0f8e65325d2295e71 /opt/venv/lib/python3.12/site-packages/lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py \
      ef51c00cd34a2ccc0840b4916e3c3d2d3e651eb5907200847e67972a67b8edf5 /opt/venv/lib/python3.12/site-packages/lmcache/v1/multiprocess/modules/lookup.py \
      ee300edb1076b656f6148f362e0cdac6f836e7d48262bf143ac64338905a07c3 /opt/venv/lib/python3.12/site-packages/lmcache/v1/multiprocess/server.py \
      3e4f1f9e9e80ffd27dd7c8f00da5194897ea89e4227c6d18fe888044398793df /opt/venv/lib/python3.12/site-packages/lmcache/v1/multiprocess/session.py \
    | sha256sum -c - \
 && chmod +x /usr/local/bin/model-turnkey-entry.sh /usr/local/bin/glm52-lmcache-wrapper.sh \
      /opt/scripts/soul_controller.py /opt/scripts/soul_config.py \
      /opt/scripts/glm52_lmcache_wrapper.sh /opt/scripts/acme_retry.sh \
 && chmod -R a-w /opt/soul \
 && ln -sf model-turnkey-entry.sh /usr/local/bin/glm52-entry.sh

EXPOSE 22 8000 8443 1111
ENTRYPOINT ["/usr/local/bin/model-turnkey-entry.sh"]
