# Full GLM-5.3 release candidate on the local-inference-lab Gilded Gnosis base.
# Lineage: GG v20 r34 -> reviewed r34 maintenance layer -> release overlays.
# Every parent state and every installed file is SHA-256 pinned; a mixed or
# unknown state is rejected. 500K+ GPU qualification precedes promotion.
ARG BASE_IMAGE=docker.io/voipmonitor/vllm@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
ARG SOURCE_REVISION=uncommitted
USER root
COPY --from=ghcr.io/astral-sh/uv@sha256:e85be844203885286c60ffad8a858d48afb6c5a5c237ca0e67f12e74b8f174b1 /uv /usr/local/bin/uv
LABEL org.opencontainers.image.title="Multi-model vLLM turnkey for Vast.ai, Runpod, and JarvisLabs" \
      org.opencontainers.image.description="Full GLM-5.3 EXL3 3.42bpw 520K reduced-workspace candidate on Gilded Gnosis v20 r34 with bounded LMCache, parser fixes, and isolated maintenance preparation." \
      ai.malaiwah.evidence="GLM-5.3-full-500K-candidate GPU-qualification-required" \
      ai.malaiwah.base="voipmonitor/vllm@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b" \
      ai.malaiwah.base.name="local-inference-lab Gilded Gnosis v20 r34" \
      ai.malaiwah.parent.license="NOASSERTION" \
      ai.malaiwah.parent.license.note="Gilded Gnosis v20 r34 aggregates Apache-2.0 vLLM and LMCache with B12X; local-inference-lab/blackwell-llm-docker publishes no repository LICENSE file" \
      ai.malaiwah.vllm.source="4d006a43928cdee01306691a766542c1e9bebb59" \
      ai.malaiwah.b12x.source="cd3ce190f0f1917402cdfd5773724267cc9a63f8" \
      ai.malaiwah.lmcache.source="67561538a08ce3db621f51f0615b67537c0c8361"
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_INDEX_STRATEGY=first-index
# Build-time only: the serving environment must not gain a new runtime variable.
ARG SITE_PACKAGES=/opt/venv/lib/python3.12/site-packages
COPY requirements-soul.lock /opt/requirements-soul.lock
# Fail closed if the base tag or digest ever resolves to bytes other than the
# exact Gilded Gnosis r34 source composition this release was reviewed against.
RUN set -eux; \
    printf '%s  %s\n' \
      42c0e9150a065c48e3780eebc8b3c89ea410d82610e2c14bc97546dee6866214 "$SITE_PACKAGES/vllm/model_executor/layers/quantization/exl3.py" \
      42c0e9150a065c48e3780eebc8b3c89ea410d82610e2c14bc97546dee6866214 /opt/vllm/vllm/model_executor/layers/quantization/exl3.py \
      d540f800e948d4e94e72fb19e91ebb37f163e37a2a64d07051dfd7b5d87dfe2f "$SITE_PACKAGES/vllm/v1/core/kv_cache_manager.py" \
      d540f800e948d4e94e72fb19e91ebb37f163e37a2a64d07051dfd7b5d87dfe2f /opt/vllm/vllm/v1/core/kv_cache_manager.py \
      1ea341f4cc28d282452597c25d97eea84be8b5f984d2e1a6b548356c8417fdce "$SITE_PACKAGES/vllm/v1/core/sched/scheduler.py" \
      1ea341f4cc28d282452597c25d97eea84be8b5f984d2e1a6b548356c8417fdce /opt/vllm/vllm/v1/core/sched/scheduler.py \
      ea580badefb9a0fad5fa2ac1bdcff4f2b36fc85138d95a5be4b180f07dd2c874 "$SITE_PACKAGES/lmcache/integration/vllm/vllm_multi_process_adapter.py" \
    | sha256sum -c -; \
    test -d "$SITE_PACKAGES/b12x"; \
    test -d "$SITE_PACKAGES/tilelang"; \
    test -d /opt/vllm/vllm/parser; \
    test -f /opt/exllamav3-python/exllamav3/modules/quant/exl3_lib/quantize.py; \
    test -f /opt/libnccl-local-inference.so.2.30.4; \
    apt-get update -qq; \
    apt-get install -y -qq nvtop htop curl openssh-server socat python3-venv util-linux patch; \
    rm -rf /var/lib/apt/lists/* /etc/ssh/ssh_host_*; \
    mkdir -p /opt/vllm/kv-scales; \
    curl --fail --show-error --location --retry 4 --retry-all-errors --connect-timeout 15 --max-time 180 -o /opt/vllm/kv-scales/glm52-nvfp4-nf3-hybrid_mla_outer_scales_v1.json \
      "https://huggingface.co/madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid/resolve/2eb778b8ac3203f31a7dbe6d9f1bc9ba8fb00c25/kv-scales/glm52-nvfp4-nf3-hybrid_mla_outer_scales_v1.json?download=true"; \
    echo "ac68fe6af3056ec35299361293c9ae568769d21696756548493f67ff17881ece  /opt/vllm/kv-scales/glm52-nvfp4-nf3-hybrid_mla_outer_scales_v1.json" | sha256sum -c -; \
    curl --fail --show-error --location --retry 4 --retry-all-errors --connect-timeout 15 --max-time 180 -o /tmp/lego.tgz https://github.com/go-acme/lego/releases/download/v4.35.2/lego_v4.35.2_linux_amd64.tar.gz; \
    echo "ee5be4bf457de8e3efa86a51651c75c87f0ee0e4e9f3ae14f6034d68365770f3  /tmp/lego.tgz" | sha256sum -c -; \
    tar xzf /tmp/lego.tgz -C /usr/local/bin lego; \
    rm /tmp/lego.tgz; \
    chmod +x /usr/local/bin/lego; \
    python3 -c 'import dns, hf_xet, huggingface_hub; assert dns.__version__ == "2.8.0"'; \
    python3 -c 'import importlib.metadata as m; print(*sorted((d.metadata["Name"] or "") + "==" + d.version for d in m.distributions()), sep="\n")' > /tmp/vllm-packages.before; \
    uv venv --no-managed-python --python /opt/venv/bin/python /opt/nanobot-venv; \
    env -u PYTHONPATH uv pip install --python /opt/nanobot-venv/bin/python --no-cache --require-hashes -r /opt/requirements-soul.lock; \
    env -u PYTHONPATH /opt/nanobot-venv/bin/python -c "from nanobot import Nanobot; import importlib.metadata as m; assert m.version('nanobot-ai') == '0.3.0'"; \
    python3 -c 'import importlib.metadata as m; print(*sorted((d.metadata["Name"] or "") + "==" + d.version for d in m.distributions()), sep="\n")' > /tmp/vllm-packages.after; \
    diff -u /tmp/vllm-packages.before /tmp/vllm-packages.after; \
    rm /tmp/vllm-packages.before /tmp/vllm-packages.after; \
    getent group soul >/dev/null || groupadd --system soul; \
    getent passwd soul >/dev/null || useradd --system --gid soul --home-dir /nonexistent --shell /bin/bash soul
# Reviewed r34 maintenance layer: expired L1 read-lease recovery with
# configurable request-session retention (LMCache/LMCache#4691 draft), then the
# vLLM PR277 compatibility, hybrid invalid-block recovery and bounded LMCache
# multiprocess retrieve sources already qualified in the running appliance.
COPY maintenance/r34-aibeast-20260815/context/patches/0001-fix-mp-recover-expired-L1-read-leases.patch /tmp/lmcache-read-lease-recovery.patch
COPY maintenance/r34-aibeast-20260815/context/runtime/vllm/vllm/model_executor/layers/quantization/exl3.py /tmp/maintenance/exl3.py
COPY maintenance/r34-aibeast-20260815/context/runtime/vllm/vllm/v1/core/kv_cache_manager.py /tmp/maintenance/kv_cache_manager.py
COPY maintenance/r34-aibeast-20260815/context/runtime/vllm/vllm/v1/core/sched/scheduler.py /tmp/maintenance/scheduler.py
COPY maintenance/r34-aibeast-20260815/context/runtime/lmcache/lmcache/integration/vllm/vllm_multi_process_adapter.py /tmp/maintenance/vllm_multi_process_adapter.py
RUN set -eux; \
    patch --batch --forward -p1 -d "$SITE_PACKAGES" < /tmp/lmcache-read-lease-recovery.patch; \
    rm /tmp/lmcache-read-lease-recovery.patch; \
    install -m 0644 /tmp/maintenance/exl3.py "$SITE_PACKAGES/vllm/model_executor/layers/quantization/exl3.py"; \
    install -m 0644 /tmp/maintenance/exl3.py /opt/vllm/vllm/model_executor/layers/quantization/exl3.py; \
    install -m 0644 /tmp/maintenance/kv_cache_manager.py "$SITE_PACKAGES/vllm/v1/core/kv_cache_manager.py"; \
    install -m 0644 /tmp/maintenance/kv_cache_manager.py /opt/vllm/vllm/v1/core/kv_cache_manager.py; \
    install -m 0644 /tmp/maintenance/scheduler.py "$SITE_PACKAGES/vllm/v1/core/sched/scheduler.py"; \
    install -m 0644 /tmp/maintenance/scheduler.py /opt/vllm/vllm/v1/core/sched/scheduler.py; \
    install -m 0644 /tmp/maintenance/vllm_multi_process_adapter.py "$SITE_PACKAGES/lmcache/integration/vllm/vllm_multi_process_adapter.py"; \
    rm -r /tmp/maintenance; \
    printf '%s  %s\n' \
      78a732362077a715228ea096eb07fc9c074ddb712fc5e76e4f289bf4244f4918 "$SITE_PACKAGES/vllm/model_executor/layers/quantization/exl3.py" \
      78a732362077a715228ea096eb07fc9c074ddb712fc5e76e4f289bf4244f4918 /opt/vllm/vllm/model_executor/layers/quantization/exl3.py \
      e7b8bb464c5f4741f482e03bb28148cbec9d1d4438d8d70774eb95bf5f9cca78 "$SITE_PACKAGES/vllm/v1/core/kv_cache_manager.py" \
      e7b8bb464c5f4741f482e03bb28148cbec9d1d4438d8d70774eb95bf5f9cca78 /opt/vllm/vllm/v1/core/kv_cache_manager.py \
      23a0f2ce9dbf2c36f5f240b3aa45d2f25cc31329bd0ac3cfd50847f8a0bd74d6 "$SITE_PACKAGES/vllm/v1/core/sched/scheduler.py" \
      23a0f2ce9dbf2c36f5f240b3aa45d2f25cc31329bd0ac3cfd50847f8a0bd74d6 /opt/vllm/vllm/v1/core/sched/scheduler.py \
      0781f930e304992c75ebf596030b6a3f6d0bf697558de14168168531ee51fe21 "$SITE_PACKAGES/lmcache/integration/vllm/vllm_multi_process_adapter.py" \
      58cc7f828c2e65edbfe0d4720a849c931f9b217274e1dad511fcf2aa480aaa3a "$SITE_PACKAGES/lmcache/cli/commands/trace/_dispatch.py" \
      bc98d1db99236847b5b5f88dc77c1a105dc796d206cf3d51e7940be642836e57 "$SITE_PACKAGES/lmcache/v1/distributed/l1_manager.py" \
      9472313897a53bf1c7af68c278921222e01cdfbf5e55198c0022bdf42e22872c "$SITE_PACKAGES/lmcache/v1/distributed/storage_manager.py" \
      04ef2d9e1be5f986c1a93b0ab5acc7c87e47655eb9a239ca5d185ef809b1bed5 "$SITE_PACKAGES/lmcache/v1/multiprocess/config.py" \
      89cfd0dc03e0e38fe6d29fc6b2775710d87fce46e800cfad6fe9f6b4b7f2cef0 "$SITE_PACKAGES/lmcache/v1/multiprocess/engine_context.py" \
      d62c23985d0b1f38ff9b7429b17b29931188512ca1c4e5c0f8e65325d2295e71 "$SITE_PACKAGES/lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py" \
      ef51c00cd34a2ccc0840b4916e3c3d2d3e651eb5907200847e67972a67b8edf5 "$SITE_PACKAGES/lmcache/v1/multiprocess/modules/lookup.py" \
      ee300edb1076b656f6148f362e0cdac6f836e7d48262bf143ac64338905a07c3 "$SITE_PACKAGES/lmcache/v1/multiprocess/server.py" \
      3e4f1f9e9e80ffd27dd7c8f00da5194897ea89e4227c6d18fe888044398793df "$SITE_PACKAGES/lmcache/v1/multiprocess/session.py" \
    | sha256sum -c -
COPY sshd_config /etc/ssh/sshd_config.d/99-model-turnkey.conf
COPY landing.py /opt/landing.py
COPY scripts/ /opt/scripts/
COPY patches/glm53-refresh/ /opt/glm53-refresh/
# Narrow selected PR59 runtime sources only; the historical experiment and its
# launcher/MTP prototype are not part of the production overlay.
COPY patches/glm53-selected/ /opt/glm53-selected/
COPY patches/scopedlmcache/ /opt/scopedlmcache/
COPY soul/ /opt/soul/
COPY entrypoint.sh /usr/local/bin/model-turnkey-entry.sh
# The public Vast template may still call glm52-entry.sh from its onstart field.
# The selected scheduler/core baseline is the refresh AFTER state. Keep both
# installers ordered here, and write provenance only after all final overlays.
RUN set -eux; \
    python3 /opt/scripts/patch_lmcache_admin_api.py; \
    python3 /opt/scripts/patch_lmcache_admin_api.py --verify-only; \
    python3 -m py_compile /opt/glm53-refresh/*.py /opt/scopedlmcache/*.py; \
    python3 /opt/scripts/apply_glm53_refresh.py /opt/glm53-refresh; \
    python3 /opt/scripts/apply_glm53_refresh.py /opt/glm53-refresh --verify-only; \
    python3 -m compileall -q /opt/glm53-selected; \
    python3 /opt/scripts/apply_glm53_selected.py /opt/glm53-selected --verify-only; \
    python3 /opt/scripts/apply_glm53_selected.py /opt/glm53-selected; \
    python3 /opt/scripts/apply_glm53_selected.py /opt/glm53-selected --verify-only; \
    python3 /opt/scripts/patch_scopedlmcache_retrieve.py; \
    python3 /opt/scripts/patch_scopedlmcache_retrieve.py --verify-only; \
    python3 /opt/scripts/patch_scopedlmcache_retrieve.py --layout; \
    python3 /opt/scripts/patch_scopedlmcache_retrieve.py --layout --verify-only; \
    python3 /opt/scripts/write_runtime_provenance.py \
      --parent-image "$BASE_IMAGE" --source-revision "$SOURCE_REVISION" \
      --output /opt/runtime-provenance.json; \
    chmod +x /usr/local/bin/model-turnkey-entry.sh /opt/scripts/soul_launcher.py \
      /opt/scripts/soul_controller.py /opt/scripts/soul_config.py \
      /opt/scripts/glm52_lmcache_wrapper.sh /opt/scripts/acme_retry.sh; \
    chmod -R a-w /opt/soul /opt/glm53-refresh /opt/glm53-selected /opt/scopedlmcache; \
    ln -sf model-turnkey-entry.sh /usr/local/bin/glm52-entry.sh
EXPOSE 22 8000 8443 1111
ENTRYPOINT ["/usr/local/bin/model-turnkey-entry.sh"]
