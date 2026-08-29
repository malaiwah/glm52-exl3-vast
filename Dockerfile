# GLM-5.3-Flash K6 production runtime. The immutable parent supplies the
# Glm5Next/vLLM/B12X/TR3/EXL3 stack validated on four RTX PRO 6000 Blackwell
# GPUs. The fail-closed overlay installer adds the exact live-qualified fixes;
# both the parent state and every resulting file are SHA-256 pinned.
FROM docker.io/verdictai/glm53-flash-exl3-k4@sha256:0f1cdcc8891f1cc3a444121eb61d366289a1cbba285f0892dcbb24bc94961692
USER root
LABEL org.opencontainers.image.title="Multi-model vLLM turnkey for Vast.ai, Runpod, and JarvisLabs" \
      org.opencontainers.image.description="Profile-driven OpenAI endpoint with a live-qualified GLM-5.3-Flash TR3 EXL3 K6 profile for four RTX PRO 6000 Blackwell GPUs." \
      ai.malaiwah.evidence="GLM-5.3-Flash-TR3-K6 TP4-DCP4 B12X-sparse-MLA Triton-MoE NVFP4-DS-MLA-KV" \
      ai.malaiwah.base="verdictai/glm53-flash-exl3-k4@sha256:0f1cdcc8891f1cc3a444121eb61d366289a1cbba285f0892dcbb24bc94961692"
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements-soul.lock /opt/requirements-soul.lock
RUN set -eux; \
    echo "f10b6ee1116d71c4b61c4603d38cb257b3f0dcfde9bcc0847839e48ac9baeb1d  /opt/glm53/calibration/glm53_nvfp4_mla_outer_scales_mtp_power2_v2.json" | sha256sum -c -; \
    test -f /opt/local-inference/nccl/lib/libnccl.so.2.31.2; \
    test -d /opt/infernal-invocation/vllm/vllm; \
    test -d /opt/infernal-invocation/b12x/b12x; \
    test -d /opt/exllamav3; \
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
    python3 -m venv /opt/nanobot-venv; \
    env -u PYTHONPATH /opt/nanobot-venv/bin/pip install --no-cache-dir --require-hashes -r /opt/requirements-soul.lock; \
    env -u PYTHONPATH /opt/nanobot-venv/bin/python -c "from nanobot import Nanobot; import importlib.metadata as m; assert m.version('nanobot-ai') == '0.3.0'"; \
    python3 -c 'import importlib.metadata as m; print(*sorted((d.metadata["Name"] or "") + "==" + d.version for d in m.distributions()), sep="\n")' > /tmp/vllm-packages.after; \
    diff -u /tmp/vllm-packages.before /tmp/vllm-packages.after; \
    rm /tmp/vllm-packages.before /tmp/vllm-packages.after; \
    getent group soul >/dev/null || groupadd --system soul; \
    getent passwd soul >/dev/null || useradd --system --gid soul --home-dir /nonexistent --shell /bin/bash soul
COPY sshd_config /etc/ssh/sshd_config.d/99-model-turnkey.conf
COPY landing.py /opt/landing.py
COPY scripts/ /opt/scripts/
COPY patches/glm53-runtime/ /opt/glm53-runtime/
COPY patches/field-review-r26/ledger.json /opt/field-review-r26-ledger.json
COPY soul/ /opt/soul/
COPY entrypoint.sh /usr/local/bin/model-turnkey-entry.sh
# The public Vast template may still call glm52-entry.sh from its onstart field.
RUN set -eux; \
    python3 /opt/scripts/apply_glm53_runtime_overlays.py /opt/glm53-runtime --verify-only; \
    python3 /opt/scripts/apply_glm53_runtime_overlays.py /opt/glm53-runtime; \
    python3 /opt/scripts/apply_glm53_runtime_overlays.py /opt/glm53-runtime --verify-only; \
    chmod +x /usr/local/bin/model-turnkey-entry.sh /opt/scripts/soul_controller.py /opt/scripts/soul_config.py \
      /opt/scripts/glm52_lmcache_wrapper.sh /opt/scripts/acme_retry.sh; \
    chmod -R a-w /opt/soul /opt/glm53-runtime; \
    ln -sf model-turnkey-entry.sh /usr/local/bin/glm52-entry.sh
EXPOSE 22 8000 8443 1111
ENTRYPOINT ["/usr/local/bin/model-turnkey-entry.sh"]
