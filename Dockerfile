# GG v20 r20 contains the native mixed-K fixes from the appliance's memory
# campaign plus content-addressed EXL3 online quantization (including K6).
# Pin the immutable August 2 manifest and verify the exact native files before
# the one remaining parity-call compatibility repair. Never replay the r14/r17
# overlay bundles onto this unrelated source tree.
FROM docker.io/voipmonitor/vllm@sha256:40c891fd3fd573a92708e8a4bfa028ec91127a92491504c59006cf9735b20560
LABEL org.opencontainers.image.title="Multi-model vLLM turnkey for Vast.ai, Runpod, and JarvisLabs" \
      org.opencontainers.image.description="Profile-driven OpenAI endpoint: validated GLM-5.2 EXL3 production defaults plus a low-cost Qwen3.6-27B NVFP4 development profile. Weights auto-download on first boot." \
      ai.malaiwah.evidence="GG-v20-r20 vLLM#228 SparkInfer#106/#112/#113 online-EXL3-K6" \
      ai.malaiwah.base="voipmonitor/vllm@sha256:40c891fd3fd573a92708e8a4bfa028ec91127a92491504c59006cf9735b20560"
COPY requirements-soul.lock /opt/requirements-soul.lock
RUN echo "efd7e23ac1ace6da9dcd9046c46bca5cca68ed5e89cd648b5f8bc1d51eafebb2  /opt/vllm/kv-scales/glm52-nvfp4-nf3-hybrid_mla_outer_scales_v1.json" | sha256sum -c - \
 && pip install --no-cache-dir huggingface_hub==1.25.1 hf-xet==1.5.2 dnspython==2.8.0 && apt-get update -qq && apt-get install -y -qq nvtop htop curl openssh-server socat python3-venv util-linux patch && rm -rf /var/lib/apt/lists/* \
 && rm -f /etc/ssh/ssh_host_* \
 && curl -sSL -o /tmp/lego.tgz https://github.com/go-acme/lego/releases/download/v4.35.2/lego_v4.35.2_linux_amd64.tar.gz \
 && echo "ee5be4bf457de8e3efa86a51651c75c87f0ee0e4e9f3ae14f6034d68365770f3  /tmp/lego.tgz" | sha256sum -c - \
 && tar xzf /tmp/lego.tgz -C /usr/local/bin lego && rm /tmp/lego.tgz && chmod +x /usr/local/bin/lego \
 && python3 -m pip freeze | sort > /tmp/vllm-packages.before \
 && python3 -m venv /opt/nanobot-venv \
 && /opt/nanobot-venv/bin/pip install --no-cache-dir --require-hashes -r /opt/requirements-soul.lock \
 && /opt/nanobot-venv/bin/python -c "from nanobot import Nanobot; import importlib.metadata as m; assert m.version('nanobot-ai') == '0.3.0'" \
 && python3 -m pip freeze | sort > /tmp/vllm-packages.after \
 && diff -u /tmp/vllm-packages.before /tmp/vllm-packages.after \
 && rm /tmp/vllm-packages.before /tmp/vllm-packages.after \
 && groupadd --system soul && useradd --system --gid soul --home-dir /nonexistent --shell /bin/bash soul
COPY sshd_config /etc/ssh/sshd_config.d/99-model-turnkey.conf
COPY landing.py /opt/landing.py
COPY scripts/ /opt/scripts/
COPY patches/field-review-r20/ledger.json /opt/field-review-r20-ledger.json
COPY soul/ /opt/soul/
COPY entrypoint.sh /usr/local/bin/model-turnkey-entry.sh
# The public Vast template predates the provider-neutral rename and may still
# call glm52-entry.sh from its remote onstart field. Keep the old path as a
# compatibility alias so a stale provider template cannot strand a rental
# before the landing page is reachable.
RUN echo "ad8b9b1d202c65d68f4f3cdcb8c6b1dac0670216f03dfdde4429416b089baae6  /opt/scripts/patch_exl3_mixk.py" | sha256sum -c - \
 && echo "4436787d86b0fd90fb844073590111f67c87c4cf990c5310ee8e090fca554751  /opt/scripts/patch_sparkinfer_mixed_trellis_cache_key.py" | sha256sum -c - \
 && /opt/venv/bin/python /opt/scripts/verify_r20_base.py \
 && python3 /opt/scripts/patch_sparkinfer_mixed_trellis_cache_key.py \
 && python3 /opt/scripts/patch_exl3_parity_abi.py \
 && python3 /opt/scripts/patch_exl3_mixk.py \
 && chmod +x /usr/local/bin/model-turnkey-entry.sh /opt/scripts/soul_controller.py /opt/scripts/soul_config.py \
      /opt/scripts/glm52_lmcache_wrapper.sh /opt/scripts/acme_retry.sh \
 && chmod -R a-w /opt/soul \
 && ln -sf model-turnkey-entry.sh /usr/local/bin/glm52-entry.sh
EXPOSE 22 8000 8443 1111
ENTRYPOINT ["/usr/local/bin/model-turnkey-entry.sh"]
