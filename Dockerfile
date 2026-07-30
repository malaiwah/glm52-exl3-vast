# GG v20 r13 retains the paired dynamic-token NVFP4 MLA cache ABI, adaptive
# sparse-indexer folding, DCP-aware LMCache and XGrammar 0.2.5 from r11. It
# consolidates the EXL3/SparkInfer fused-MoE path, fixes target/draft small-row
# planning and makes the post-warmup Trellis arena peak repeatable. Dynamic KV
# scaling remains the flagship EXL3 default after repeatable KLD and exact
# long-context retrieval gates.
# Pin the immutable July 30 manifest. This compute-base change deliberately
# changes the compile-cache fingerprint and remains a requalification boundary
# for memory planning, retrieval, degeneration and performance.
FROM docker.io/voipmonitor/vllm@sha256:02796036c96a52fda0919aa260c45c70bc97d8e662a6ae5e614b5f987c20851b
LABEL org.opencontainers.image.title="Multi-model vLLM turnkey for Vast.ai, Runpod, and JarvisLabs" \
      org.opencontainers.image.description="Profile-driven OpenAI endpoint: validated GLM-5.2 EXL3 production defaults plus a low-cost Qwen3.6-27B NVFP4 development profile. Weights auto-download on first boot." \
      ai.malaiwah.evidence="gists: cae272443a 7d5d7e68 f3096ae9 e8a587ad 65bb725e 929d7d8e" \
      ai.malaiwah.base="voipmonitor/vllm@sha256:02796036c96a52fda0919aa260c45c70bc97d8e662a6ae5e614b5f987c20851b"
COPY requirements-soul.lock /opt/requirements-soul.lock
RUN echo "efd7e23ac1ace6da9dcd9046c46bca5cca68ed5e89cd648b5f8bc1d51eafebb2  /opt/vllm/kv-scales/glm52-nvfp4-nf3-hybrid_mla_outer_scales_v1.json" | sha256sum -c - \
 && pip install --no-cache-dir huggingface_hub==1.25.1 hf-xet==1.5.2 dnspython==2.8.0 && apt-get update -qq && apt-get install -y -qq nvtop htop curl openssh-server socat python3-venv util-linux && rm -rf /var/lib/apt/lists/* \
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
COPY soul/ /opt/soul/
COPY entrypoint.sh /usr/local/bin/model-turnkey-entry.sh
# The public Vast template predates the provider-neutral rename and may still
# call glm52-entry.sh from its remote onstart field. Keep the old path as a
# compatibility alias so a stale provider template cannot strand a rental
# before the landing page is reachable.
RUN echo "44e171a8aea0009e3c786265914c6cf9d8a23ca9b474761aba990adcb7f74440  /opt/scripts/patch_exl3_mixk.py" | sha256sum -c - \
 && python3 /opt/scripts/patch_exl3_parity_abi.py \
 && python3 /opt/scripts/patch_exl3_mixk.py \
 && chmod +x /usr/local/bin/model-turnkey-entry.sh /opt/scripts/soul_controller.py /opt/scripts/soul_config.py \
      /opt/scripts/glm52_lmcache_wrapper.sh /opt/scripts/acme_retry.sh \
 && chmod -R a-w /opt/soul \
 && ln -sf model-turnkey-entry.sh /usr/local/bin/glm52-entry.sh
EXPOSE 22 8000 8443 1111
ENTRYPOINT ["/usr/local/bin/model-turnkey-entry.sh"]
