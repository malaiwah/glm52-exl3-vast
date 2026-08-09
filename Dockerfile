# GG v20 r28 keeps r26's lossless TP4/DCP4 policy and adds the complete
# shared_h_v1 + runtime-dynamic mixed-Trellis contract needed by the 3.42bpw
# quality checkpoint. Pin the immutable August 4 manifest and fail closed on
# its reviewed source hashes.
# A small vLLM overlay suppresses a false scheduler warning for serial GLM MTP,
# whose draft path consumes no additional scheduler slots.
FROM docker.io/voipmonitor/vllm@sha256:501e10e79b4bc854237804d215e454c531ac9c2d354a8fa1a93e450fe7ba6ce0
LABEL org.opencontainers.image.title="Multi-model vLLM turnkey for Vast.ai, Runpod, and JarvisLabs" \
      org.opencontainers.image.description="Profile-driven OpenAI endpoint: validated GLM-5.2 EXL3 production defaults plus a low-cost Qwen3.6-27B NVFP4 development profile. Weights auto-download on first boot." \
      ai.malaiwah.evidence="GG-v20-r28 shared_h_v1 TP4-DCP4-lossless-auto online-EXL3-K6 serial-MTP-warning-fix" \
      ai.malaiwah.base="voipmonitor/vllm@sha256:501e10e79b4bc854237804d215e454c531ac9c2d354a8fa1a93e450fe7ba6ce0"
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
COPY patches/field-review-r26/ledger.json /opt/field-review-r26-ledger.json
COPY soul/ /opt/soul/
COPY entrypoint.sh /usr/local/bin/model-turnkey-entry.sh
# The public Vast template predates the provider-neutral rename and may still
# call glm52-entry.sh from its remote onstart field. Keep the old path as a
# compatibility alias so a stale provider template cannot strand a rental
# before the landing page is reachable.
RUN echo "99f597b78c83fce4d7d568305ba147d17a8daf866e833eca42be79d8ec185544  /opt/scripts/patch_exl3_mixk.py" | sha256sum -c - \
 && echo "a0b7bc8377a5e29a921da4971d63b5260dac34601598285fee6cce3cd94bc65c  /opt/scripts/patch_vllm_serial_spec_warning.py" | sha256sum -c - \
 && echo "e5a442f9aac0493f7fefe8584acfee923f99f4c952382dc6e41670d8b7c8a638  /opt/scripts/patch_exl3_parity_abi.py" | sha256sum -c - \
 && /opt/venv/bin/python /opt/scripts/verify_r28_base.py \
 && python3 /opt/scripts/patch_vllm_serial_spec_warning.py \
 && python3 /opt/scripts/patch_exl3_parity_abi.py \
 && python3 /opt/scripts/patch_exl3_mixk.py \
 && chmod +x /usr/local/bin/model-turnkey-entry.sh /opt/scripts/soul_controller.py /opt/scripts/soul_config.py \
      /opt/scripts/glm52_lmcache_wrapper.sh /opt/scripts/acme_retry.sh \
 && chmod -R a-w /opt/soul \
 && ln -sf model-turnkey-entry.sh /usr/local/bin/glm52-entry.sh
EXPOSE 22 8000 8443 1111
ENTRYPOINT ["/usr/local/bin/model-turnkey-entry.sh"]
