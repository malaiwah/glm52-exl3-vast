# v29 = GG v20 + the current EXL3/Trellis layer. In particular, draft layers
# now advertise a capturable m=1 minimum while target layers retain m=4. That
# removes the old global VLLM_EXL3_TRELLIS_MIN_M workaround and its startup
# failure at the MTP draft's m=1..3 graph shapes.
FROM docker.io/verdictai/glm52-exl3-sparkinfer@sha256:2996b8ac37ff126a8aeebaa24df72e2154a2a1573df41f99eb48a4275e33eb41
LABEL org.opencontainers.image.title="Multi-model vLLM turnkey for Vast.ai and Runpod" \
      org.opencontainers.image.description="Profile-driven OpenAI endpoint: validated GLM-5.2 EXL3 production defaults plus a low-cost Qwen3.6-27B NVFP4 development profile. Weights auto-download on first boot." \
      ai.malaiwah.evidence="gists: cae272443a 7d5d7e68 f3096ae9 e8a587ad 65bb725e 929d7d8e" \
      ai.malaiwah.base="verdictai/glm52-exl3-sparkinfer@sha256:2996b8ac37ff126a8aeebaa24df72e2154a2a1573df41f99eb48a4275e33eb41"
COPY requirements-soul.lock /opt/requirements-soul.lock
RUN pip install --no-cache-dir huggingface_hub && apt-get update -qq && apt-get install -y -qq nvtop htop curl openssh-server socat python3-venv util-linux && rm -rf /var/lib/apt/lists/* \
 && rm -f /etc/ssh/ssh_host_* \
 && curl -sSL -o /tmp/lego.tgz https://github.com/go-acme/lego/releases/download/v4.21.0/lego_v4.21.0_linux_amd64.tar.gz \
 && echo "c8cc7fb636f8a5f1167e013dbd01485a72eb7393faf1776664c765a722cd6070  /tmp/lego.tgz" | sha256sum -c - \
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
RUN chmod +x /usr/local/bin/model-turnkey-entry.sh /opt/scripts/soul_controller.py /opt/scripts/soul_config.py \
 && chmod -R a-w /opt/soul \
 && ln -sf model-turnkey-entry.sh /usr/local/bin/glm52-entry.sh
EXPOSE 22 8000 8443 1111
ENTRYPOINT ["/usr/local/bin/model-turnkey-entry.sh"]
