# v26 = GG v20 final + EXL3 (upstream local-inference-lab/vllm#139 rebased on
# gilded-gnosis v20). The previous pin (sha256:bfd6d667, vllm 0.17.0rc1.dev4499+g60c82d972)
# was the PRE-convergence EXL3 build and carried none of the GG v20 work.
FROM docker.io/verdictai/glm52-exl3-sparkinfer@sha256:2bb9e804a283d1da3b7e3425ff87375121285141d0d0a40d3dc09d41bf881a10
LABEL org.opencontainers.image.title="Multi-model vLLM turnkey for Vast.ai and Runpod" \
      org.opencontainers.image.description="Profile-driven OpenAI endpoint: validated GLM-5.2 EXL3 production defaults plus a low-cost Qwen3.6-27B NVFP4 development profile. Weights auto-download on first boot." \
      ai.malaiwah.evidence="gists: cae272443a 7d5d7e68 f3096ae9 e8a587ad 65bb725e 929d7d8e" \
      ai.malaiwah.base="verdictai/glm52-exl3-sparkinfer@sha256:2bb9e804a283d1da3b7e3425ff87375121285141d0d0a40d3dc09d41bf881a10"
RUN pip install --no-cache-dir huggingface_hub && apt-get update -qq && apt-get install -y -qq nvtop htop curl openssh-server && rm -rf /var/lib/apt/lists/* \
 && rm -f /etc/ssh/ssh_host_* \
 && curl -sSL -o /tmp/lego.tgz https://github.com/go-acme/lego/releases/download/v4.21.0/lego_v4.21.0_linux_amd64.tar.gz \
 && echo "c8cc7fb636f8a5f1167e013dbd01485a72eb7393faf1776664c765a722cd6070  /tmp/lego.tgz" | sha256sum -c - \
 && tar xzf /tmp/lego.tgz -C /usr/local/bin lego && rm /tmp/lego.tgz && chmod +x /usr/local/bin/lego
COPY sshd_config /etc/ssh/sshd_config.d/99-model-turnkey.conf
COPY landing.py /opt/landing.py
COPY scripts/ /opt/scripts/
COPY entrypoint.sh /usr/local/bin/model-turnkey-entry.sh
RUN chmod +x /usr/local/bin/model-turnkey-entry.sh
EXPOSE 22 8000 1111
ENTRYPOINT ["/usr/local/bin/model-turnkey-entry.sh"]
