FROM docker.io/verdictai/glm52-exl3-sparkinfer@sha256:bfd6d6670db37b04e9cbef7375722e3f71d66745abf1714c05cc5b71fd126715
LABEL org.opencontainers.image.title="GLM-5.2 EXL3 turnkey for vast.ai (4x RTX PRO 6000)" \
      org.opencontainers.image.description="512K-context GLM-5.2 OpenAI endpoint: EXL3 trellis weights, fp8 KV (stock-driver-safe), MTP spec decode, auto DRAM KV offload. Weights auto-download on first boot." \
      ai.malaiwah.evidence="gists: cae272443a 7d5d7e68 f3096ae9 e8a587ad 65bb725e 929d7d8e" \
      ai.malaiwah.base="verdictai/glm52-exl3-sparkinfer@sha256:bfd6d667"
RUN pip install --no-cache-dir "huggingface_hub[hf_transfer]" && apt-get update -qq && apt-get install -y -qq nvtop htop && rm -rf /var/lib/apt/lists/*
COPY entrypoint.sh /usr/local/bin/glm52-entry.sh
RUN chmod +x /usr/local/bin/glm52-entry.sh
ENTRYPOINT ["/usr/local/bin/glm52-entry.sh"]
