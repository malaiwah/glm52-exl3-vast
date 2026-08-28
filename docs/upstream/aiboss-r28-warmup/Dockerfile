FROM docker.io/voipmonitor/vllm:gilded-gnosis-v20-vllme1e9426-si200c1db-fi801d57a-cu132-20260804-r28

ARG B12X_PR=126
ARG B12X_COMMIT=6022e6e7c7ea1199a06a27cf5a777c2804b13cfb
ARG VLLM_PR=250
ARG VLLM_COMMIT=2a34b0760f4d9bbd5d2ff6809238593098bd46ff

LABEL local-inference.test.base="GGv20r28" \
      local-inference.test.b12x-pr="${B12X_PR}@${B12X_COMMIT}" \
      local-inference.test.vllm-pr="${VLLM_PR}@${VLLM_COMMIT}" \
      local-inference.test.note="PR patches only; b12x namespace normalized to r28 legacy sparkinfer"

COPY b12x-r28.patch /tmp/b12x-r28.patch
COPY vllm-r28.patch /tmp/vllm-r28.patch

RUN set -eu; \
    cd /opt/venv/lib/python3.12/site-packages; \
    git apply --check /tmp/b12x-r28.patch; \
    git apply --check /tmp/vllm-r28.patch; \
    git apply /tmp/b12x-r28.patch; \
    git apply /tmp/vllm-r28.patch; \
    /opt/venv/bin/python -m py_compile \
      sparkinfer/moe/_shared/kernels/w4a16/host.py \
      sparkinfer/moe/_shared/kernels/w4a16/mixed_trellis.py \
      sparkinfer/moe/fused_moe/_impl.py \
      vllm/model_executor/layers/quantization/exl3.py \
      vllm/model_executor/warmup/kernel_warmup.py; \
    rm -f /tmp/b12x-r28.patch /tmp/vllm-r28.patch
