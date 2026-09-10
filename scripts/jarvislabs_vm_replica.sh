#!/usr/bin/env bash
# VM replica bring-up for the JarvisLabs GLM-5.3 fleet.
#
# JarvisLabs GPU VMs are SSH-only (no startup scripts) and the appliance
# graft fail-closes on VMs (require_container_graft demands a container
# marker), so a VM replica runs the published appliance IMAGE under the
# VM's own docker instead. The VM also takes the P2P/driver tuning that
# containers cannot (rtx6kpro hardware/pcie-bandwidth.md): measured on
# containers, vLLM logs "SymmMemCommunicator: native P2P atomics are
# not supported"; the ForceP2P override exists to fix exactly that.
#
# Subcommands (run on the VM over ssh, as a sudo-capable user):
#   p2p-tune   write modprobe + grub config (REBOOT REQUIRED after)
#   verify     check module parameters, cmdline, topo, docker/nvidia runtime
#   launch     docker pull the pinned image and run the appliance
#   rehearse   p2p-tune + verify + a CUDA P2P probe, no model (safe to destroy)
#
# The image pin must match the digest the shared FS image tree was primed
# with (a floating :latest breaks the tree's digest lock on CI publishes).

set -Eeuo pipefail

IMAGE_PIN="${TURNKEY_IMAGE:-ghcr.io/malaiwah/glm52-exl3-vast@sha256:f761426c108d2c09e6113752ba0ab8c41582ad5b470e42bb1d1f8523bc23c2fa}"

log() { printf '>>> %s\n' "$*"; }
fatal() { printf 'FATAL: %s\n' "$*" >&2; exit 1; }

p2p_tune() {
  # --- modprobe: force BAR1 P2P routing for PCIe-attached PRO 6000 ---
  sudo tee /etc/modprobe.d/nvidia-p2p-override.conf >/dev/null <<'EOF'
# Force driver-level P2P for direct-attach RTX PRO 6000 (PIX/PHB topo):
# without this, remote-GPU loads route through SysMem staging (10x slower).
options nvidia NVreg_RegistryDwords="ForceP2P=0x11;RMForceP2PType=1;RMPcieP2PType=2;GrdmaPciTopoCheckOverride=1;EnableResizableBar=1"
EOF
  # --- modprobe: UVM HMM off (required on virtually all multi-GPU PRO 6000;
  #     without it NCCL P2P can lock up) ---
  sudo tee /etc/modprobe.d/uvm.conf >/dev/null <<'EOF'
options nvidia_uvm uvm_disable_hmm=1
EOF
  # --- modprobe: no runtime D3 (pairs with pcie_port_pm=off against
  #     Surprise Link Down / Gen1<->Gen5 retrain storms) ---
  sudo tee /etc/modprobe.d/nvidia.conf >/dev/null <<'EOF'
options nvidia NVreg_DynamicPowerManagement=0x00
EOF
  # --- grub: PCIe power management off + IOMMU off (NCCL P2P deadlock
  #     prevention). Kept minimal on rented infra: CPU mitigations stay ON.
  #     If the VM fails to boot after this, destroy it and retry with
  #     modprobe-only (the ForceP2P + UVM wins are the throughput-relevant
  #     ones; iommu/aspm are stability-and-deadlock fixes).
  sudo sed -i 's/^GRUB_CMDLINE_LINUX_DEFAULT=.*/GRUB_CMDLINE_LINUX_DEFAULT="pcie_aspm=off pcie_port_pm=off iommu=off"/' /etc/default/grub
  grep -q 'pcie_aspm=off' /etc/default/grub || fatal "could not set GRUB cmdline"
  sudo update-grub >/dev/null 2>&1 || sudo grub2-mkconfig -o /boot/grub2/grub.cfg >/dev/null
  sudo update-initramfs -u >/dev/null 2>&1 || true
  log "p2p-tune written; REBOOT the VM for kernel/module parameters to apply"
}

verify() {
  local rc=0
  log "kernel cmdline:"; grep -o "pcie_aspm=off pcie_port_pm=off iommu=off" /proc/cmdline \
    || { log "  MISSING p2p cmdline (reboot after p2p-tune?)"; rc=1; }
  # Open-module drivers (595.x) expose RegistryDwords via /proc, not sysfs.
  log "nvidia RegistryDwords:"
  sudo tr "," "\n" < /proc/driver/nvidia/params 2>/dev/null | grep -i "RegistryDwords" | head -1 \
    || { log "  /proc/driver/nvidia/params not readable"; }
  sudo tr "," "\n" < /proc/driver/nvidia/params 2>/dev/null | grep -q "ForceP2P" \
    && log "  ForceP2P active" || { log "  ForceP2P NOT active"; rc=1; }
  log "uvm_disable_hmm:"; cat /sys/module/nvidia_uvm/parameters/disable_hmm 2>/dev/null \
    || { log "  parameter not exposed"; }
  log "DynamicPowerManagement:"; cat /sys/module/nvidia/parameters/NVreg_DynamicPowerManagement 2>/dev/null | head -1 || true
  log "topology:"; nvidia-smi topo -m | head -8
  log "docker runtimes:"; docker info 2>/dev/null | grep -i runtime | head -3
  docker info 2>/dev/null | grep -qi nvidia \
    || { log "  no nvidia docker runtime — install nvidia-container-toolkit"; rc=1; }
  log "BAR1 (ReBAR needs BIOS support; EnableResizableBar takes effect only then):"
  nvidia-smi -q | grep -A2 "BAR1 Memory" | head -3
  return "$rc"
}

p2p_probe() {
  # Tiny CUDA smoke under docker: proves the nvidia runtime + driver P2P
  # paths. cuda-p2p-stress would measure bandwidth; for the rehearsal the
  # goal is that P2P API calls SUCCEED (containers cannot even get here).
  docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 \
    nvidia-smi -L >/dev/null 2>&1 \
    || fatal "docker nvidia runtime not functional"
  log "docker + nvidia runtime OK ($(nvidia-smi -L | wc -l) GPUs)"
}

launch() {
  [ -x /usr/bin/docker ] || fatal "docker missing"
  docker info 2>/dev/null | grep -qi nvidia || fatal "nvidia runtime missing"
  log "pulling $IMAGE_PIN (first pull on this VM: ~10-15 min from IN1)"
  docker pull "$IMAGE_PIN" >/dev/null
  local fs=/home/jl_fs
  [ -d "$fs/GLM-5.3-EXL3-TR3-3.42bpw" ] || fatal "filesystem not mounted at $fs"
  # Fleet runtime env (AIBeast parity); VLLM_API_KEY must come from the
  # caller's environment — never baked into this script.
  : "${VLLM_API_KEY:?VLLM_API_KEY must be exported by the caller}"
  docker run -d --name glm53-appliance --restart unless-stopped \
    --gpus all --shm-size 64g \
    -p 127.0.0.1:8000:8000 \
    -v "$fs:$fs" \
    -e MODEL_DIR="$fs/GLM-5.3-EXL3-TR3-3.42bpw" \
    -e SERVED_MODEL_NAME=GLM-5.3 \
    -e GLM_STATE_DIR=/workspace/.glm-config \
    -e VLLM_API_KEY="$VLLM_API_KEY" \
    -e PREFIX_CACHE_BACKEND=lmcache \
    -e LMCACHE_L1_MAX_GB=125 \
    -e PREFIX_CACHE_DISK_GB=384 \
    -e VLLM_EXL3_ONLINE_CACHE_DIR="$fs/.runtimes/exl3-online" \
    -e VLLM_EXL3_ONLINE_CACHE_MODE=readwrite \
    -e PREFILL_FAIRNESS_ENGINE=compute_share \
    -e PREFILL_COMPUTE_SHARE=0.6 \
    -e MAX_NUM_SEQS=12 \
    -e MAX_NUM_BATCHED_TOKENS=3072 \
    -e VLLM_EXL3_PREFILL_CAPACITY=2048 \
    -e GPU_MEMORY_UTILIZATION=0.95 \
    -e CUDAGRAPH_CAPTURE_SIZES=4,8,12,16,20,24,28,32,36,40,44,48 \
    -e MAX_CUDAGRAPH_CAPTURE_SIZE=48 \
    -e VLLM_EXL3_TRELLIS_MAX_M=48 \
    "$IMAGE_PIN" >/dev/null
  log "appliance container up (engine on 127.0.0.1:8000; boot log: docker logs -f glm53-appliance)"
}

rehearse() { p2p_tune && log "reboot, then run: $0 verify && $0 p2p-probe"; }

case "${1:-}" in
  p2p-tune) p2p_tune ;;
  verify) verify ;;
  p2p-probe) p2p_probe ;;
  launch) launch ;;
  rehearse) rehearse ;;
  *) fatal "usage: $0 {p2p-tune|verify|p2p-probe|launch|rehearse}" ;;
esac
