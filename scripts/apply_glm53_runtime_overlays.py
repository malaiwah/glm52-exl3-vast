#!/usr/bin/env python3
"""Install the live-qualified GLM-5.3 runtime overlays fail-closed.

The parent image is immutable, but it does not contain every fix used for the
K6 qualification. Each source payload and each destination state is pinned by
SHA-256. A build may start from the exact reviewed base state or from the exact
fully-overlaid state; a mixed or unknown state is rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
from pathlib import Path


# Parent: verdictai/glm53-flash-exl3-k4@
# sha256:0f1cdcc8891f1cc3a444121eb61d366289a1cbba285f0892dcbb24bc94961692
OVERLAYS = (
    (
        "exl3_patched.py",
        "/opt/infernal-invocation/vllm/vllm/model_executor/layers/quantization/exl3.py",
        "ae92592ea8fcd249978134357ea3cd2510fe2aa9bdb1d1a3ab02afdbaeb39f45",
        "ffef5aea103117a1bfb0023a43a59fba15b704566ad5acb0ddc47a18b9acede4",
    ),
    (
        "w4a16_kernel_k6.py",
        "/opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/w4a16/kernel.py",
        "28e6a2ca95934921718f75f8cf575be034c23da78776b40d478bef6bf8b0b3cd",
        "03be88e2044f54eb03d897ffa9782b5e5ce284ebf095a962bde149212dcbff7f",
    ),
    (
        "b12x_mixed_kernel.py",
        "/opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/w4a16/mixed_kernel.py",
        None,
        "f181cec7cb4e7573aea55fafaede3ec8293d4afcd6707469f1e09bf79b78376f",
    ),
    (
        "b12x_w4a16_host.py",
        "/opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/w4a16/host.py",
        "b0f7cc6e8e8c3e0dc35ae598af4a2410c953453bae24319085611afefb5446aa",
        "7c70320d150c590ee5a1c09a1194fccc79bb20ea69db02b8e98107ee7bd35089",
    ),
    (
        "b12x_mixed_trellis.py",
        "/opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/w4a16/mixed_trellis.py",
        "cbd3dac9d7608019f860999bdd256a42c7737cf2a00d33c986964c58f4147072",
        "4f934a40dbf10804e759239a34205d85a6622b19007d8fae2af3c16b2191785a",
    ),
    (
        "b12x_fused_moe_impl_image.py",
        "/opt/infernal-invocation/b12x/b12x/moe/fused_moe/_impl.py",
        "bc08e69ba855307000608d57eb64dc066c73330e797b4083195b2f4241da1309",
        "f7f905346fcc6c13db36616aedb987fedb5874eee296558ee2efa3dca3aaf111",
    ),
    (
        "tilelang_kernels_mhc_serial.py",
        "/opt/infernal-invocation/vllm/vllm/model_executor/kernels/mhc/tilelang_kernels.py",
        "ddbf95d0aa7b79820a277578c5060ac26650d9432d7c1fcd13ae9e9ee9ae25f8",
        "6aecacff6b2e3e83a6f6d92b9a9d9b6285b80d80e48bc4cb9193fcee68fa3a58",
    ),
    (
        "dcp_utils_layout.py",
        "/opt/infernal-invocation/vllm/vllm/v1/attention/ops/dcp_utils.py",
        "30cde4ed5ffb651f91c412e7880d4e1c2ae66aae83e61c21b162db9d6fad8644",
        "c1bd0dbb3c17c0dd439b4a02b6d01f42326f96f41b9445c6a60a97a1b78f802a",
    ),
    (
        "torchao_utils_enum.py",
        "/usr/local/lib/python3.12/dist-packages/torchao/utils.py",
        "4956a728a2b021a5eae3482fd3f686eb2b1424fbb350afc86804c166e4dc6c8e",
        "81868a8940e0009b7c216e5ea4b7f126ba3cb99d81eae2e9984dfc0d99501cbc",
    ),
    (
        "envs_b12x_glm_nope.py",
        "/opt/infernal-invocation/vllm/vllm/envs.py",
        "e489510c9fb40bb6f0b8fd803a534edbb9c4fc430d20780b5ca91d1eded2b992",
        "37701b15f9401abeab90da44423d8af8a9aa315c25e89c5ff457b60b73040da2",
    ),
    (
        "b12x_mla_smem.py",
        "/opt/infernal-invocation/b12x/b12x/attention/_shared/mla/smem.py",
        "20da45a335c09088a9b1200d646364d74115ea60b1e57d02b601ae06442dab83",
        "49ba08ba98d92d14d836c484558141d7b03f3c6e4809ea1b8e17011e0eece40a",
    ),
    (
        "b12x_mla_smem_mg.py",
        "/opt/infernal-invocation/b12x/b12x/attention/_shared/mla/smem_mg.py",
        "f3e61a8527ad7b0d84970058b3a7a125e0cba517ced54f74afba17368a4fe92f",
        "32b73906b61a6cb8a648ef36fc41de9979bd920af6346a505d6b7f9fd7ada37f",
    ),
    (
        "kda_no_block_ptr.py",
        "/opt/infernal-invocation/vllm/vllm/third_party/flash_linear_attention/ops/kda.py",
        "8508d5e091e645baa86ba698ef1366298fed2070bfa63b40522ce15a434a2c68",
        "5be4e062f3f140c2beb0d7e667a96cf4bd1ad60103e837fabf941ed3a2bec0c3",
    ),
    (
        "solve_tril_no_block_ptr.py",
        "/opt/infernal-invocation/vllm/vllm/third_party/flash_linear_attention/ops/solve_tril.py",
        "d1a4ff27623a938825052500afa7e2aa24fadd2fa4c9be6376c9128824f86f82",
        "ceb27ecebb48324a2cfe1bcd82364da3450af3d1046b3eadf94b95699520da7d",
    ),
    (
        "chunk_delta_h_no_block_ptr.py",
        "/opt/infernal-invocation/vllm/vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py",
        "1b3ad391f939d9443c6b7adb19e57fe381bd5dccea064e8417a4f85b0e713b26",
        "d98d59ff1c05ae78edc069dfe85da9ab5dfd5b2568ffbe0c9b52ac5806864e5a",
    ),
    (
        "rotary_common_image.py",
        "/opt/infernal-invocation/vllm/vllm/model_executor/layers/rotary_embedding/common.py",
        "8fabf57cee7b127c71d57745a0fe99a312793ec4848fe38dfb9810d0e0554ade",
        "9e2e8ca4d011e4baeae9ed1e14160dbc9f52ea7cbdadc3531cfddce3cc534895",
    ),
    (
        "deepseek_v4_mhc_warmup.py",
        "/opt/infernal-invocation/vllm/vllm/model_executor/warmup/deepseek_v4_mhc_warmup.py",
        "5c2d2a1a64593b2357d58fee6311b1c8b898e021eee01046be0713ddddf2df21",
        "de793cc359f1491d6893de9ace95ec96bfc19af6c0a03f3258455aa793ff33b1",
    ),
    (
        "pcie_oneshot_device_barrier.py",
        "/opt/infernal-invocation/b12x/b12x/comm/pcie/pcie_oneshot.py",
        "6630cd2dea7d04ba44d8639f66bebf761cb0021266311240d35ff81cd0246d1d",
        "ddbcbdde5b2092a5fa9c2adef3e83c54bbb99b34314b515e55bf69859984dadc",
    ),
    (
        "mla_attention_sparse_info.py",
        "/opt/infernal-invocation/vllm/vllm/model_executor/layers/attention/mla_attention.py",
        "a070481091d425eddc311e25c0cb770e7ed79d1ee3d196157ad28d789fb5c60c",
        "e494a440a0a6b504c54560de637a878f42ab4f281cd10532d0aa753075c6cfea",
    ),
    (
        "glm5next_model_router_once.py",
        "/opt/infernal-invocation/vllm/vllm/models/glm5next/nvidia/model.py",
        "bb75317190834687f69519ec9d530e511e6a284ddea884b982035bec199fdd95",
        "8e78413ce4c952162437858d11dd0ab4ad8de5391b62310f54b686cb4e87e133",
    ),
    (
        "route_pack_image_exact.py",
        "/opt/infernal-invocation/b12x/b12x/moe/_shared/kernels/w4a16/route_pack.py",
        "6fab8230d72e664415f1242d975de342d40b642173fa395b16e4ccb29d2b69f6",
        "48e005c6aab7b325fe4a4c09817bc7960ece3b34eec989a779b761084ab7194a",
    ),
    (
        "sparse_mla_triton_warmup.py",
        "/opt/infernal-invocation/vllm/vllm/model_executor/warmup/sparse_mla_triton_warmup.py",
        "52916c8d800fa2dba760366ad7349e3bfc72fe23c55ed83442315ece55441527",
        "f5548259fedb4cd4d37ecb752ba414cdd78e30aa7ef05872c26a4fd73f8b111e",
    ),
    (
        "indexer.py",
        "/opt/infernal-invocation/vllm/vllm/v1/attention/backends/mla/indexer.py",
        "05fbd197ab2995b05ae24247f02fe7bdeb8f431fb860fc6fb0a40b8afb8cf665",
        "0b77e7305ff7c28aac589a797b67efc445c356e06c21c13ea978f583cc9e7cb5",
    ),
    (
        "b12x_mla_sparse.py",
        "/opt/infernal-invocation/vllm/vllm/v1/attention/backends/mla/b12x_mla_sparse.py",
        "0499c674b6890266b50fa0d5724dcfbb83cba3917714a6787e5dddc6feb65572",
        "bd43a2865580884d43658c15c718f5f94fcbcb4de0985f5da0ca93d94818aa22",
    ),
    (
        "kpool_compress.py",
        "/opt/infernal-invocation/vllm/vllm/models/glm5next/nvidia/ops/kpool_compress.py",
        "01bfba91f667214760e1fe8af8c4151498ae9b0407493a4670e255945ae01a58",
        "245c889b25afd8969adebf27414ff5d5274c5c6143c5d5183ac1dee85ed4ca0d",
    ),
    (
        "glm5_kpool_warmup.py",
        "/opt/infernal-invocation/vllm/vllm/model_executor/warmup/glm5_kpool_warmup.py",
        None,
        "1499c7650e44279b98a82fd1779e63f2c2a2a8619f46b07e5ae4b446c7e2c4fb",
    ),
    (
        "kernel_warmup_image_exact.py",
        "/opt/infernal-invocation/vllm/vllm/model_executor/warmup/kernel_warmup.py",
        "038f4b54da0a23e11c7505bc9d4c4da0e4b5f96d51d43223748a0b7eeca40e1e",
        "36e24893a211b004baa8265bec46fbf06a9fc4f574b65a868b88a35eb5974f7f",
    ),
)


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    if not path.is_file():
        raise RuntimeError(f"expected regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install(source_dir: Path, *, verify_only: bool = False) -> None:
    source_states: list[str] = []
    target_states: list[str] = []
    for source_name, target_name, before, after in OVERLAYS:
        source = source_dir / source_name
        source_hash = file_sha256(source)
        if source_hash != after:
            source_states.append(
                f"{source}: expected overlay={after}, got={source_hash}"
            )
        target_hash = file_sha256(Path(target_name))
        if target_hash == before:
            target_states.append("before")
        elif target_hash == after:
            target_states.append("after")
        else:
            target_states.append("unknown")
            source_states.append(
                f"{target_name}: expected base={before} or overlay={after}, "
                f"got={target_hash}"
            )

    if source_states:
        raise RuntimeError("refusing unknown GLM-5.3 runtime state:\n  " +
                           "\n  ".join(source_states))
    if all(state == "after" for state in target_states):
        print(">>> GLM-5.3 runtime overlays already installed and verified")
        return
    if not all(state == "before" for state in target_states):
        raise RuntimeError(
            "refusing mixed GLM-5.3 runtime state: " + ", ".join(target_states)
        )
    if verify_only:
        print(">>> GLM-5.3 runtime base and overlay payloads verified")
        return

    staged: list[tuple[Path, Path]] = []
    try:
        for source_name, target_name, _before, _after in OVERLAYS:
            source = source_dir / source_name
            target = Path(target_name)
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(
                prefix=f".{target.name}.glm53-", dir=target.parent
            )
            os.close(fd)
            temporary_path = Path(temporary)
            shutil.copyfile(source, temporary_path)
            os.chmod(temporary_path, target.stat().st_mode & 0o777 if target.exists()
                     else 0o644)
            staged.append((temporary_path, target))
        for temporary, target in staged:
            os.replace(temporary, target)
    finally:
        for temporary, _target in staged:
            temporary.unlink(missing_ok=True)

    bad = [target for _source, target, _before, after in OVERLAYS
           if file_sha256(Path(target)) != after]
    if bad:
        raise RuntimeError("overlay verification failed: " + ", ".join(bad))
    print(f">>> installed and verified {len(OVERLAYS)} GLM-5.3 runtime overlays")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    install(args.source_dir, verify_only=args.verify_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
