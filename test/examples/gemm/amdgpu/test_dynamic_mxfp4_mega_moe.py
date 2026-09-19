"""MegaMoE host contracts, packed transport quantization, and EP smoke tests."""

from dataclasses import replace

import pytest
import torch
from avelang_kernels.amdgpu.mega_moe import (
    ActivationFunction,
    DataType,
    MegaMoeConfig,
    MegaMoeSolutionId,
    ProducerGeometry,
    get_2stage_cfgs,
    quantize_input,
    registered_solutions,
)
from avelang_kernels.amdgpu.mega_moe.workspace import WorkspaceLayout


def gpt_config(world=2):
    return get_2stage_cfgs(8, world, 32, 4, 2880, 3072, activation="swiglu", bias_dtype="bf16")


def test_mega_codec_matches_petit_abi_and_preserves_all_registered_fields():
    assert int(gpt_config().solution) == 0x28B501918511
    assert int(gpt_config(8).solution) == 0x28B503918511
    solutions = registered_solutions()
    assert len(solutions) == len(set(map(int, solutions))) == 12
    for solution in solutions:
        assert MegaMoeSolutionId.from_int(int(solution)) == solution
    with pytest.raises(ValueError, match="reserved bits"):
        MegaMoeSolutionId.from_int(int(solutions[0]) | (1 << 52))


@pytest.mark.parametrize(
    "field,value", [("world_size", 1), ("experts", 16), ("topk", 32), ("hidden", 2881), ("intermediate", 384)]
)
def test_mega_codec_rejects_unrepresentable_fields(field, value):
    with pytest.raises(ValueError):
        replace(gpt_config().solution, **{field: value})


@pytest.mark.parametrize(
    "tokens,geometry,tile",
    [
        (8, ProducerGeometry.CTA128, (32, 4, 32)),
        (16, ProducerGeometry.CTA64, (32, 4, 32)),
        (256, ProducerGeometry.CTA56, (64, 4, 32)),
        (1024, ProducerGeometry.CTA56, (64, 8, 64)),
    ],
)
def test_gptoss120b_selects_petit_producer_and_stage1_geometry(tokens, geometry, tile):
    cfg = get_2stage_cfgs(tokens, 8, 128, 4, 2880, 3072, activation="swiglu", bias_dtype="bf16")
    assert cfg.solution.producer_geometry == geometry
    assert (cfg.stage1_tile_m, cfg.stage1_num_warps, cfg.stage1_wave_m) == tile


@pytest.mark.parametrize("first,second", [(0, 8), (12, 16), (32, 128), (256, 512)])
def test_kernel_factories_reuse_equivalent_token_buckets(first, second):
    from avelang_kernels.amdgpu.mega_moe.dispatch import make_kernels

    config = get_2stage_cfgs(first, 8, 128, 4, 2880, 3072, activation="swiglu", bias_dtype="bf16")
    other = get_2stage_cfgs(second, 8, 128, 4, 2880, 3072, activation="swiglu", bias_dtype="bf16")
    assert config == other
    a, b = make_kernels(config), make_kernels(other)
    assert all(left is right for left, right in zip(a, b, strict=True))


@pytest.mark.parametrize("first,second", [(8, 16), (128, 256), (512, 1024)])
def test_stage1_cache_preserves_producer_and_tile_changes(first, second):
    from avelang_kernels.amdgpu.mega_moe.dispatch import make_kernels

    config = get_2stage_cfgs(first, 8, 128, 4, 2880, 3072, activation="swiglu", bias_dtype="bf16")
    other = get_2stage_cfgs(second, 8, 128, 4, 2880, 3072, activation="swiglu", bias_dtype="bf16")
    assert make_kernels(config)[0] is not make_kernels(other)[0]


def test_deepseek_density_threshold_and_workspace_layout():
    cfg = get_2stage_cfgs(128, 8, 256, 8, 7168, 2048, activation="silu", bias_dtype="none")
    assert cfg.solution.producer_geometry == ProducerGeometry.CTA192
    assert (cfg.stage1_tile_m, cfg.stage1_num_warps, cfg.stage1_wave_m) == (64, 8, 64)
    smaller = get_2stage_cfgs(127, 8, 256, 8, 7168, 2048, activation="silu", bias_dtype="none")
    assert (smaller.stage1_tile_m, smaller.stage1_num_warps, smaller.stage1_wave_m) == (32, 4, 32)
    layout = WorkspaceLayout(cfg)
    assert layout.barrier_record_bytes == 8192
    assert layout.pool_rows == 66528
    assert cfg.input_token_bytes == 3808
    assert layout.workspace_bytes < 2**32
    for tokens in (-1, 1025):
        with pytest.raises(ValueError, match="token count"):
            get_2stage_cfgs(tokens, 8, 256, 8, 7168, 2048, activation="silu", bias_dtype="none")


def test_codec_does_not_claim_kernel_support():
    solution = MegaMoeSolutionId(2, 32, 4, 1024, 512, ActivationFunction.SILU_DOT, DataType.NONE)
    assert MegaMoeSolutionId.from_int(int(solution)) == solution
    assert MegaMoeConfig(solution).compute_hidden == 1024
    with pytest.raises(ValueError, match="no MegaMoE implementation"):
        get_2stage_cfgs(8, 2, 32, 4, 1024, 512, activation="silu", bias_dtype="none")


def test_kimi_ep8_codec_dispatch_and_pr38_workspace_layout():
    cfg = get_2stage_cfgs(32, 8, 896, 16, 3584, 3072, activation="kimi_situ", bias_dtype="none")
    assert MegaMoeSolutionId.from_int(int(cfg.solution)) == cfg.solution
    assert cfg.solution.activation == ActivationFunction.SITU_V2
    assert cfg.local_experts == 112
    assert cfg.solution.producer_geometry == ProducerGeometry.CTA56
    layout = WorkspaceLayout(cfg)
    assert layout.barrier_record_bytes == 16384
    assert layout.send_counts >= layout.local_offset
    assert layout.recv_sum >= layout.local_offset
    assert layout.recv_tokens >= layout.local_offset
    assert layout.input_weights >= layout.local_offset
    assert layout.input_tokens >= layout.local_offset


def has_gpus(count):
    return (
        torch.cuda.is_available()
        and torch.cuda.device_count() >= count
        and all(
            getattr(torch.cuda.get_device_properties(i), "gcnArchName", "").startswith("gfx950") for i in range(count)
        )
    )


@pytest.mark.skipif(not has_gpus(1), reason="gfx950 required")
@pytest.mark.parametrize("columns", [2880, 7168])
def test_transport_quantization_encoding_padding_and_preallocated_views(columns):
    table = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.bfloat16, device="cuda"
    )
    x = torch.empty((2, columns + 8), dtype=torch.bfloat16, device="cuda")[:, :columns]
    x.copy_(table.repeat(2, columns // 16))
    values, scales = quantize_input(x)
    expected = torch.tensor([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE], dtype=torch.uint8, device="cuda").repeat(
        2, columns // 16
    )
    assert torch.equal(values, expected)
    assert torch.equal(scales, torch.full_like(scales, 127))
    stride = 1536 if columns == 2880 else 3808
    assert values.stride() == scales.stride() == (stride, 1)
    assert scales.data_ptr() == values.data_ptr() + columns // 2
    if columns == 2880:
        physical = values.as_strided((2, stride), (stride, 1))
        assert torch.count_nonzero(physical[:, 1530:]).item() == 0
    quantize_input(torch.zeros_like(x), out=(values, scales))
    assert torch.count_nonzero(values).item() == torch.count_nonzero(scales).item() == 0
    empty, empty_scales = quantize_input(x[:0], out=(values[:0], scales[:0]))
    assert empty.shape[0] == empty_scales.shape[0] == 0


def make_scale_probe(config):
    import avelang
    import avelang.language as al
    from avelang_kernels.amdgpu.mega_moe.input_mxfp4_packed import make_mxfp4_packed_input

    BM, WORDS, THREADS = config.stage1_tile_m, config.stage1_lds_words, config.stage1_num_warps * 64
    ROW_BYTES = config.input_token_bytes
    prepare_scales, _, _ = make_mxfp4_packed_input(config, config.solution.hidden, config.input_token_bytes, 16)

    @avelang.jit
    def probe(x: al.Pointer(al.u8), output: al.Pointer(al.u32)):
        tid = al.convert(al.thread_id(0), al.u32)
        data = al.make_tensor(x, al.u8, al.make_layout((BM * ROW_BYTES,), (1,)))
        out = al.make_tensor(output, al.u32, al.make_layout((WORDS,), (1,)))
        resource = al.amdgpu.make_rsrc(data, BM * ROW_BYTES)
        storage = al.make_shared((WORDS,), al.u32)
        prepare_scales(resource, storage, al.convert(0, al.u32), tid)
        for index in al.range(tid, WORDS, THREADS):
            out[index] = storage[index]

    return probe


@pytest.mark.skipif(not has_gpus(1), reason="gfx950 required")
@pytest.mark.parametrize("model,tokens", [("gptoss120b", 8), ("gptoss120b", 256), ("gptoss120b", 1024), ("dsv32", 128)])
def test_scale_prefetch_inactive_waves_do_not_clobber_valid_scales(model, tokens):
    if model == "gptoss120b":
        cfg = get_2stage_cfgs(tokens, 8, 128, 4, 2880, 3072, activation="swiglu", bias_dtype="bf16")
    else:
        cfg = get_2stage_cfgs(tokens, 8, 256, 8, 7168, 2048, activation="silu", bias_dtype="none")
    probe = make_scale_probe(cfg)
    rows, d = cfg.stage1_tile_m, cfg.solution.hidden
    scale_bytes = cfg.input_token_bytes - d // 2
    expected = (torch.arange(rows * scale_bytes, device="cuda").reshape(rows, scale_bytes) % 254 + 1).to(torch.uint8)
    x = torch.zeros((rows, cfg.input_token_bytes), dtype=torch.uint8, device="cuda")
    x[:, d // 2 :] = expected
    output = torch.empty(cfg.stage1_lds_words, dtype=torch.int32, device="cuda")
    probe[lambda: ((1, 1, 1), (cfg.stage1_num_warps * 64, 1, 1))](x, output, num_warps=cfg.stage1_num_warps)
    actual, expected = output.cpu().view(torch.uint8), expected.cpu()
    lane = torch.arange(64)
    for tile in range(cfg.compute_hidden // 256):
        for m32 in range(rows // 32):
            row, col = m32 * 32 + lane % 16, tile * 8 + lane // 16
            want = torch.stack(
                (expected[row, col], expected[row + 16, col], expected[row, col + 4], expected[row + 16, col + 4]),
                dim=1,
            )
            start = ((tile % 2) * cfg.stage1_input_stage_words + rows * 32 + (tile // 2 * (rows // 32) + m32) * 64) * 4
            assert torch.equal(actual[start : start + 256].reshape(64, 4), want)
