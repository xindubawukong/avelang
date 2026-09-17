"""Device activation functions matching Petit's local MoE epilogue."""

import avelang
import avelang.language as al


@avelang.jit
def silu_dot(gate: al.f32, up: al.f32) -> al.f32:
    coefficient = al.convert(-1.4426950408889634, al.f32)
    sigmoid = al.amdgpu.rcp(al.convert(1.0, al.f32) + al.exp2(coefficient * gate))
    return gate * sigmoid * up


@avelang.jit
def openai_swiglu(gate: al.f32, up: al.f32) -> al.f32:
    limit = al.convert(7.0, al.f32)
    gc = al.select(gate < limit, gate, limit)
    uc = al.select(up > -limit, up, -limit)
    uc = al.select(uc < limit, uc, limit)
    uc = uc + al.convert(1.0, al.f32)
    coefficient = al.convert(-2.455307455790015, al.f32)
    sigmoid = al.amdgpu.rcp(al.convert(1.0, al.f32) + al.exp2(coefficient * gc))
    return gc * sigmoid * uc


@avelang.jit
def situ_v2(gate: al.f32, up: al.f32) -> al.f32:
    """Direct 100*tanh(gate/4)*sigmoid(gate)*tanh(up/25)."""
    one = al.convert(1.0, al.f32)
    eg = al.exp2(-al.abs(gate) * al.convert(0.7213475204444817, al.f32))
    eu = al.exp2(-al.abs(up) * al.convert(0.11541560327111708, al.f32))
    tg = (one - eg) * al.amdgpu.rcp(one + eg)
    tu = (one - eu) * al.amdgpu.rcp(one + eu)
    tg = al.select(gate < 0.0, -tg, tg)
    tu = al.select(up < 0.0, -tu, tu)
    sigmoid = al.amdgpu.rcp(one + al.exp2(-gate * al.convert(1.4426950408889634, al.f32)))
    return al.convert(100.0, al.f32) * tg * sigmoid * tu
