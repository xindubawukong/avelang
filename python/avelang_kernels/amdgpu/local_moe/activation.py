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
