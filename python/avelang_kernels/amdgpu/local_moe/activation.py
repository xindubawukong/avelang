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
    """Kimi K3 SiTU, matching Petit's ``SituV2Op`` scalar expression."""
    gate_exp = al.convert(-0.7213475204444817, al.f32)
    up_exp = al.convert(-0.11541560327111708, al.f32)
    eg = al.exp2(al.abs(gate) * gate_exp)
    eu = al.exp2(al.abs(up) * up_exp)
    eg2 = eg * eg
    numerator = (al.convert(1.0, al.f32) - eg) * (al.convert(1.0, al.f32) - eu)
    denominator = (al.convert(1.0, al.f32) + eg) * (al.convert(1.0, al.f32) + eg2) * (al.convert(1.0, al.f32) + eu)
    positive = al.convert(100.0, al.f32) * numerator * al.amdgpu.rcp(denominator)
    gated = al.select(gate > 0.0, positive, -positive * eg2)
    return al.select(up > 0.0, gated, -gated)
