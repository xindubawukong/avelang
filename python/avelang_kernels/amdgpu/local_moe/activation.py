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
def situ_v2_exponents(gate: al.f32, up: al.f32) -> (al.f32, al.f32):
    """Independent exponentials that Stage1 can issue across its whole tile."""
    eg = al.exp2(al.abs(gate) * al.convert(-0.7213475204444817, al.f32))
    eu = al.exp2(al.abs(up) * al.convert(-0.11541560327111708, al.f32))
    return eg, eu


@avelang.jit
def situ_v2_finish(gate: al.f32, up: al.f32, eg: al.f32, eu: al.f32) -> al.f32:
    """Finish SiTU from precomputed exponentials, preserving HIP rounding."""
    eg2 = eg * eg
    one = al.convert(1.0, al.f32)
    numerator = (one - eg) * (one - eu)
    # HIP contracts eg*eg + 1 in Petit's denominator. Preserve that single
    # rounding explicitly; a one-ulp difference can cross an FP4 midpoint.
    denominator = (one + eg) * al.fma(eg, eg, one) * (one + eu)
    positive = al.convert(100.0, al.f32) * numerator * al.amdgpu.rcp(denominator)
    gated = al.select(gate > al.convert(0.0, al.f32), positive, -positive * eg2)
    return al.select(up > al.convert(0.0, al.f32), gated, -gated)


@avelang.jit
def situ_v2(gate: al.f32, up: al.f32) -> al.f32:
    eg, eu = situ_v2_exponents(gate, up)
    return situ_v2_finish(gate, up, eg, eu)
