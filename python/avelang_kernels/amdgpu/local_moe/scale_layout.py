"""Native E8M0 scale layout shared by inputs, weights and intermediates.

Shape, device byte addressing and host decoding describe the same order:
[row32, K256, K32, row16, K128, row-half].
"""

import torch


def ceildiv(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def scale_byte_shape(rows: int, k: int) -> tuple[int, ...]:
    # [row32 tile, K256 tile, K32 lane group, row lane, K128, row16]
    # Packing the last two dimensions into u32 makes scale selection 2*k+n.
    return ceildiv(rows, 32), ceildiv(k, 256), 4, 16, 2, 2


def unsort_scales(scales: torch.Tensor, rows: int, columns: int) -> torch.Tensor:
    """Decode native scale tiles into logical rows for inspection/reference checks."""
    return (
        scales.view(torch.uint8)
        .reshape(ceildiv(rows, 32), ceildiv(columns, 256), 4, 16, 2, 2)
        .permute(0, 5, 3, 1, 4, 2)
        .reshape(ceildiv(rows, 32) * 32, ceildiv(columns, 256) * 8)[:rows, : columns // 32]
    )
