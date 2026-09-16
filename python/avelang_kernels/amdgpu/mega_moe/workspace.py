"""Byte-exact layout of Petit's VMM workspace and direct-push epoch records."""

from dataclasses import dataclass

from .config import MegaMoeConfig


def align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class WorkspaceLayout:
    config: MegaMoeConfig

    def __post_init__(self):
        s = self.config.solution
        ranks, experts, local, topk = s.world_size, s.experts, self.config.local_experts, s.topk
        cap = self.config.max_tokens_per_rank
        pool = align_up(ranks * cap * min(topk, local) + local * 31, 32)
        fields = {
            "pool_rows": pool,
            "pool_blocks": pool // 32,
            "scale_rows": align_up(pool, 256),
            "scale_cols": (s.intermediate + 255) // 256 * 8,
            "barrier_record_bytes": 8192 if experts > 128 else 4096,
        }
        # Offsets relative to the beginning of a rank slot.
        cursor = 0
        for name, size in (
            ("send_counts", experts * 8),
            ("recv_counts", ranks * local * 8),
            ("recv_sum", local * 8),
            ("recv_tokens", ranks * local * cap * 4),
            ("input_weights", cap * topk * 4),
            ("input_tokens", cap * self.config.input_token_bytes),
            ("route_output", cap * topk * s.hidden * 2),
            ("route_ready", 2 * cap * 4),
            ("l1_ready", pool // 32 * 4),
            ("metadata", pool * 8),
            ("l1_tokens", pool * self.config.input_token_bytes),
            ("l1_weights", pool * 4),
        ):
            fields[name] = cursor
            cursor += size
        fields["rank_sym_buffer_base"] = ranks * fields["barrier_record_bytes"]
        fields["rank_slot_bytes"] = align_up(cursor, fields["rank_sym_buffer_base"])
        fields["local_offset"] = align_up(
            fields["rank_sym_buffer_base"] + ranks * fields["rank_slot_bytes"], 2 * 1024 * 1024
        )
        cursor = fields["local_offset"]
        for name, size in (
            ("grid_sync", 128),
            ("work_heads", 1024),
            ("input_ids", cap * topk * 4),
            ("l2_ready", pool // 32 * 4),
            ("l2_tokens", pool * s.intermediate // 2),
            ("l2_scales", fields["scale_rows"] * fields["scale_cols"]),
        ):
            fields[name] = cursor
            cursor += size
        fields["local_bytes"] = cursor - fields["local_offset"]
        fields["workspace_bytes"] = cursor
        # Offsets relative to each rank's independently owned barrier record.
        cursor = 384
        for name, size in (
            ("entry_count", 256 * 4),
            ("plan_base", experts * 8),
            ("count_done", 2 * ranks * 4),
            ("plan_ready", 2 * ranks * 4),
            ("payload_ready", 2 * local * 4),
            ("epoch_gate", 4),
            ("launch_ready", ranks * 4),
        ):
            fields[name] = cursor
            cursor += size
        if cursor > fields["barrier_record_bytes"] or fields["workspace_bytes"] >= 2**32:
            raise ValueError("MegaMoE workspace exceeds its control record or 4 GiB address range")
        for name, value in fields.items():
            object.__setattr__(self, name, value)

    def rank_base(self, rank):
        return self.rank_sym_buffer_base + rank * self.rank_slot_bytes
