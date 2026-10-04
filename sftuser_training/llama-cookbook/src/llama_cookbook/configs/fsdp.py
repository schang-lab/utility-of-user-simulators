# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class fsdp_config:
    mixed_precision: bool = True
    use_fp16: bool = False
    pure_bf16: bool = False  # disables mixed precision, and runs in pure bfloat16
    fsdp_activation_checkpointing: bool = True
    # Checkpoint type: "SHARDED_STATE_DICT" (default, uses DCP) or "FULL_STATE_DICT" (gathers to rank 0)
    checkpoint_type: str = "SHARDED_STATE_DICT"
    optimizer: str = "AdamW"
