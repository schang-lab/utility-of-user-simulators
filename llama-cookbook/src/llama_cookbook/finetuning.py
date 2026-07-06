# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import dataclasses
import os
import random
from collections import Counter
from warnings import warn

import fire
import numpy as np
import torch
import torch.optim as optim
from accelerate.utils import is_xpu_available

from llama_cookbook.configs import (
    fsdp_config as FSDP_CONFIG,
    quantization_config as QUANTIZATION_CONFIG,
    train_config as TRAIN_CONFIG,
)
from llama_cookbook.data.concatenator import ConcatDataset

from llama_cookbook.utils.config_utils import (
    check_fsdp_config,
    generate_dataset_config,
    generate_peft_config,
    get_dataloader_kwargs,
    update_config,
)
from llama_cookbook.utils.dataset_utils import (
    get_custom_data_collator,
    get_preprocessed_dataset,
)
from llama_cookbook.data.concatenator import BucketPaddingCollator

from llama_cookbook.utils.train_utils import (
    clear_gpu_cache,
    freeze_transformer_layers,
    freeze_LLM_only,
    print_model_size,
    print_frozen_model_status,
    setup,
    setup_environ_flags,
    train,
    CosineWarmupStableLRScheduler,
)
from peft import get_peft_model, PeftModel
from torch.optim.lr_scheduler import StepLR
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    MllamaForConditionalGeneration,
    Qwen2ForCausalLM,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mllama.modeling_mllama import (
    MllamaCrossAttentionDecoderLayer,
    MllamaSelfAttentionDecoderLayer,
    MllamaVisionEncoderLayer,
)
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

# FSDP2 imports
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy


def setup_wandb(train_config, fsdp_config, **kwargs):
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "You are trying to use wandb which is not currently installed. "
            "Please install it using pip install wandb"
        )
    from llama_cookbook.configs import wandb_config as WANDB_CONFIG

    wandb_config = WANDB_CONFIG()
    update_config(wandb_config, **kwargs)
    init_dict = dataclasses.asdict(wandb_config)
    run = wandb.init(**init_dict)
    run.config.update(train_config)
    run.config.update(fsdp_config, allow_val_change=True)
    return run


def _get_decoder_layer_types(config):
    """Return the decoder layer classes to shard for a given model type."""
    if config.model_type == "qwen2":
        return (Qwen2DecoderLayer,)
    elif config.model_type == "llama":
        return (LlamaDecoderLayer,)
    elif config.model_type == "mllama":
        return (
            MllamaSelfAttentionDecoderLayer,
            MllamaCrossAttentionDecoderLayer,
            MllamaVisionEncoderLayer,
        )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")


def _apply_fsdp2(model, config, fsdp_config, rank, mp_policy):
    """
    Apply FSDP2 fully_shard to each decoder layer, then to the root model.
    This is the FSDP2 equivalent of wrapping with auto_wrap_policy.
    """
    decoder_layer_types = _get_decoder_layer_types(config)

    # Shard each decoder layer individually (analogous to transformer_auto_wrap_policy)
    for module in model.modules():
        if isinstance(module, decoder_layer_types):
            fully_shard(module, mp_policy=mp_policy, reshard_after_forward=True)

    # Shard the root model
    fully_shard(model, mp_policy=mp_policy, reshard_after_forward=True)

    return model


def main(**kwargs):
    # Update the configuration for the training and sharding process
    train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
    update_config((train_config, fsdp_config), **kwargs)
    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)

    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        if is_xpu_available():
            torch.xpu.set_device(local_rank)
        elif torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    wandb_run = None

    if train_config.use_wandb:
        if not train_config.enable_fsdp or rank == 0:
            wandb_run = setup_wandb(train_config, fsdp_config, **kwargs)

     # setting quantization configs
    bnb_config = None
    if train_config.quantization:
        if type(train_config.quantization) == type(True):
            warn(
                "Quantization (--quantization) is a boolean, please specify quantization as '4bit' or '8bit'. Defaulting to '8bit' but this might change in the future.",
                FutureWarning,
            )
            train_config.quantization = "8bit"

        if train_config.quantization == "8bit" and train_config.enable_fsdp:
            raise ValueError(
                "8bit quantization is not supported with FSDP, please use 4bit quantization"
            )

        quant_config = QUANTIZATION_CONFIG()
        update_config(quant_config, **kwargs)
        bnb_config = quant_config.create_bnb_config(train_config.quantization)


        if train_config.enable_fsdp:
            if train_config.quantization == "4bit":
                bnb_config.bnb_4bit_quant_storage = bnb_config.bnb_4bit_compute_dtype
                from logging import getLogger
                logger = getLogger()
                logger.warning(
                    "FSDP and 4-bit QLoRA enabled. Setting `bnb_4bit_quant_storage` "
                    f"to {bnb_config.bnb_4bit_compute_dtype} for compatibility."
                )

    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp else None
    config = AutoConfig.from_pretrained(train_config.model_name)
    if config.model_type == "mllama":
        is_vision = True
        model = MllamaForConditionalGeneration.from_pretrained(
            train_config.model_name,
            quantization_config=bnb_config,
            attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            device_map=(
                "auto"
                if train_config.quantization and not train_config.enable_fsdp
                else None
            ),
            torch_dtype=torch.float16 if train_config.use_fp16 else "auto",
        )
        processor = AutoProcessor.from_pretrained(
            train_config.model_name
            if train_config.tokenizer_name is None
            else train_config.tokenizer_name
        )
        processor.tokenizer.padding_side = "right"
        model.supports_gradient_checkpointing = True
        model.language_model.supports_gradient_checkpointing = True
    elif config.model_type == "llama":
        is_vision = False
        model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            quantization_config=bnb_config,
            use_cache=use_cache,
            attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            device_map=(
                "auto"
                if train_config.quantization and not train_config.enable_fsdp
                else None
            ),
            torch_dtype=torch.float16 if train_config.use_fp16 else "auto",
        )
    elif config.model_type == "qwen2":  # optional: qwen2 support
        is_vision = False
        if train_config.enable_fsdp and train_config.low_cpu_fsdp:
            if rank == 0:
                model = Qwen2ForCausalLM.from_pretrained(
                    train_config.model_name,
                    quantization_config=bnb_config,
                    use_cache=use_cache,
                    attn_implementation="sdpa" if train_config.use_fast_kernels else None,
                    device_map="auto" if train_config.quantization and not train_config.enable_fsdp else None,
                    torch_dtype=torch.float16 if train_config.use_fp16 else torch.bfloat16,
                )
            else:
                qwen2_cfg = AutoConfig.from_pretrained(train_config.model_name)
                qwen2_cfg.use_cache = use_cache
                with torch.device("meta"):
                    model = Qwen2ForCausalLM(qwen2_cfg)
        else:
            model = Qwen2ForCausalLM.from_pretrained(
                train_config.model_name,
                quantization_config=bnb_config,
                use_cache=use_cache,
                attn_implementation="sdpa" if train_config.use_fast_kernels else None,
                device_map="auto" if train_config.quantization and not train_config.enable_fsdp else None,
                torch_dtype=torch.float16 if train_config.use_fp16 else torch.bfloat16,
            )
    else:
        raise ValueError(
            f"Model type {config.model_type} is not supported. Please use llama, mllama, qwen2 model."
        )
    # Load the tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(
        train_config.model_name
        if train_config.tokenizer_name is None
        else train_config.tokenizer_name
    )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # If there is a mismatch between tokenizer vocab size and embedding matrix,
    # throw a warning and then expand the embedding matrix
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print(
            "WARNING: Resizing the embedding matrix to match the tokenizer vocab size."
        )
        model.resize_token_embeddings(len(tokenizer))
        # custom: initialize to eos token
        with torch.no_grad():
            model.get_input_embeddings().weight[-1].copy_(
                model.get_input_embeddings().weight[tokenizer.eos_token_id]
            )
            model.get_output_embeddings().weight[-1].copy_(
                model.get_output_embeddings().weight[tokenizer.eos_token_id]
            )

    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if (
        train_config.enable_fsdp
        and fsdp_config.pure_bf16
        and not train_config.quantization
    ):
        model.to(torch.bfloat16)

    if train_config.use_peft:
        # Load the pre-trained peft model checkpoint and setup its configuration
        if train_config.from_peft_checkpoint:
            model = PeftModel.from_pretrained(
                model, train_config.from_peft_checkpoint, is_trainable=True
            )
            peft_config = model.peft_config
        # Generate the peft config and start fine-tuning from original model
        else:
            peft_config = generate_peft_config(train_config, kwargs)
            model = get_peft_model(model, peft_config)
        if wandb_run:
            wandb_run.config.update(peft_config)
        model.print_trainable_parameters()

    # setting up FSDP2 if enable_fsdp is enabled
    if train_config.enable_fsdp:
        check_fsdp_config(fsdp_config)

        if not train_config.use_peft and train_config.freeze_layers:
            freeze_transformer_layers(model, train_config.num_freeze_layers)
            print_frozen_model_status(model, train_config, rank if train_config.enable_fsdp else 0)

        if not train_config.use_peft and train_config.freeze_LLM_only and config.model_type == "mllama":
            freeze_LLM_only(model)
            print_frozen_model_status(model, train_config, rank if train_config.enable_fsdp else 0)

        # Build FSDP2 MixedPrecisionPolicy
        if fsdp_config.pure_bf16:
            mp_policy = MixedPrecisionPolicy()  # no casting, model already in bf16
        elif fsdp_config.mixed_precision:
            if fsdp_config.use_fp16:
                mp_policy = MixedPrecisionPolicy(
                    param_dtype=torch.float16,
                    reduce_dtype=torch.float16,
                )
            else:
                mp_policy = MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16,
                )
        else:
            mp_policy = MixedPrecisionPolicy()

        # Move model to GPU before sharding
        # FSDP2 requires all parameters to be materialized (not on meta device)
        # before fully_shard is called.
        if train_config.low_cpu_fsdp and rank != 0:
            # Non-rank-0 processes have meta-device params; materialize to empty
            # GPU tensors. FSDP2 will scatter rank 0's values during first all-gather.
            model.to_empty(device=f"cuda:{torch.cuda.current_device()}")
        else:
            model.to(f"cuda:{torch.cuda.current_device()}")

        # Apply activation checkpointing BEFORE FSDP2 sharding
        # Use HuggingFace's built-in gradient checkpointing which is compatible with torch.compile + FSDP2
        if fsdp_config.fsdp_activation_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if rank == 0:
                print("Gradient checkpointing enabled (non-reentrant).")

        # Apply FSDP2 sharding
        _apply_fsdp2(model, config, fsdp_config, rank, mp_policy)

        if rank == 0:
            print("FSDP2 sharding applied successfully.")

    elif not train_config.quantization and not train_config.enable_fsdp:
        if is_xpu_available():
            model.to("xpu:0")
        elif torch.cuda.is_available():
            model.to("cuda")

    # torch.compile — applied after FSDP2 sharding.
    # With the default skip_fsdp_hooks=True, dynamo creates per-layer compiled graphs
    # (graph break at each FSDP hook boundary) while FSDP2 handles comm/compute overlap
    # via CUDA streams. Per-layer compile (layer.compile()) is not used because gradient
    # checkpointing re-runs each layer during backward under a different autograd context,
    # which trips dynamo guards and causes recompilation every step.
    if train_config.use_torch_compile:
        if not train_config.enable_fsdp or rank == 0:
            print("Compiling model with torch.compile...")
        model = torch.compile(model, dynamic=False)
        if not train_config.enable_fsdp or rank == 0:
            print("Model compiled successfully.")

    dataset_config = generate_dataset_config(train_config, kwargs)
    if is_vision:
        dataset_processer = processor
    else:
        dataset_processer = tokenizer

    # Load and preprocess the dataset for training and validation
    model_name_string = train_config.model_name.replace("/", "--")
    dataset_train = get_preprocessed_dataset(
        dataset_processer,
        dataset_config,
        split="train",
        override_datapath=train_config.dataset_path,
        model_name_string=model_name_string,
    )
    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    dataset_val = get_preprocessed_dataset(
        dataset_processer,
        dataset_config,
        split="val",
        override_datapath=train_config.dataset_path,
        model_name_string=model_name_string,
    )
    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Validation Set Length = {len(dataset_val)}")

    # if train_config.batching_strategy == "packing":
    # this is custom packing!
    if True:
        if is_vision:
            raise ValueError("Packing is not supported for vision datasets")
        else:
            dataset_train = ConcatDataset(
                dataset_train,
                chunk_size=max(train_config.context_length,
                               getattr(dataset_config, "max_length", 0)),
            )

    train_dl_kwargs = get_dataloader_kwargs(
        train_config, dataset_train, dataset_processer, "train"
    )
    print("length of dataset_train", len(dataset_train))
    custom_data_collator = get_custom_data_collator(dataset_processer, dataset_config)
    if custom_data_collator:
        print("custom_data_collator is used")
        train_dl_kwargs["collate_fn"] = custom_data_collator
    if train_config.use_torch_compile:
        if not train_config.enable_fsdp or rank == 0:
            print("BucketPaddingCollator enabled for torch.compile static-shape speedup.")
        train_dl_kwargs["collate_fn"] = BucketPaddingCollator(
            train_dl_kwargs["collate_fn"], pad_token_id=tokenizer.pad_token_id
        )
    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **train_dl_kwargs,
    )
    print(f"--> Num of Training Set Batches loaded = {len(train_dataloader)}")

    eval_dataloader = None
    if train_config.run_validation:
        # if train_config.batching_strategy == "packing":
        if True:
            if is_vision:
                raise ValueError("Packing is not supported for vision datasets")
            else:
                dataset_val = ConcatDataset(
                    dataset_val,
                    chunk_size=max(train_config.context_length,
                                   getattr(dataset_config, "max_length", 0)),
                )

        val_dl_kwargs = get_dataloader_kwargs(
            train_config, dataset_val, dataset_processer, "val"
        )
        if custom_data_collator:
            val_dl_kwargs["collate_fn"] = custom_data_collator
        if train_config.use_torch_compile:
            val_dl_kwargs["collate_fn"] = BucketPaddingCollator(
                val_dl_kwargs["collate_fn"], pad_token_id=tokenizer.pad_token_id
            )

        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **val_dl_kwargs,
        )
        print(f"--> Num of Validation Set Batches loaded = {len(eval_dataloader)}")
        if len(eval_dataloader) == 0:
            raise ValueError(
                f"The eval set size is too small for dataloader to load even one batch. Please increase the size of eval set. ({len(eval_dataloader)=})"
            )
        else:
            print(f"--> Num of Validation Set Batches loaded = {len(eval_dataloader)}")

    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        from llama_cookbook.policies import AnyPrecisionAdamW
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )

    total_train_steps = (
        len(train_dataloader) // train_config.gradient_accumulation_steps
    ) * train_config.num_epochs
    print("Total training steps:", total_train_steps)
    # Use cosine warmup scheduler if warmup_ratio > 0, otherwise fall back to StepLR
    if train_config.warmup_ratio > 0:
        scheduler = CosineWarmupStableLRScheduler(
            optimizer,
            warmup_steps=int(train_config.warmup_ratio * total_train_steps),
            total_steps=total_train_steps,
            min_lr=train_config.min_lr
        )
    else:
        scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if train_config.enable_fsdp else None,
        rank if train_config.enable_fsdp else None,
        wandb_run,
    )
    if not train_config.enable_fsdp or rank == 0:
        [print(f"Key: {k}, Value: {v}") for k, v in results.items()]
        if train_config.use_wandb:
            for k, v in results.items():
                wandb_run.summary[k] = v


if __name__ == "__main__":
    fire.Fire(main)
