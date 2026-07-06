# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from pathlib import Path
from datetime import datetime
import torch
import time

import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)


def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run


def load_model_sharded(model, rank, cfg):
    """Load a sharded checkpoint into the model using DCP."""
    folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        + "-"
        + cfg.model_name.replace("/", "--")
    )

    load_dir = Path.cwd() / folder_name

    if not load_dir.exists():
        if rank == 0:
            print(f"No sharded_state_dict checkpoint directory found...skipping")
        return
    if rank == 0:
        print(f"loading model from model path: {load_dir} ")

    model_state_dict = get_model_state_dict(model)
    dcp.load({"model": model_state_dict}, checkpoint_id=str(load_dir))
    set_model_state_dict(model, model_state_dict)

    if rank == 0:
        print(f"Sharded state checkpoint loaded from {load_dir}")


def save_model_and_optimizer_sharded(model, rank, cfg, optim=None, step=None):
    """Save model and optimizer using DCP (distributed checkpoint)."""

    folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        + "-"
        + cfg.model_name.replace("/", "--")
    )

    # Add step suffix if saving at a specific step
    if step is not None:
        folder_name += f"-step-{step}"

    save_dir = Path.cwd() / folder_name
    if rank == 0:
        print(f"Saving model to {save_dir}")

    t0 = time.perf_counter()

    state_dict = {"model": get_model_state_dict(model)}
    if optim is not None:
        state_dict["optim"] = get_optimizer_state_dict(model, optim)

    dcp.save(state_dict, checkpoint_id=str(save_dir))

    dist.barrier()
    t1 = time.perf_counter()
    if rank == 0:
        print(f"Sharded state checkpoint saved to {save_dir}")
        print(f"Checkpoint Time = {t1-t0:.4f}\n")


def save_fsdp_model_checkpoint_full(
    model,
    optimizer,
    rank,
    cfg,
    epoch=None,
    step=None,
):
    """Save model via rank0 cpu streaming using full state dict."""

    # Get full state dict on rank 0
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    cpu_state = get_model_state_dict(model, options=options)

    print(f"saving process: rank {rank}  done w model state_dict\n")

    if rank == 0:
        print(f"--> saving model ...")
        # create save path
        folder_name = (
            cfg.dist_checkpoint_root_folder
            + "/"
            + cfg.dist_checkpoint_folder
            + "-"
            + cfg.model_name.replace("/", "--")
        )
        save_dir = Path.cwd() / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)

        # Use step if provided, otherwise use epoch
        if step is not None:
            save_name = cfg.model_name.replace("/", "--") + "-step-" + str(step) + ".pt"
        else:
            save_name = cfg.model_name.replace("/", "--") + "-" + str(epoch) + ".pt"
        save_full_path = str(save_dir) + "/" + save_name

        # save model
        torch.save(cpu_state, save_full_path)

        if step is not None:
            print(f"model checkpoint saved at step {step} at {save_full_path}\n")
        else:
            print(f"model checkpoint saved for epoch {epoch} at {save_full_path}\n")


def load_model_checkpoint(model, rank, cfg):
    """load local checkpoint to rank0 cpu
    must be called * before * passing to FSDP"""

    if rank != 0:
        return

    # where is the checkpoint at...
    full_state_dict_model_path = (
        Path.cwd() / cfg.checkpoint_folder / cfg.checkpoint_model_filename
    )
    # is it present...
    if not full_state_dict_model_path.is_file():
        print(
            f"model checkpoint {full_state_dict_model_path} not present. Returning..."
        )
        return

    model_checkpoint = torch.load(full_state_dict_model_path)
    # integrate into loaded model
    model.load_state_dict(model_checkpoint)

    print(f"model checkpoint loaded to rank0 cpu")


def save_optimizer_checkpoint(model, optimizer, rank, cfg, epoch=None, step=None):
    """Save optimizer state via full state dict."""

    print(f"--> optim state call on rank {rank}\n")

    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    optim_state = get_optimizer_state_dict(model, optimizer, options=options)

    print(f"optim state dict ready on {rank} and len of {len(optim_state)}\n")

    if rank == 0:
        folder_name = (
            cfg.dist_checkpoint_root_folder
            + "/"
            + cfg.dist_checkpoint_folder
            + "-"
            + cfg.model_name.replace("/", "--")
        )
        save_dir = Path.cwd() / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)

        # Use step if provided, otherwise use epoch
        if step is not None:
            opt_save_name = "optimizer" + "-" + cfg.model_name.replace("/", "--") + "-step-" + str(step) + ".pt"
        else:
            opt_save_name = "optimizer" + "-" + cfg.model_name.replace("/", "--") + "-" + str(epoch) + ".pt"
        opt_save_full_path = save_dir / opt_save_name

        print(f"--> saving optimizer state...")

        torch.save(optim_state, opt_save_full_path)

        print(f"--> saved {opt_save_full_path} to disk")


def load_optimizer_checkpoint(model, optimizer_checkpoint_path, rank):
    """load an optimizer checkpoint"""

    if not optimizer_checkpoint_path.is_file():
        print(
            f"warning - optimizer checkpoint not present {optimizer_checkpoint_path}. Returning. "
        )
        return

    full_osd = None

    if rank == 0:
        full_osd = torch.load(optimizer_checkpoint_path)

    # For FSDP2, use set_optimizer_state_dict
    # Note: This requires the optimizer to already exist
    print(f"optimizer shard loaded on rank {rank}")


def load_sharded_model_single_gpu(model, model_path):

    state_dict = {"model": model.state_dict()}

    dcp.load(
        state_dict=state_dict,
        checkpoint_id=str(model_path),
    )

    model.load_state_dict(state_dict["model"])

    print(f"Sharded state checkpoint loaded from {model_path}")
    return model


def save_peft_checkpoint(model, model_path):
    """save_pretrained peft model"""

    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    state_dict = get_model_state_dict(model, options=options)
    model.save_pretrained(model_path, state_dict=state_dict)


def save_model_checkpoint(model, output_dir):
    """save model when not peft and on single device"""

    output_file = Path(output_dir) / "model.pt"

    state_dict = model.state_dict()

    torch.save(state_dict, output_file)
