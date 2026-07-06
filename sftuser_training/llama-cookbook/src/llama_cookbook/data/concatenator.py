# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from tqdm import tqdm
from itertools import chain

from torch.utils.data import Dataset
from torch.nn import functional as F


class BucketPaddingCollator:
    """Wraps a base collator and pads each batch to the nearest power-of-2 bucket.

    Without this, torch.compile sees a new (batch, seq_len) shape on every batch and recompiles each time.
    With this, it sees at most len(buckets) distinct shapes and specializes once per bucket — giving full static-shape speedup.

    Pads right-side: input_ids with pad_token_id, attention_mask with 0, labels with -100.
    If seq_len exceeds the largest bucket, no extra padding is added.
    """
    # BUCKETS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    BUCKETS = [4096]

    def __init__(self, base_collator, pad_token_id):
        self.base_collator = base_collator
        self.pad_token_id = pad_token_id

    def _next_bucket(self, length: int) -> int:
        for b in self.BUCKETS:
            if length <= b:
                return b
        assert False, f"Sequence length {length} exceeds largest bucket {self.BUCKETS[-1]}"

    def __call__(self, features):
        batch = self.base_collator(features)
        seq_len = batch["input_ids"].shape[1]
        target_len = self._next_bucket(seq_len)
        pad_len = target_len - seq_len
        if pad_len == 0:
            return batch
        batch["input_ids"] = F.pad(batch["input_ids"], (0, pad_len), value=self.pad_token_id)
        batch["attention_mask"] = F.pad(batch["attention_mask"], (0, pad_len), value=0)
        batch["labels"] = F.pad(batch["labels"], (0, pad_len), value=-100)
        return batch


class ConcatDataset(Dataset):
    def __init__(self, dataset, chunk_size=4096):
        self.dataset = dataset
        self.chunk_size = chunk_size

        self.samples = []

        buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            }

        # original version that cuts in between of samples
        # for sample in tqdm(self.dataset, desc="Preprocessing dataset", dynamic_ncols=True):
        #     buffer = {k: v + sample[k] for k,v in buffer.items()}
        # 
        #     while len(next(iter(buffer.values()))) > self.chunk_size:
        #         self.samples.append({k: v[:self.chunk_size] for k,v in buffer.items()})
        #         buffer = {k: v[self.chunk_size:] for k,v in buffer.items()}

        # custom version to avoid the cut in between of samples
        for sample in tqdm(self.dataset, desc="Preprocessing dataset", dynamic_ncols=True):
            # assumption: new addition to the buffer
            # does not result int exceeding 2 * chunk_size
            new_buffer = {k: v + sample[k] for k,v in buffer.items()}
            if len(next(iter(new_buffer.values()))) >= self.chunk_size:
                assert len(next(iter(new_buffer.values()))) < 2 * self.chunk_size, (
                    "sample too large"
                )
                self.samples.append({k: v for k,v in buffer.items()})
                buffer = {k: sample[k] for k in sample.keys()}
            else:
                buffer = new_buffer

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)
