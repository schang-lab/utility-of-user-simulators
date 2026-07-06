import os
from multiprocessing import cpu_count as mp_cpu_count

from datasets import load_dataset, Features, Sequence, Value

_features = Features({
    "input_ids": Sequence(Value("int64")),
    "labels": Sequence(Value("int64")),
})

def get_preprocessed_distillation(dataset_config, tokenizer, split, override_datapath, model_name_string):
    """
    Copy-paste of the userlm_dataste.py::get_preprocessed_flipped_dialogue()
    """
    # Map llama-cookbook split names to our split names
    split_mapping = {
        'train': 'train',
        'validation': 'val',
        'test': 'test'
    }
    _max_context_length = dataset_config.max_length
    partition = split_mapping.get(split, split)
    load_path = os.path.join(
        override_datapath,
        f"{partition}_{model_name_string}_{dataset_config.unique_hash}_samples.jsonl"
    )
    ds = load_dataset("json",data_files=load_path,features=_features,split="train")
    print(f"Loaded {partition} dataset with {len(ds)} samples from {load_path}")
    ds = ds.filter(
        lambda ex: len(ex["input_ids"]) <= _max_context_length,
        num_proc=mp_cpu_count()
    )
    print(f"Filtered to {len(ds)} samples with max length {_max_context_length}")

    def _add_attn_mask(ex):
        n = len(ex["input_ids"])
        ex["attention_mask"] = [1] * n
        return ex
    
    ds = ds.map(_add_attn_mask, num_proc=mp_cpu_count())
    return ds
