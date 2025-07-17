# Copyright (c) 2025 FlexAI
# This file is part of the FlexAI Experiments repository.
# SPDX-License-Identifier: MIT

import warnings
from dataclasses import dataclass, field
from typing import Optional

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer
from trl import ModelConfig, SFTConfig
from trl.commands.cli_utils import SFTScriptArguments, TrlParser


@dataclass
class AdditionalArguments:
    """
    Additional arguments that are not part of the TRL arguments.
    """

    tokenized_dataset_save_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Directory to save the tokenized dataset to be loaded using "
                "the `datasets.load_from_disk` method."
            )
        },
    )


def _prepare_dataset(
    tokenizer,
    dataset,
    dataset_text_field,
    max_seq_length,
    add_special_tokens=True,
    remove_unused_columns=True,
    dataset_batch_size=1000,
    dataset_num_proc=1,
):
    def tokenize(element):
        outputs = tokenizer(
            element[dataset_text_field],
            add_special_tokens=add_special_tokens,
            truncation=True,
            padding=False,
            max_length=max_seq_length,
            return_overflowing_tokens=False,
            return_length=False,
        )

        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }

    if dataset_num_proc > 1:
        warnings.warn(
            "`num_proc > 1` is not supported yet for dataset caching (as it creates one tokenized files per proc)."
        )

    map_kwargs = {
        "batched": True,
        "remove_columns": dataset.column_names if remove_unused_columns else None,
        "batch_size": dataset_batch_size,
        "num_proc": dataset_num_proc,
    }
    tokenized_dataset = dataset.map(tokenize, **map_kwargs)

    return tokenized_dataset


if __name__ == "__main__":
    parser = TrlParser(
        (SFTScriptArguments, SFTConfig, ModelConfig, AdditionalArguments)
    )
    args, training_args, model_config, additional_args = parser.parse_args_and_config()

    ################
    # Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    dataset = load_dataset(args.dataset_name)

    train_dataset = dataset[args.dataset_train_split]
    eval_dataset = dataset[args.dataset_test_split]
    tokenized_datasets = DatasetDict()

    training_args.max_seq_length = min(tokenizer.model_max_length, 1024)
    if training_args.dataset_kwargs is None:
        training_args.dataset_kwargs = {}

    if train_dataset is not None:
        train_dataset = _prepare_dataset(
            tokenizer,
            train_dataset,
            training_args.dataset_text_field,
            training_args.max_seq_length,
            **training_args.dataset_kwargs,
        )
        tokenized_datasets[args.dataset_train_split] = train_dataset
    if eval_dataset is not None:
        _multiple = isinstance(eval_dataset, dict)
        _eval_datasets = eval_dataset if _multiple else {"singleton": eval_dataset}

        for _eval_dataset_name, _eval_dataset in _eval_datasets.items():
            _eval_datasets[_eval_dataset_name] = _prepare_dataset(
                tokenizer,
                _eval_dataset,
                training_args.dataset_text_field,
                training_args.max_seq_length,
                **training_args.dataset_kwargs,
            )
        if not _multiple:
            eval_dataset = _eval_datasets["singleton"]
        tokenized_datasets[args.dataset_test_split] = eval_dataset
    tokenized_datasets.save_to_disk(additional_args.tokenized_dataset_save_dir)
