# Adapted from: https://github.com/huggingface/trl/blob/2cad48d511fab99ac0c4b327195523a575afcad3/examples/scripts/sft.py
# flake8: noqa
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
# regular:
code/causal-language-modeling-qlora/train.py \
    --model_name_or_path="facebook/opt-350m" \
    --dataset_text_field="text" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing

# peft:
code/causal-language-modeling-qlora/train.py \
    --model_name_or_path="facebook/opt-350m" \
    --dataset_text_field="text" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing \
    --use_peft \
    --lora_r=64 \
    --lora_alpha=16
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.commands.cli_utils import SFTScriptArguments, TrlParser

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.experiment_tracking import set_wandb


@dataclass
class AdditionalArguments:
    """
    Additional arguments that are not part of the TRL arguments.
    """

    tokenized_dataset_load_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Directory to load the tokenized dataset using the "
                "`datasets.load_from_disk` method. No tokenization will be done"
                " the dataset will be loaded from this directory."
            )
        },
    )


if __name__ == "__main__":
    parser = TrlParser(
        (SFTScriptArguments, SFTConfig, ModelConfig, AdditionalArguments)
    )
    args, training_args, model_config, additional_args = parser.parse_args_and_config()
    set_wandb(training_args)
    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    if additional_args.tokenized_dataset_load_dir:
        dataset = datasets.load_from_disk(additional_args.tokenized_dataset_load_dir)
        skip_prepare_dataset = True
        if training_args.dataset_kwargs is None:
            training_args.dataset_kwargs = {}
        training_args.dataset_kwargs["skip_prepare_dataset"] = True
    else:
        dataset = load_dataset(args.dataset_name)

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        args=training_args,
        train_dataset=dataset[args.dataset_train_split],
        eval_dataset=dataset[args.dataset_test_split],
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_config),
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
