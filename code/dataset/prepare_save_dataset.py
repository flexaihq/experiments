from dataclasses import dataclass, field
from functools import partial
from itertools import chain
from typing import Dict, Optional

import datasets
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser

MAX_BLOCK_SIZE_WHEN_NONE = 8192


@dataclass
class DatasetArguments:
    """
    Dataset related arguments.
    """

    dataset_name: str = field(
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the dataset configuration."},
    )
    tokenized_dataset_save_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Directory to save the tokenized dataset to be loaded using "
                "the `datasets.load_from_disk` method."
            )
        },
    )
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
    dataset_train_split: Optional[str] = field(
        default="train",
        metadata={"help": "Train split name of the dataset."},
    )
    dataset_test_split: Optional[str] = field(
        default="test",
        metadata={"help": "Test split name of the dataset."},
    )
    dataset_text_field: Optional[str] = field(
        default="text",
        metadata={"help": "Text field name of the dataset."},
    )
    dataset_max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Max sequence length for a dataset sample"
                "(the tokenizer truncates it if longer, so it can impact evaluation results)."
            )
        },
    )
    dataset_num_proc: Optional[int] = field(
        default=1,
        metadata={"help": "Number of processes to use for tokenization."},
    )
    dataset_group_text: Optional[bool] = field(
        default=False,
        metadata={
            "help": "The training dataset texts will be concatenated as chunks of fixed size (=block_size)."
        },
    )
    dataset_group_text_validation: Optional[bool] = field(
        default=True,
        metadata={"help": "Also group the validation dataset."},
    )
    dataset_block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length (take into account special tokens)."
            )
        },
    )
    dataset_streaming: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to stream the dataset or not."},
    )


@dataclass
class AdditionalArguments:
    """
    Additional arguments.
    """

    tokenizer_model_name: str = field(
        default=None,
        metadata={"help": "Model name of the tokenizer."},
    )


def parse_args():
    parser = HfArgumentParser((DatasetArguments, AdditionalArguments))
    return parser.parse_args()


def _get_block_size(tokenizer, tokenizer_model_name, block_size):
    if block_size is None:
        config = AutoConfig.from_pretrained(tokenizer_model_name)
        block_size = tokenizer.model_max_length
        max_pos_embeddings = config.max_position_embeddings
        if block_size > max_pos_embeddings:
            print(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(MAX_BLOCK_SIZE_WHEN_NONE, max_pos_embeddings)} instead. You can change that default value by passing --block_size xxx."
            )
            if max_pos_embeddings > 0:
                block_size = min(MAX_BLOCK_SIZE_WHEN_NONE, max_pos_embeddings)
            else:
                block_size = MAX_BLOCK_SIZE_WHEN_NONE
    else:
        if block_size > tokenizer.model_max_length:
            print(
                f"The block_size passed ({block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(block_size, tokenizer.model_max_length)
    return block_size


def _filter_empty(sample, dataset_text_field):
    return len(sample[dataset_text_field].strip()) > 0


def _tokenize(
    element, tokenizer, dataset_text_field, add_special_tokens, padding, max_seq_length
):
    outputs = tokenizer(
        element[dataset_text_field],
        add_special_tokens=add_special_tokens,
        truncation=True,
        padding=padding,
        max_length=max_seq_length,
        return_overflowing_tokens=False,
        return_length=False,
    )
    return {
        "input_ids": outputs["input_ids"],
        "attention_mask": outputs["attention_mask"],
        "labels": outputs["input_ids"].copy(),
    }


def _tokenize_dataset(
    tokenizer,
    dataset,
    dataset_text_field,
    max_seq_length,
    padding,
    column_names,
    add_special_tokens=True,
    map_args=None,
):

    if map_args is None:
        map_args = {}
    tokenized_dataset = dataset.map(
        partial(
            _tokenize,
            tokenizer=tokenizer,
            dataset_text_field=dataset_text_field,
            add_special_tokens=add_special_tokens,
            padding=padding,
            max_seq_length=max_seq_length,
        ),
        batched=True,
        remove_columns=column_names,
        **map_args,
    )

    return tokenized_dataset


def _group_texts(elements, block_size):
    # Concatenate all texts.
    concatenated_elements = {k: list(chain(*elements[k])) for k in elements.keys()}
    total_length = len(concatenated_elements[list(elements.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_elements.items()
    }
    return result


def load_and_tokenize(
    dataset_name: str,
    dataset_config_name: str,
    tokenizer_model_name: str,
    tokenized_dataset_save_dir: str,
    tokenized_dataset_load_dir: str,
    dataset_train_split: str,
    dataset_test_split: str,
    dataset_text_field: str,
    dataset_max_seq_length: int,
    dataset_num_proc: int,
    dataset_group_text: bool,
    dataset_group_text_validation: bool,
    dataset_block_size: int,
    dataset_streaming: bool,
    do_eval: bool = True,
) -> Dict[str, Dataset]:
    if tokenized_dataset_load_dir is not None:
        print(f"Loading tokenized dataset from: '{tokenized_dataset_load_dir}'")
        tokenized_datasets = datasets.load_from_disk(tokenized_dataset_load_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # Downloading and loading a dataset from the hub.
        loaded_datasets = load_dataset(
            dataset_name,
            dataset_config_name,
            streaming=dataset_streaming,
        )
        train_dataset = loaded_datasets[dataset_train_split]
        raw_datasets = {
            dataset_train_split: train_dataset,
        }
        if do_eval:
            eval_dataset = loaded_datasets[dataset_test_split]
            raw_datasets[dataset_test_split] = eval_dataset

        tokenized_datasets = DatasetDict()
        map_args = {}
        if dataset_streaming:
            print("Streaming dataset")
            column_names = list(list(train_dataset.take(1))[0].keys())
        else:
            map_args["num_proc"] = dataset_num_proc
            column_names = train_dataset.column_names
        for k, ds in raw_datasets.items():
            ds = ds.filter(
                partial(_filter_empty, dataset_text_field=dataset_text_field)
            )
            padding = (
                False
                if (k == dataset_train_split or dataset_group_text_validation)
                else "max_length"
            )
            tokenized_ds = _tokenize_dataset(
                tokenizer,
                ds,
                dataset_text_field=dataset_text_field,
                max_seq_length=dataset_max_seq_length,
                padding=padding,
                column_names=column_names,
                map_args=map_args,
            )
            if dataset_group_text and (
                k == dataset_train_split or dataset_group_text_validation
            ):
                block_size = _get_block_size(
                    tokenizer, tokenizer_model_name, dataset_block_size
                )
                tokenized_ds = tokenized_ds.map(
                    partial(_group_texts, block_size=block_size),
                    batched=True,
                    **map_args,
                )
            tokenized_datasets[k] = tokenized_ds
        if tokenized_dataset_save_dir is not None:
            tokenized_datasets.save_to_disk(tokenized_dataset_save_dir)

    return (
        tokenized_datasets[dataset_train_split],
        tokenized_datasets[dataset_test_split] if do_eval else None,
    )


if __name__ == "__main__":
    args = parse_args()
    load_and_tokenize(**vars(args))
