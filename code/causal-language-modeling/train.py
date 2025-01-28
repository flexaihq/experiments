import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import evaluate
import numpy
import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from dataset.prepare_save_dataset import DatasetArguments, load_and_tokenize
from utils.experiment_tracking import set_wandb

transformers.logging.set_verbosity_info()


@dataclass
class ModelArguments:
    """
    Model arguments.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Model name hosted inside a model repo on HF or a directory containing model weights."
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )


@dataclass
class AdditionalArguments:
    """
    Additional arguments that are not part of the Hugging Face Trainer/Model/Dataset arguments.
    """

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )


def parse_args():
    parser = HfArgumentParser(
        (DatasetArguments, ModelArguments, TrainingArguments, AdditionalArguments)
    )
    return parser.parse_args_into_dataclasses()


def _load_model_and_tokenizer(model_args, print_model=False):
    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch_dtype,
    )
    if print_model:
        print(model)

    return model, tokenizer


def train(dataset_args, model_args, training_args, additional_args):
    set_wandb(training_args)
    print(f"Training/evaluation parameters {training_args}")

    # Get dataset
    train_dataset, eval_dataset = load_and_tokenize(
        tokenizer_model_name=model_args.model_name_or_path,
        do_eval=training_args.do_eval,
        **vars(dataset_args),
    )

    max_train_samples = float("inf")
    max_eval_samples = float("inf")
    if not dataset_args.dataset_streaming:
        max_train_samples = len(train_dataset)
        if training_args.do_eval:
            max_eval_samples = len(eval_dataset)
    if additional_args.max_train_samples is not None:
        max_train_samples = min(max_train_samples, additional_args.max_train_samples)
        train_dataset = train_dataset.take(max_train_samples)
    if additional_args.max_eval_samples is not None:
        assert training_args.do_eval, "Cannot set max_eval_samples without do_eval"
        max_eval_samples = min(max_eval_samples, additional_args.max_eval_samples)
        eval_dataset = eval_dataset.take(max_eval_samples)

    model, tokenizer = _load_model_and_tokenizer(model_args, print_model=True)

    # Setup evaluation metric
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_preds):
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics
        preds, labels = eval_preds

        #  we need to mask the padding tokens before computing the metric
        # and we need to shift the labels
        mask = (labels != tokenizer.pad_token_id) & (labels != -100)
        labels = numpy.concatenate(
            [label[mask[i]][1:] for i, label in enumerate(labels)]
        )
        preds = numpy.concatenate([pred[mask[i]][:-1] for i, pred in enumerate(preds)])

        return metric.compute(predictions=preds, references=labels)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # Train
    train_result = trainer.train(
        resume_from_checkpoint=training_args.resume_from_checkpoint
    )
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    metrics["train_samples"] = max_train_samples

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if training_args.do_eval:
        # Evaluate
        metrics = trainer.evaluate()

        metrics["eval_samples"] = max_eval_samples
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    train(*parse_args())
