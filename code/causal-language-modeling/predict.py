import os
import sys
from dataclasses import dataclass, field

from transformers import HfArgumentParser

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from train import ModelArguments, _load_model_and_tokenizer


@dataclass
class PredictArguments:
    """
    Arguments for prediction.
    """

    input_str: str = field(
        metadata={"help": "Input string to generate predictions for."},
    )
    max_new_tokens: int = field(
        default=100,
        metadata={"help": "Maximum number of tokens to generate."},
    )


def parse_args():
    parser = HfArgumentParser((ModelArguments, PredictArguments))
    return parser.parse_args_into_dataclasses()


def predict(model_args, predict_args):
    model, tokenizer = _load_model_and_tokenizer(model_args, None)
    input_str = predict_args.input_str
    model_inputs = tokenizer(input_str, return_tensors="pt")
    output = model.generate(**model_inputs, max_new_tokens=predict_args.max_new_tokens)
    print("\nGenerated text:")
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    predict(*parse_args())
