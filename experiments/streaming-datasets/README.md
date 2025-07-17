# Streaming large datasets during a Training Job

In some cases you might want to use large datasets that would be too large to download or [push to FCS](https://docs.flex.ai/cli/guides/uploading-datasets/) and you'd prefer to use that data transfer time more efficiently. _Streaming_ such datasets can be a useful technique in those cases.

This experiment demonstrates how to stream a large dataset during a Training Job on FCS. We'll use the [HuggingFace Datasets library](https://huggingface.co/docs/datasets/en/stream)'s Streaming capabilities to achieve this.

## Step 1: Connect to GitHub (if needed)

If you haven't already connected FlexAI to GitHub, you'll need to set up a code registry connection:

```bash
flexai code-registry connect
```

This will allow FlexAI to pull repositories directly from GitHub using the `-u` flag in training commands.

## Step 2: Preparing the Dataset

> [!NOTE]
> The `flexai training run` command requires the `--dataset` flag to be set. Even though we're going to stream the dataset during runtime, we still need to specify a dataset. Here we use can use the `gpt2-tokenized-wikitext` dataset we used in previous experiments, however, any other dataset you have available can be used. Even creating a new empty dataset would work:
>
> ```bash
> touch empty-file && flexai dataset push empty-dataset --file empty-file
> ```

## Running the Training Job streaming a dataset

Here is an example using the `code/causal-language-modeling/train.py` script to stream the over 90 TB [Fineweb dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb):

```bash
flexai training run gpt2training-stream --repository-url https://github.com/flexaihq/experiments --dataset empty-dataset --requirements-path code/causal-language-modeling/requirements.txt \
   -- code/causal-language-modeling/train.py \
    --dataset_streaming true \
    --do_train \
    --eval_strategy no \
    --dataset_name HuggingFaceFW/fineweb \
    --dataset_config_name CC-MAIN-2024-10 \
    --dataset_group_text true \
    --dataloader_num_workers 8 \
    --max_steps 2500 \
    --model_name_or_path openai-community/gpt2 \
    --output_dir /output-checkpoint \
    --per_device_train_batch_size 8 \
    --logging_steps 50 \
    --save_steps 1000
```

The first line defines the 3 main components required to run a Training Job in FCS:

1. The Training Job's name (`gpt2training-stream`).
1. The URL of the repository containing the training script (`https://github.com/flexaihq/experiments`).
1. The name of the dataset to be used (`empty-dataset` or any other dataset you have available).

The second line is defines the script that will be executed when the Training Job is started (`code/causal-language-modeling/train.py`).

Below that, the first argument passed to the script is `--dataset_streaming true`, which value tells the script to use the Datasets library with streaming capabilities enabled.

The next lines specify the arguments that will be passed to the training script during execution to adjust the Training Job's hyperparameters or customize its behavior. For instance, `--max_train_samples` and `--max_eval_samples` can be used to tweak the sample size.

## The code

You will notice that the `train` function in the `code/causal-language-modeling/train.py` script makes a call to the `_load_model_and_tokenizer` function to load the model and tokenizer using the user-provided arguments:

```python
def train(dataset_args, model_args, training_args, additional_args):     # <--- 1. This is the function that will be called by the `flexai training run` command
    set_wandb(training_args)
    print(f"Training/evaluation parameters {training_args}")

    # Get dataset
    train_dataset, eval_dataset = load_and_tokenize(                     # <--- 2. Here the script calls the `load_and_tokenize` helper function
        tokenizer_model_name=model_args.model_name_or_path,
        do_eval=training_args.do_eval,
        **vars(dataset_args),                                            # <--- 3. These are the arguments passed to the script
    )
```

The `load_and_tokenize` helper function from the `code/dataset/prepare_save_dataset.py` file is the one responsible for using the HuggingFace's Datasets library and enable its _streaming capabilities_ by simply setting the `load_dataset`'s `streaming` argument to `True`:

```python
def load_and_tokenize(
    dataset_name: str,
    # ...
    dataset_streaming: bool,
) -> Dict[str, Dataset]:
    # ...
    loaded_datasets = load_dataset(                                       # <--- 1. HuggingFace's Datasets library `load_dataset` function is called
        dataset_name,
        dataset_config_name,
        streaming=dataset_streaming,                                      # <--- 2. The `streaming` argument is set to `True`
    )
```

This is all that is needed to stream a dataset during a Training Job on FCS! You are no longer restricted by the challenges that come with large dataset transfer processes, and can now use them more efficiently.
