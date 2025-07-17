# Fine-Tuning a language Model with flash-attention

> [!NOTE]
> This experiment is temporarily disabled.

This experiment demonstrates how easy it is to leverage **FlexAI** to run a Training Job making use of _Flash Attention_ through the [flash-attention](https://github.com/Dao-AILab/flash-attention) package with a couple of commands. We will use an example of training a causal language model (LLM) on the `wikitext` dataset using the `GPT-2` model.

You will see that this straightforward process only requires two components: a training script and a dataset. The training script is responsible for defining the model, setting up and applying hyperparameters, running the training loop, and applying its respective evaluation logic, while the dataset contains the information that will be used to train the model.

## Step 1: Connect to GitHub (if needed)

If you haven't already connected FlexAI to GitHub, you'll need to set up a code registry connection:

```bash
flexai code-registry connect
```

This will allow FlexAI to pull repositories directly from GitHub using the `-u` flag in training commands.

## Step 2: Preparing the Dataset

In this experiment, we will use a pre-processed version of the the `wikitext` dataset that has been set up for the `GPT-2` model.

> If you'd like to reproduce the pre-processing steps yourself to use a different dataset or simply to learn more about the process, you can refer to the [Manual Dataset Pre-processing](#manual-dataset-pre-processing) section below.

1. Download the dataset:

    ```bash
    DATASET_NAME=gpt2-tokenized-wikitext && curl -L -o ${DATASET_NAME}.zip "https://bucket-docs-samples-99b3a05.s3.eu-west-1.amazonaws.com/${DATASET_NAME}.zip" && unzip ${DATASET_NAME}.zip && rm ${DATASET_NAME}.zip
    ```

1. Upload the dataset (located in `gpt2-tokenized-wikitext/`) to FCS:

    ```bash
    flexai dataset push gpt2-tokenized-wikitext --file gpt2-tokenized-wikitext
    ```

## Step 3: Train the Model

Now, it's time to train your LLM on the dataset you just _pushed_ in the previous step, `gpt2-tokenized-wikitext`. This experiment uses the `GPT-2` model, however, the training script script we will use ([`code/causal-language-modeling/train.py`](../../code/causal-language-modeling/train.py)) leverages the [HuggingFace Transformers `Trainer` class](https://huggingface.co/docs/transformers/en/trainer), which makes it easy to replace `GPT-2` with another [model](https://huggingface.co/docs/text-generation-inference/en/conceptual/flash_attention) compatible with `flash-attention`.

To start the Training Job, run the following command:

```bash
flexai training run fcs-experiments-flash-attention --repository-url https://github.com/flexaihq/experiments --dataset gpt2-tokenized-wikitext --requirements-path code/causal-language-modeling/requirements-flash-attn.txt \
 -- code/causal-language-modeling/train.py \
    --do_eval \
    --do_train \
    --dataset_name wikitext \
    --tokenized_dataset_load_dir /input/gpt2-tokenized-wikitext \
    --model_name_or_path openai-community/gpt2 \
    --output_dir /output-checkpoint \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --logging_steps 50 \
    --save_steps 500 \
    --eval_steps 500 \
    --attn_implementation flash_attention_2 \
    --torch_dtype float16 \
    --eval_strategy steps
```

The first line defines the 3 main components required to run a Training Job in FCS:

1. The Training Job's name (`fcs-experiments-flash-attention`).
1. The URL of the repository containing the training script (`https://github.com/flexaihq/experiments`).
1. The name of the dataset to be used (`gpt2-tokenized-wikitext`).

The second line defines the script that will be executed when the Training Job is started (`code/causal-language-modeling/train.py`).

After the second line come the script's arguments, which are passed to the script when it is executed to adjust the Training Job hyperparameters or customize its behavior. For instance, `--max_train_samples` and `--max_eval_samples` can be used to tweak the sample size.

## Step 4: Checking up on the Training Job

You can check the status and life cycle events of your Training Job by running:

```bash
flexai training inspect fcs-experiments-flash-attention
```

Additionally, you can view the logs of your Training Job by running:

```bash
flexai training logs fcs-experiments-flash-attention
```

## Step 5: Fetching the Trained Model artifacts

Once the Training Job completes successfully, you will be able to download its output artifacts by running:

```bash
flexai training fetch fcs-experiments-flash-attention
```

This will download a `zip` file containing the trained model artifacts to your current working directory.

You can now have a look at other FCS experiments within this repository to explore other use cases and techniques.

## Optional Extra Steps

### Manual Dataset Pre-processing

To prepare and save the `wikitext` dataset for the `GPT-2` model run the following command:

```bash
python code/dataset/prepare_save_dataset.py \
    --dataset_name wikitext \
    --tokenized_dataset_save_dir gpt2-tokenized-wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --tokenizer_model_name openai-community/gpt2 \
    --dataset_group_text true
```

The generated dataset will be created in the directory set as the value of `--tokenized_dataset_save_dir`, in this case: `gpt2-tokenized-wikitext`.
Keep in mind that you can use other combinations of datasets and models available on HuggingFace.
