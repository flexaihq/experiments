# Fine-tuning Llama3.1 with QLoRA

In this experiment, we will fine-tune a causal language model using QLoRA and the `SFTTrainer` from `trl`.

We will use `Llama3.1` as our large language model (LLM) and train it on the `openassistant-guanaco` dataset but you can easily choose another model or dataset from the HuggingFace hub.

## Step 1: Adding this repository as a Source

If you haven't already, you will need to add this repository as a [Source](https://docs.flex.ai/quickstart/adding-sources) to your FlexAI account.

This repository contains the list of required dependencies (in the `requirements.txt` file) and the code that will handle the training process. To add a Source, run the following command:

```bash
flexai source add fcs-experiments https://github.com/flexaihq/fcs-experiments.git
```

## Step 2: Preparing the Dataset

In this experiment, we will use a pre-processed version of the the `openassistant-guanaco` dataset that has been set up for the `Llama3.1` model.

```bash
DATASET_NAME=llama-tokenized-oag && curl -L -o ${DATASET_NAME}.zip "https://bucket-docs-samples-99b3a05.s3.eu-west-1.amazonaws.com/${DATASET_NAME}.zip" && unzip ${DATASET_NAME}.zip && rm ${DATASET_NAME}.zip
```

> If you'd like to reproduce the pre-processing steps yourself to use a different dataset or simply to learn more about the process, you can refer to the [Manual Dataset Pre-processing](#manual-dataset-pre-processing) section below.

Next, push the contents of the `llama-tokenized-oag/` directory as a new FCS dataset:

```bash
flexai dataset push llama-tokenized-oag --file llama-tokenized-oag
```

## Create Secrets

To access the Llama-3.1-8B model, you need to [accept the license](https://huggingface.co/meta-llama/Llama-3.1-8B) with your HuggingFace account.

To be authenticated within your code, you will use your _HuggingFace Token_.

Use the [`flexai secret create` command](https://docs.flex.ai/commands/secret) to store your _HuggingFace Token_ as a secret. Replace `<HF_AUTH_TOKEN_SECRET_NAME>` with your desired name for the secret:

```bash
flexai secret create <HF_AUTH_TOKEN_SECRET_NAME>
```

Then paste your _HuggingFace Token_ API key value.

## Training

To start the Training Job, run the following command:

```bash
flexai training run llama3-1-training-ddp --source-name fcs-experiments --dataset llama-tokenized-oag --secret HF_TOKEN=<HF_AUTH_TOKEN_SECRET_NAME> --secret WANDB_API_KEY=<WANDB_API_KEY_SECRET_NAME> -env WANDB_PROJECT=<YOUR_PROJECT_NAME> \
  --nodes 1 --accels 2 \
  -- code/causal-language-modeling-qlora/train.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
    --dataset_name timdettmers/openassistant-guanaco \
    --tokenized_dataset_load_dir /input \
    --dataset_text_field text \
    --load_in_4bit \
    --use_peft \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --output_dir /output \
    --log_level info
```

## Optional Extra Steps

### Manual Dataset Pre-processing

If you'd prefer to perform the dataset pre-processing step yourself, you can follow these instructions.

You can run these in an [FCS Interactive Session](https://docs.flex.ai/guides/interactive-training) or in a local env (e.g. `pipenv install --python 3.10`), if you have hardware that's capable of doing inference.

#### Clone this repository

If you haven't already, clone this repository on your host machine:

```bash
git clone https://github.com/flexaihq/fcs-experiments.git --depth 1 --branch main && cd fcs-experiments
```

#### Install the dependencies

Depending on your environment, you might need to install - if not already - the experiments' dependencies by running:

```bash
pip install -r requirements.txt
```

#### Dataset preparation

Prepare the dataset by running the following command:

```bash
python dataset/prepare_save_sft.py \
  --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
  --dataset_name timdettmers/openassistant-guanaco \
  --dataset_text_field text \
  --log_level info \
  --tokenized_dataset_save_dir llama-tokenized-oag \
  --output_dir ./.sft.output # This argument is not used but is required to use the SFT argument parser.
```

The prepared dataset will be saved to the `llama-tokenized-oag/` directory.
