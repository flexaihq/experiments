# Fine-tuning a Text-to-Speech model

The goal of this experiment is to fine-tune the `parler_tts_mini_v0.1` model to create a French version.

The model generates high-quality speech from input text, which can be controlled using a description prompt (e.g., gender, speaking rate, etc.).

The training uses the a `text-to-speech` dataset in French, enabling the model to produce natural and expressive speech in this language.

## Step 1: Connect to GitHub (if needed)

If you haven't already connected FlexAI to GitHub, you'll need to set up a code registry connection:

```bash
flexai code-registry connect
```

This will allow FlexAI to pull repositories directly from GitHub using the `-u` flag in training commands.

## Step 2: Getting the Dataset

You can download the pre-processed version of the dataset by running the following command:

```bash
DATASET_NAME=text-to-speech-fr && curl -L -o ${DATASET_NAME}.zip "https://bucket-docs-samples-99b3a05.s3.eu-west-1.amazonaws.com/${DATASET_NAME}.zip" && unzip ${DATASET_NAME}.zip && rm ${DATASET_NAME}.zip
```

> If you'd like to reproduce the pre-processing steps yourself to use a different dataset or simply to learn more about the process, you can refer to the [Manual Dataset Pre-processing](#manual-dataset-pre-processing) section below.

Next, push the contents of the `text-to-speech-fr/` directory as a new FCS dataset:

```bash
flexai dataset push text-to-speech-fr --file text-to-speech-fr
```

## Training

To start the Training Job, run the following command:

```bash
flexai training run text-to-speech-ddp --repository-url https://github.com/flexaihq/fcs-experiments --dataset text-to-speech-fr --secret WANDB_API_KEY=<WANDB_API_KEY_SECRET_NAME> \
  --nodes 1 --accels 8 \
  -- code/text-to-speech/run_parler_tts_training.py ./code/text-to-speech/french_training.json
```

Instead of passing a `.json` file as input, you can also set the arguments manually. For example:

```bash
flexai training run text-to-speech-ddp --repository-url https://github.com/flexaihq/fcs-experiments --dataset text-to-speech-fr --secret WANDB_API_KEY=<WANDB_API_KEY_SECRET_NAME> \
  --nodes 1 --accels 8 \
  -- code/text-to-speech/run_parler_tts_training.py \
    --model_name_or_path=parler-tts/parler_tts_mini_v0.1 \
    --save_to_disk=/input/text-to-speech-fr \
    --temporary_save_to_disk=./audio_code_tmp/ \
    --wandb_project=parler-francais \
    --feature_extractor_name=ylacombe/dac_44khZ_8kbps \
    --description_tokenizer_name=google/flan-t5-large \
    --prompt_tokenizer_name=google/flan-t5-large \
    --report_to=wandb \
    --overwrite_output_dir \
    --output_dir=/output-checkpoint \
    --train_dataset_name=PHBJT/cml-tts-20percent-subset \
    --train_metadata_dataset_name=PHBJT/cml-tts-20percent-subset-description \
    --train_dataset_config_name=default \
    --train_split_name=train \
    --eval_dataset_name=PHBJT/cml-tts-20percent-subset \
    --eval_metadata_dataset_name=PHBJT/cml-tts-20percent-subset-description \
    --eval_dataset_config_name=default \
    --eval_split_name=test \
    --target_audio_column_name=audio \
    --description_column_name=text_description \
    --prompt_column_name=text \
    --max_eval_samples=10
```

## Optional Extra Steps

You can run these extra steps in an [FCS Interactive Session](https://docs.flex.ai/guides/interactive-training) or in a local env (e.g. `pipenv install --python 3.10`), if you have hardware that's capable of doing inference.

### Inference

A simple inference script that you can easily adapt to your needs is available [here](/code/text-to-speech/predict.py).

### Manual Dataset Pre-processing

If you'd prefer to perform the dataset pre-processing step yourself, you can follow these instructions.

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

Prepare the dataset by running the training command with the `--preprocessing_only` flag in `./code/text-to-speech/french_training.json`.

> Note: For large datasets, it is recommended to run the preprocessing on a single machine to avoid timeouts when running the script in distributed mode.

The content will be saved to the destination specified in `--save_to_disk=./text-to-speech-fr/`.

Run the dataset preparation using:

```bash
python code/text-to-speech/run_parler_tts_training.py ./code/text-to-speech/french_training.json
```

Make sure to remove the `--preprocessing_only` flag before attempting to run the script for training purposes.
