# Experiment Tracking with Weights and Biases

## Experiment Tracking

_Experiment tracking_ involves systematically recording and managing details of machine learning experiments, such as code, data, configurations, parameters, metrics, and results.
It ensures reproducibility, comparability, and accountability across experiments, aiding in efficient model development and deployment.
Weights & Biases (_wandb_) is one approach to achieving this.

Follow the next instructions to log experiments to your _wandb_ account.

## Setting Up the Weights and Biases Secret

To enable seamless integration with _wandb_ in your experiments, follow these steps to create the _wandb_ secret:

1. **Retrieve Your API Key**

   Visit your [Weights & Biases Settings page](https://app.wandb.ai/settings) to find your API key. Copy the key for use in the next step.

2. **Create the Secret**

   Use the [`flexai secret create` command](https://docs.flex.ai/cli/commands/secret/) to store your _wandb_ API key as a secret. Replace `<WANDB_API_KEY_SECRET_NAME>` with your desired name for the secret:

   ```bash
   flexai secret create <WANDB_API_KEY_SECRET_NAME>
   ```

   Then past your _wandb_ API key value.

3. **Note on Project Name**

   Keep in mind that the project name used in your _wandb_ setup does not need to be an FlexAI Secret. Additionally, the project name does not need to be pre-created in _wandb_ — it will be automatically created if it doesn’t exist when you log your first experiment.

## Log to Weights and Biases

 You can now log experiments to your _wandb_ account by adding the following flags to any `flexai training run` command:

```bash
--secret WANDB_API_KEY=<WANDB_API_KEY_SECRET_NAME> --env WANDB_PROJECT=<YOUR_PROJECT_NAME>
```

You can optionally set your _run name_ using the `--run_name <YOUR_RUN_NAME>` HuggingFace arg.

For more ways to customize and configure your _wandb_ environment, check out the [Weights & Biases Environment Variables Guide](https://docs.wandb.ai/guides/track/environment-variables/).

## Setting Up the Experiment

### Step 1: Connect to GitHub (if needed)

If you haven't already connected FlexAI to GitHub, you'll need to set up a code registry connection:

```bash
flexai code-registry connect
```

This will allow FlexAI to pull repositories directly from GitHub using the `-u` flag in training commands.

### Step 2: Preparing the Dataset

In this experiment, we will use a pre-processed version of the the `wikitext` dataset that has been set up for the `GPT-2` model.

1. Download the dataset:

    ```bash
    DATASET_NAME=gpt2-tokenized-wikitext && curl -L -o ${DATASET_NAME}.zip "https://bucket-docs-samples-99b3a05.s3.eu-west-1.amazonaws.com/${DATASET_NAME}.zip" && unzip ${DATASET_NAME}.zip && rm ${DATASET_NAME}.zip
    ```

2. Upload the dataset (located in `gpt2-tokenized-wikitext/`) to FlexAI:

    ```bash
    flexai dataset push gpt2-tokenized-wikitext --file gpt2-tokenized-wikitext
    ```

## Running the Training Job with Experiment Tracking

Now that all the pieces are in place (_wandb_ Secret, Source, and Dataset), you can run the training job with experiment tracking enabled.

```bash
flexai training run gpt2training-tracker --repository-url https://github.com/flexaihq/experiments --dataset gpt2-tokenized-wikitext --secret WANDB_API_KEY=<WANDB_API_KEY_SECRET_NAME> --env WANDB_PROJECT=<YOUR_PROJECT_NAME> --requirements-path code/causal-language-modeling/requirements.txt \
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
    --eval_strategy steps \
    --run_name <YOUR_RUN_NAME>
```

You can now visit your _wandb_ dashboard and look for your project's name to follow the progress of the Training Job and analyze its results in near real-time.
