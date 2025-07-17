# Fine-Tuning a Language Model with LlamaFactory on FCS

This experiment demonstrates how to fine-tune a language model using [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) on **FlexAI**. We'll use the `Llama-3-1B` model and the `identity` and `alpaca-en-demo` [LlamaFactory datasets](https://github.com/hiyouga/LLaMA-Factory/tree/main/data) as an example, but you can adapt this guide for other models and datasets.

As you'll see below, you only need to pass your LlamaFactory configuration YAML.

> **Note**: If you haven't already connected FlexAI to GitHub, run `flexai code-registry connect` to set up a code registry connection. This allows FCS to pull repositories directly using the `-u` flag in training commands.

## Create Secrets

To be authenticated into your HuggingFace account within your code, you will use your _HuggingFace Token_.

Use the [`flexai secret create` command](https://docs.flex.ai/cli/commands/secret/) to store your _HuggingFace Token_ as a secret. Replace `<HF_AUTH_TOKEN_SECRET_NAME>` with your desired name for the secret:

```bash
flexai secret create <HF_AUTH_TOKEN_SECRET_NAME>
```

Then paste your _HuggingFace Token_ API key value.

## [Optional] Pre-fetch the Model

To speed up training and avoid downloading large models at runtime, you can pre-fetch your HuggingFace model to FlexAI storage. For example, to pre-fetch the `Qwen/Qwen2.5-72B` model:

1. **Create a HuggingFace storage provider:**

    ```bash
    flexai storage create HF-STORAGE --provider huggingface --hf-token-name <HF_AUTH_TOKEN_SECRET_NAME>
    ```

2. **Push the model checkpoint to your storage:**

    ```bash
    flexai checkpoint push qwen25-72b --storage-provider HF-STORAGE --source-path Qwen/Qwen2.5-72B
    ```

During your training run, you can use the pre-fetched model by adding the following argument to your training command:

```bash
--checkpoint qwen25-72b
```

## Train Llama Qwen2.5-72B (no model prefetch)

The [`qwen25-72B_sft.yaml`](../../code/llama-factory/qwen25-72B_sft.yaml) file has been adapted from [this example](https://github.com/hiyouga/LLaMA-Factory/blob/0b188ca00c9de9efee63807e72e444ea74195da5/examples/train_full/llama3_full_sft.yaml#L1).

To launch the training job:

```bash
flexai training run llamafactory-sft-llama3 \
  --accels 8 --nodes 2 \
  --repository-url https://github.com/flexaihq/flexai-experiments \
  --env FORCE_TORCHRUN=1 \
  --secret HF_TOKEN=<HF_AUTH_TOKEN_SECRET_NAME> \
  --requirements-path code/llama-factory/requirements.txt \
  --runtime nvidia-25.06 \
  -- /layers/flexai_pip-install/packages/bin/llamafactory-cli train code/llama-factory/qwen25-72B_sft.yaml
```

## Train Llama Qwen2.5-72B (with model prefetch)

To take advantage of model pre-fetching performed in the [Optional: Pre-fetch the Model](#optional-pre-fetch-the-model) section, use:

```bash
flexai training run llamafactory-sft-llama3 \
  --accels 8 --nodes 2 \
  --repository-url https://github.com/flexaihq/flexai-experiments \
  --checkpoint qwen25-72b \
  --env FORCE_TORCHRUN=1 \
  --secret HF_TOKEN=<HF_AUTH_TOKEN_SECRET_NAME> \
  --requirements-path code/llama-factory/requirements.txt \
  --runtime nvidia-25.06 \
  -- /layers/flexai_pip-install/packages/bin/llamafactory-cli train code/llama-factory/qwen25-prefetched_sft.yaml
```

## Train Llama 3

The [`llama3_sft.yaml`](../../code/llama-factory/llama3_sft.yaml) file has been adapted from [this example](https://github.com/hiyouga/LLaMA-Factory/blob/0b188ca00c9de9efee63807e72e444ea74195da5/examples/train_full/llama3_full_sft.yaml#L1).

To launch the training job:

```bash
flexai training run llamafactory-sft-llama3 \
  --accels 8 --nodes 2 \
  --repository-url https://github.com/flexaihq/flexai-experiments \
  --env FORCE_TORCHRUN=1 \
  --secret HF_TOKEN=<HF_AUTH_TOKEN_SECRET_NAME> \
  --requirements-path code/llama-factory/requirements.txt \
  --runtime nvidia-25.06 \
  -- /layers/flexai_pip-install/packages/bin/llamafactory-cli train code/llama-factory/llama3_sft.yaml
```

---

## [Optional] Prefetch Your Own Dataset

You can check our other examples, e.g. [`experiments/running-a-simple-training-job/README.md`](../running-a-simple-training-job/README.md), to see how to bring your own dataset using:

```bash
flexai dataset push my-dataset ...
```

Then, follow the [LlamaFactory dataset instructions](https://github.com/hiyouga/LLaMA-Factory/tree/main/data#readme) to prepare your data for training.

---
