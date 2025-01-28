# Fine-tuning Stable Diffusion XL with LoRA

In this experiment, we will fine-tune Stable Diffusion XL on the with LoRA using the `Diffusers` library.

The goal is to specialize the model for generating content related to Naruto.
Fine-tuning enables the model to better produce outputs tailored to the themes and characteristics of the dataset, which may include unique styles, characters, or captions associated with the Naruto universe.

## Prepare the Dataset

We will be using the `lambdalabs/naruto-blip-captions` dataset. You can download the pre-processed version of the dataset by running the following command:

```bash
DATASET_NAME=sdxl-tokenized-naruto && curl -L -o ${DATASET_NAME}.zip "https://bucket-docs-samples-99b3a05.s3.eu-west-1.amazonaws.com/${DATASET_NAME}.zip" && unzip ${DATASET_NAME}.zip && rm ${DATASET_NAME}.zip
```

> If you'd like to reproduce the pre-processing steps yourself to use a different dataset or simply to learn more about the process, you can refer to the [Manual Dataset Pre-processing](#manual-dataset-pre-processing) section below.

Next, push the contents of the `sdxl-tokenized-naruto/` directory as a new FCS dataset:

```bash
flexai dataset push sdxl-tokenized-naruto --file sdxl-tokenized-naruto
```

## Training

To start the Training Job, run the following command:

```bash
flexai training run text-to-image-lora-SDXL-training-ddp --source-name fcs-experiments --dataset sdxl-tokenized-naruto --secret HF_TOKEN=<HF_AUTH_TOKEN_SECRET_NAME> --secret WANDB_API_KEY=<WANDB_API_KEY_SECRET_NAME> \
  --nodes 1 --accels 2 \
  -- code/diffuser/train_text_to_image_lora_sdxl.py \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
    --pretrained_vae_model_name_or_path madebyollin/sdxl-vae-fp16-fix \
    --train_dataset_load_dir /input \
    --caption_column text \
    --resolution 1024 \
    --random_flip \
    --train_batch_size 1 \
    --num_train_epochs 2 \
    --checkpointing_steps 500 \
    --learning_rate 1e-04 \
    --lr_scheduler constant \
    --lr_warmup_steps 0 \
    --mixed_precision fp16 \
    --seed 42 \
    --output_dir /output \
    --validation_prompt "'cute dragon creature'"
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
python dataset/prepare_save_text_to_image.py \
  --dataset_name lambdalabs/naruto-blip-captions \
  --train_dataset_save_dir sdxl-tokenized-naruto
```

The prepared dataset will be saved to the `sdxl-tokenized-naruto/` directory.
