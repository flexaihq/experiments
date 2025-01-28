# Resuming a Training Job from Checkpoint

This experiment will continue a training from a Checkpoint emitted by the Training Job in the [A simple Training Job on FCS](/experiments/running-a-simple-training-job/README.md) experiment, so make sure to complete it and download its output artifacts before proceeding.

Extract the contents of the `output_0.zip` file into a directory named `fetched_checkpoints`:

```bash
unzip output_0.zip -d fetched_checkpoints
```

This `fetched_checkpoints` directory contains the different checkpoints that have been saved in the `/output` of the Training Job's runtime environment during execution.

Let's use the checkpoint (saved at step 500) located in `fetched_checkpoints/output/checkpoint-500/`.

Create the FCS checkpoint to be passed to the next run that will resume the training:

```bash
flexai checkpoint push gpt2-ckpt500 --file fetched_checkpoints/output/checkpoint-500
```

Resume training from your checkpoint with the following command:

```bash
flexai training run gpt2training-resume --source-name fcs-experiments --dataset gpt2-tokenized-wikitext --checkpoint gpt2-ckpt500 \
  -- code/causal-language-modeling/train.py \
    --do_eval \
    --do_train \
    --dataset_name wikitext \
    --tokenized_dataset_load_dir /input \
    --model_name_or_path /checkpoint \
    --resume_from_checkpoint /checkpoint \
    --output_dir /output \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --logging_steps 50 \
    --save_steps 500 \
    --eval_steps 500 \
    --eval_strategy steps \
    --num_train_epochs 6
```

Compared to the experiment that starts training from the base model, note that:

- `--checkpoint gpt2-ckpt500` has been added - referring to the checkpoint created above, the content of the `checkpoint-500` folder will be mounted on `/checkpoint`
- `--model_name_or_path` has been updated - pointing to the new checkpoint location

together with additional HuggingFace args to resume the training from the checkpoint:

- `--resume_from_checkpoint /checkpoint`
- `--num_train_epochs 6`
