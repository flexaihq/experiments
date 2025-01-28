import os


def set_wandb(training_args=None):
    report_to = None
    if "WANDB_API_KEY" in os.environ:
        os.environ["WANDB_LOG_MODEL"] = "false"  # Don't upload model checkpoints to W&B
        if training_args is not None:
            training_args.report_to = ["wandb"]
        report_to = "wandb"
    else:
        # Disable W&B logging
        os.environ["WANDB_SILENT"] = "true"
        os.environ["WANDB_MODE"] = "offline"
        if training_args is not None:
            training_args.report_to = []
    return report_to
