#!/usr/bin/env python3
"""
Ray cluster bootstrap script for LSF environments.
This script connects to an existing Ray cluster and runs the training job.
Based on the CLUSTER_DEBUGGING.md documentation.
"""

import os
import sys
from pathlib import Path
import ray
from ray.train import ScalingConfig, RunConfig, get_context
from ray.train.torch import TorchTrainer
from ray.train.torch.config import TorchConfig
from ray.air.config import FailureConfig

# Add the experiments directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def train_loop_per_worker(cfg):
    """
    Ray Train worker function that runs on each GPU.
    This is the entry point for each distributed worker.
    """
    import torch
    from torch.nn.parallel import DistributedDataParallel as DDP

    # Get Ray Train context
    ctx = get_context()
    local_rank = ctx.get_local_rank()
    world_rank = ctx.get_world_rank()
    world_size = ctx.get_world_size()

    print(f"Worker {world_rank}/{world_size} starting on GPU {local_rank}")

    # Set the GPU device for this worker
    torch.cuda.set_device(local_rank)

    # Initialize the process group (Ray Train handles this automatically)
    # but we need to call our existing training function

    # Extract config from the Ray Train config
    config_path = cfg.get("config_path")
    num_gpus = cfg.get("num_gpus")
    disable_deepspeed = cfg.get("disable_deepspeed", False)

    # Make config path absolute if it's relative
    if not os.path.isabs(config_path):
        # The working directory is experiments/, so resolve relative to that
        experiments_dir = (
            "/nrs/cellmap/rhoadesj/resolution-aware-transformer/experiments"
        )
        config_path = os.path.join(experiments_dir, config_path)

    print(f"Loading config from: {config_path}")

    # Verify config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load and parse the config
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Override DeepSpeed setting if disabled
    if disable_deepspeed:
        print("DeepSpeed disabled via command line flag")
        if "training" not in config:
            config["training"] = {}
        config["training"]["use_deepspeed"] = False

    # Set up environment to mimic the original setup
    os.environ["RANK"] = str(world_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)

    # Run the training using the actual training function
    try:
        # Import the training function that we need
        from ray_train import train_function

        # Call the training function directly with the config
        train_function(config)

    except Exception as e:
        print(f"Training failed on worker {world_rank}: {e}")
        raise


def main():
    """Main function that sets up and runs Ray Train."""
    import argparse

    parser = argparse.ArgumentParser(description="Ray Train cluster bootstrap")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--num-gpus", type=int, default=8, help="Number of GPUs")
    parser.add_argument("--name", default="RAT_training", help="Experiment name")
    parser.add_argument(
        "--disable-deepspeed",
        action="store_true",
        help="Disable DeepSpeed and use standard distributed training",
    )

    args = parser.parse_args()

    print("=== Ray Cluster Bootstrap ===")
    print(f"Config: {args.config}")
    print(f"GPUs: {args.num_gpus}")
    print(f"World GPUs (from env): {os.environ.get('WORLD_GPUS', 'not set')}")
    print(f"Working directory: {os.getcwd()}")

    # Check if config file exists in experiments dir
    config_abs_path = os.path.join(
        "/nrs/cellmap/rhoadesj/resolution-aware-transformer/experiments", args.config
    )
    print(f"Looking for config at: {config_abs_path}")
    print(f"Config exists: {os.path.exists(config_abs_path)}")

    # Initialize ExperimentTracker for proper organization
    try:
        from common.experiment_tracker import ExperimentTracker

        tracker = ExperimentTracker(
            "/nrs/cellmap/rhoadesj/resolution-aware-transformer/experiments/results"
        )

        # Create descriptive experiment name with key details
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = Path(args.config).stem  # e.g., "medical_segmentation"
        deepspeed_suffix = "_no_ds" if args.disable_deepspeed else "_ds"
        experiment_name = f"{args.name}_{config_name}_{args.num_gpus}gpu{deepspeed_suffix}_{timestamp}"

        # Register experiment and get organized directory
        experiment_id = tracker.register_experiment(
            experiment_name=experiment_name,
            config_path=config_abs_path,
            task_type="ray_train_distributed",
            num_gpus=args.num_gpus,
            additional_info={
                "ray_train": True,
                "deepspeed_disabled": args.disable_deepspeed,
                "config_file": args.config,
                "lsf_job_id": os.environ.get("LSB_JOBID", "unknown"),
                "node_count": 1,
                "gpus_per_node": args.num_gpus,
            },
        )

        # Get the organized experiment directory
        exp_metadata = tracker.registry["experiments"][experiment_id]
        organized_storage_path = str(
            Path(exp_metadata["directory"]).parent.parent
        )  # Go up to experiments/ level

        print(f"Experiment registered as: {experiment_id}")
        print(f"Results will be saved to: {organized_storage_path}")

    except ImportError as e:
        print(f"Warning: Could not import ExperimentTracker: {e}")
        print("Using default Ray storage path")
        organized_storage_path = "/nrs/cellmap/rhoadesj/resolution-aware-transformer/experiments/results/experiments"
        experiment_id = args.name
        tracker = None

    # Connect to the existing Ray cluster (LSF-launched)
    print("Connecting to Ray cluster...")
    ray.init(address="auto")  # Always attach to the LSF-launched cluster

    print("Ray cluster info:")
    print(ray.cluster_resources())

    # Configure Ray Train scaling
    scaling = ScalingConfig(
        num_workers=int(os.environ.get("WORLD_GPUS", args.num_gpus)),
        use_gpu=True,
        resources_per_worker={"CPU": 11, "GPU": 1},  # Matches 96c/8g per node
    )

    # Configure PyTorch backend
    torch_config = TorchConfig(
        backend="nccl",
        timeout_s=1800,  # 30 minute timeout for NCCL
    )

    # Configure run settings with organized storage
    run_config = RunConfig(
        name=experiment_id,
        storage_path=organized_storage_path,
    )

    # Set up the trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={
            "config_path": args.config,
            "num_gpus": args.num_gpus,
            "disable_deepspeed": args.disable_deepspeed,
        },
        scaling_config=scaling,
        torch_config=torch_config,
        run_config=run_config,
    )

    print("Starting Ray Train...")
    try:
        result = trainer.fit()
        print("Training completed successfully!")
        print("Result:", result)

        # Update experiment status to completed
        if tracker:
            tracker.update_experiment_status(
                experiment_id, "completed", {"ray_result": str(result)}
            )

        return True

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback

        traceback.print_exc()

        # Update experiment status to failed
        if tracker:
            tracker.update_experiment_status(experiment_id, "failed", {"error": str(e)})

        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
