#!/usr/bin/env python3
"""
Ray cluster bootstrap script for LSF environments.
This script connects to an existing Ray cluster and runs the training job.
Based on the CLUSTER_DEBUGGING.md documentation.
"""

import os
import sys
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

    # Make config path absolute if it's relative
    if not os.path.isabs(config_path):
        # The main working directory should be the repository root
        repo_root = "/nrs/cellmap/rhoadesj/resolution-aware-transformer"
        config_path = os.path.join(repo_root, config_path)

    print(f"Loading config from: {config_path}")

    # Load and parse the config
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

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

    args = parser.parse_args()

    print("=== Ray Cluster Bootstrap ===")
    print(f"Config: {args.config}")
    print(f"GPUs: {args.num_gpus}")
    print(f"World GPUs (from env): {os.environ.get('WORLD_GPUS', 'not set')}")

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

    # Configure run settings
    run_config = RunConfig(
        name=args.name,
        storage_path=f"/scratch/{os.getenv('USER', 'unknown')}/ray_results",
    )

    # Set up the trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={
            "config_path": args.config,
            "num_gpus": args.num_gpus,
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
        return True

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
