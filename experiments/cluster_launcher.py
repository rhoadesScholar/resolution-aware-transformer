#!/bin/bash
"""
Unified cluster launcher for LSF and SLURM systems.
Auto-detects the cluster type and submits jobs appropriately.
"""

import os
import sys
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml

# Get the experiments directory
EXPERIMENTS_DIR = Path(__file__).parent.parent
CORE_DIR = EXPERIMENTS_DIR / "core"

# Add core to Python path
sys.path.insert(0, str(CORE_DIR))

from utils import detect_cluster_environment, get_cluster_job_info


def detect_available_resources() -> Dict[str, Any]:
    """
    Detect available computational resources on the cluster.
    
    Returns:
        Dictionary with resource information
    """
    cluster_type = detect_cluster_environment()
    resources = {"cluster_type": cluster_type}
    
    if cluster_type == "lsf":
        # Query LSF for available hosts and GPUs
        try:
            result = subprocess.run(
                ["bhosts", "-w"], capture_output=True, text=True, check=True
            )
            hosts = []
            for line in result.stdout.split('\n')[1:]:  # Skip header
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 2 and parts[1] == "ok":
                        hosts.append(parts[0])
            resources["available_hosts"] = hosts
            
            # Try to get GPU information
            try:
                gpu_result = subprocess.run(
                    ["bqueues", "-l"], capture_output=True, text=True, check=True
                )
                # Parse GPU queues (simplified)
                gpu_queues = []
                for line in gpu_result.stdout.split('\n'):
                    if "gpu" in line.lower():
                        queue_name = line.split()[0]
                        gpu_queues.append(queue_name)
                resources["gpu_queues"] = gpu_queues
            except subprocess.CalledProcessError:
                resources["gpu_queues"] = ["gpu"]  # Default
                
        except subprocess.CalledProcessError:
            resources["available_hosts"] = []
            resources["gpu_queues"] = ["gpu"]
    
    elif cluster_type == "slurm":
        # Query SLURM for available nodes and GPUs
        try:
            result = subprocess.run(
                ["sinfo", "-h", "-o", "%N %T %G"], capture_output=True, text=True, check=True
            )
            nodes = []
            gpu_nodes = []
            for line in result.stdout.split('\n'):
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 2 and parts[1] == "idle":
                        nodes.append(parts[0])
                        if len(parts) >= 3 and "gpu" in parts[2]:
                            gpu_nodes.append(parts[0])
            
            resources["available_nodes"] = nodes
            resources["gpu_nodes"] = gpu_nodes
            
            # Get partition information
            try:
                partition_result = subprocess.run(
                    ["sinfo", "-h", "-o", "%P"], capture_output=True, text=True, check=True
                )
                partitions = [line.strip() for line in partition_result.stdout.split('\n') if line.strip()]
                resources["partitions"] = partitions
            except subprocess.CalledProcessError:
                resources["partitions"] = ["gpu", "compute"]
                
        except subprocess.CalledProcessError:
            resources["available_nodes"] = []
            resources["gpu_nodes"] = []
            resources["partitions"] = ["gpu", "compute"]
    
    return resources


def create_lsf_script(
    job_name: str,
    config_path: str,
    output_dir: str,
    num_gpus: int = 8,
    queue: str = "gpu",
    walltime: str = "24:00",
    memory_gb: int = 64,
    additional_args: Optional[List[str]] = None,
) -> str:
    """
    Create LSF job script for distributed training.
    
    Args:
        job_name: Name of the job
        config_path: Path to configuration file
        output_dir: Output directory for results
        num_gpus: Number of GPUs to request
        queue: LSF queue name
        walltime: Wall time limit (HH:MM format)
        memory_gb: Memory requirement in GB
        additional_args: Additional arguments for training script
    
    Returns:
        Path to created LSF script
    """
    additional_args = additional_args or []
    
    # Determine if we need multiple hosts
    gpus_per_host = 8  # Assume 8 GPUs per host (common for modern systems)
    num_hosts = max(1, (num_gpus + gpus_per_host - 1) // gpus_per_host)
    
    # Create LSF script content
    script_content = f"""#!/bin/bash
#BSUB -J {job_name}
#BSUB -q {queue}
#BSUB -n {num_gpus}
#BSUB -gpu "num={min(num_gpus, gpus_per_host)}:mode=exclusive_process"
#BSUB -R "span[hosts={num_hosts}]"
#BSUB -W {walltime}
#BSUB -M {memory_gb * 1024}
#BSUB -o {output_dir}/lsf_logs/{job_name}_%J.out
#BSUB -e {output_dir}/lsf_logs/{job_name}_%J.err

# Set up environment
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=$LSB_GPU_INFOS

# Create output directories
mkdir -p {output_dir}
mkdir -p {output_dir}/lsf_logs
mkdir -p {output_dir}/checkpoints
mkdir -p {output_dir}/tensorboard_logs

# Set up distributed training environment
export MASTER_ADDR=$(echo $LSB_HOSTS | cut -d' ' -f1)
export MASTER_PORT=12355
export WORLD_SIZE={num_gpus}

# Launch distributed training
cd {EXPERIMENTS_DIR}

if [ {num_gpus} -gt 1 ]; then
    # Multi-GPU training
    python -m torch.distributed.run \\
        --nnodes={num_hosts} \\
        --nproc_per_node={min(num_gpus, gpus_per_host)} \\
        --master_addr=$MASTER_ADDR \\
        --master_port=$MASTER_PORT \\
        unified_train.py \\
        --config {config_path} \\
        --output_dir {output_dir} \\
        {' '.join(additional_args)}
else
    # Single GPU training
    python unified_train.py \\
        --config {config_path} \\
        --output_dir {output_dir} \\
        {' '.join(additional_args)}
fi

echo "Training completed at $(date)"
"""
    
    # Write script to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.lsf', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    
    return script_path


def create_slurm_script(
    job_name: str,
    config_path: str,
    output_dir: str,
    num_gpus: int = 8,
    partition: str = "gpu",
    walltime: str = "24:00:00",
    memory_gb: int = 64,
    additional_args: Optional[List[str]] = None,
) -> str:
    """
    Create SLURM job script for distributed training.
    
    Args:
        job_name: Name of the job
        config_path: Path to configuration file
        output_dir: Output directory for results
        num_gpus: Number of GPUs to request
        partition: SLURM partition name
        walltime: Wall time limit (HH:MM:SS format)
        memory_gb: Memory requirement in GB
        additional_args: Additional arguments for training script
    
    Returns:
        Path to created SLURM script
    """
    additional_args = additional_args or []
    
    # Determine number of nodes needed
    gpus_per_node = 8  # Assume 8 GPUs per node
    num_nodes = max(1, (num_gpus + gpus_per_node - 1) // gpus_per_node)
    ntasks_per_node = min(num_gpus, gpus_per_node)
    
    # Create SLURM script content
    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --gres=gpu:{min(num_gpus, gpus_per_node)}
#SBATCH --time={walltime}
#SBATCH --mem={memory_gb}G
#SBATCH --output={output_dir}/slurm_logs/{job_name}_%j.out
#SBATCH --error={output_dir}/slurm_logs/{job_name}_%j.err

# Set up environment
export OMP_NUM_THREADS=4

# Create output directories
mkdir -p {output_dir}
mkdir -p {output_dir}/slurm_logs
mkdir -p {output_dir}/checkpoints
mkdir -p {output_dir}/tensorboard_logs

# Set up distributed training environment
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=12355

# Launch distributed training
cd {EXPERIMENTS_DIR}

if [ {num_gpus} -gt 1 ]; then
    # Multi-GPU training
    srun python unified_train.py \\
        --config {config_path} \\
        --output_dir {output_dir} \\
        {' '.join(additional_args)}
else
    # Single GPU training
    python unified_train.py \\
        --config {config_path} \\
        --output_dir {output_dir} \\
        {' '.join(additional_args)}
fi

echo "Training completed at $(date)"
"""
    
    # Write script to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.slurm', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    
    return script_path


def submit_job(script_path: str, cluster_type: str) -> str:
    """
    Submit job to cluster.
    
    Args:
        script_path: Path to job script
        cluster_type: Type of cluster ("lsf" or "slurm")
    
    Returns:
        Job ID
    """
    if cluster_type == "lsf":
        result = subprocess.run(
            ["bsub", "<", script_path], 
            capture_output=True, text=True, shell=True, check=True
        )
        # Parse job ID from LSF output
        for line in result.stdout.split('\n'):
            if "Job <" in line:
                job_id = line.split('<')[1].split('>')[0]
                return job_id
    
    elif cluster_type == "slurm":
        result = subprocess.run(
            ["sbatch", script_path], 
            capture_output=True, text=True, check=True
        )
        # Parse job ID from SLURM output
        for line in result.stdout.split('\n'):
            if "Submitted batch job" in line:
                job_id = line.split()[-1]
                return job_id
    
    raise RuntimeError(f"Failed to parse job ID from cluster output")


def main():
    parser = argparse.ArgumentParser(description="Unified cluster launcher for RAT experiments")
    
    # Required arguments
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to experiment configuration file")
    parser.add_argument("--experiment-type", type=str, required=True,
                       choices=["medical_segmentation", "object_detection", "custom"],
                       help="Type of experiment")
    
    # Resource specification
    parser.add_argument("--num-gpus", type=int, default=8,
                       help="Number of GPUs to request (default: 8)")
    parser.add_argument("--memory", type=int, default=64,
                       help="Memory requirement in GB (default: 64)")
    parser.add_argument("--walltime", type=str, default="24:00",
                       help="Wall time (LSF: HH:MM, SLURM: HH:MM:SS)")
    
    # Cluster-specific options
    parser.add_argument("--queue", type=str, help="LSF queue name")
    parser.add_argument("--partition", type=str, help="SLURM partition name")
    
    # Output and job options
    parser.add_argument("--output-dir", type=str, default="./outputs",
                       help="Output directory for results")
    parser.add_argument("--job-name", type=str, help="Job name (auto-generated if not provided)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Generate scripts but don't submit jobs")
    
    # Additional training arguments
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Detect cluster environment
    cluster_type = detect_cluster_environment()
    if cluster_type == "local":
        print("Warning: No cluster environment detected. Use local training instead.")
        sys.exit(1)
    
    print(f"Detected cluster type: {cluster_type}")
    
    # Detect available resources
    resources = detect_available_resources()
    print(f"Available resources: {resources}")
    
    # Load configuration to get experiment details
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate job name if not provided
    if args.job_name:
        job_name = args.job_name
    else:
        experiment_name = config.get("experiment_name", "rat_experiment")
        job_name = f"{experiment_name}_{args.experiment_type}"
    
    # Prepare additional arguments
    additional_args = []
    if args.resume:
        additional_args.extend(["--resume", args.resume])
    if args.debug:
        additional_args.append("--debug")
    
    # Create output directory
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create cluster script
    if cluster_type == "lsf":
        queue = args.queue or resources.get("gpu_queues", ["gpu"])[0]
        script_path = create_lsf_script(
            job_name=job_name,
            config_path=str(config_path.resolve()),
            output_dir=str(output_dir),
            num_gpus=args.num_gpus,
            queue=queue,
            walltime=args.walltime,
            memory_gb=args.memory,
            additional_args=additional_args,
        )
        print(f"Created LSF script: {script_path}")
    
    elif cluster_type == "slurm":
        partition = args.partition or resources.get("partitions", ["gpu"])[0]
        # Convert LSF time format to SLURM if needed
        walltime = args.walltime
        if ":" in walltime and len(walltime.split(":")) == 2:
            walltime = walltime + ":00"  # Add seconds
        
        script_path = create_slurm_script(
            job_name=job_name,
            config_path=str(config_path.resolve()),
            output_dir=str(output_dir),
            num_gpus=args.num_gpus,
            partition=partition,
            walltime=walltime,
            memory_gb=args.memory,
            additional_args=additional_args,
        )
        print(f"Created SLURM script: {script_path}")
    
    # Display script content
    print("\n" + "="*60)
    print("Generated script content:")
    print("="*60)
    with open(script_path, 'r') as f:
        print(f.read())
    print("="*60)
    
    # Submit job unless dry run
    if args.dry_run:
        print(f"Dry run mode: Script created but not submitted")
        print(f"To submit manually:")
        if cluster_type == "lsf":
            print(f"  bsub < {script_path}")
        elif cluster_type == "slurm":
            print(f"  sbatch {script_path}")
    else:
        try:
            job_id = submit_job(script_path, cluster_type)
            print(f"Job submitted successfully!")
            print(f"Job ID: {job_id}")
            print(f"Output directory: {output_dir}")
            
            # Provide monitoring commands
            if cluster_type == "lsf":
                print(f"Monitor job: bjobs {job_id}")
                print(f"View output: tail -f {output_dir}/lsf_logs/{job_name}_{job_id}.out")
            elif cluster_type == "slurm":
                print(f"Monitor job: squeue -j {job_id}")
                print(f"View output: tail -f {output_dir}/slurm_logs/{job_name}_{job_id}.out")
            
        except subprocess.CalledProcessError as e:
            print(f"Error submitting job: {e}")
            print(f"Command output: {e.stdout}")
            print(f"Command error: {e.stderr}")
            sys.exit(1)
    
    # Clean up temporary script if job was submitted
    if not args.dry_run:
        os.unlink(script_path)


if __name__ == "__main__":
    main()