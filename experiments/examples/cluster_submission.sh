#!/bin/bash
"""
Example: Cluster Submission
Demonstrates how to submit jobs to LSF or SLURM clusters using the unified launcher.
"""

set -e  # Exit on error

# Configuration
CONFIG_FILE="configs/unified_template.yaml"
EXPERIMENT_TYPE="segmentation"
NUM_GPUS=8
WALLTIME="24:00"
MEMORY_GB=128
OUTPUT_DIR="/shared/outputs/cluster_example"

echo "=== Cluster Submission Example ==="
echo "Submitting RAT training job to cluster"
echo ""

# Check if we're on a cluster
if [[ -n "$SLURM_JOB_ID" || -n "$LSB_JOBID" ]]; then
    echo "Error: This script should be run from a login node, not within a job."
    exit 1
fi

# Detect cluster type
if command -v sbatch &> /dev/null; then
    CLUSTER_TYPE="SLURM"
    echo "Detected SLURM cluster"
elif command -v bsub &> /dev/null; then
    CLUSTER_TYPE="LSF"
    echo "Detected LSF cluster"
else
    echo "Error: No cluster scheduler detected (sbatch or bsub not found)"
    echo "This example requires a SLURM or LSF cluster environment"
    exit 1
fi

echo "Configuration:"
echo "  Cluster type: $CLUSTER_TYPE"
echo "  Config file: $CONFIG_FILE"
echo "  Experiment type: $EXPERIMENT_TYPE"
echo "  Number of GPUs: $NUM_GPUS"
echo "  Wall time: $WALLTIME"
echo "  Memory: ${MEMORY_GB}GB"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Method 1: Interactive submission (Recommended)
echo "=== Method 1: Interactive Submission ==="
echo "This will submit the job and monitor it interactively."
echo ""

python cluster_launcher.py \
    --config "$CONFIG_FILE" \
    --experiment-type "$EXPERIMENT_TYPE" \
    --num-gpus $NUM_GPUS \
    --walltime "$WALLTIME" \
    --memory $MEMORY_GB \
    --output-dir "$OUTPUT_DIR" \
    --job-name "rat_${EXPERIMENT_TYPE}_${NUM_GPUS}gpu"

echo ""
echo "Job submitted successfully!"
echo ""

# Method 2: Dry run (Generate scripts without submitting)
echo "=== Method 2: Dry Run Example ==="
echo "Generating scripts without submitting (for inspection)..."
echo ""

python cluster_launcher.py \
    --config "$CONFIG_FILE" \
    --experiment-type "$EXPERIMENT_TYPE" \
    --num-gpus $NUM_GPUS \
    --walltime "$WALLTIME" \
    --memory $MEMORY_GB \
    --output-dir "$OUTPUT_DIR" \
    --job-name "rat_${EXPERIMENT_TYPE}_dryrun" \
    --dry-run

echo ""

# Method 3: Cluster-specific options
echo "=== Method 3: Cluster-Specific Options ==="

if [[ "$CLUSTER_TYPE" == "SLURM" ]]; then
    echo "SLURM-specific submission with custom partition..."
    
    # Check available partitions
    echo "Available partitions:"
    sinfo -h -o "%P" | sort | uniq
    
    echo ""
    echo "Submitting to GPU partition..."
    python cluster_launcher.py \
        --config "$CONFIG_FILE" \
        --experiment-type "$EXPERIMENT_TYPE" \
        --num-gpus 4 \
        --partition gpu \
        --walltime "12:00:00" \
        --memory 64 \
        --output-dir "$OUTPUT_DIR" \
        --job-name "rat_slurm_example" \
        --dry-run

elif [[ "$CLUSTER_TYPE" == "LSF" ]]; then
    echo "LSF-specific submission with custom queue..."
    
    # Check available queues
    echo "Available queues:"
    bqueues | grep -E "(gpu|GPU)" || echo "No GPU queues found"
    
    echo ""
    echo "Submitting to GPU queue..."
    python cluster_launcher.py \
        --config "$CONFIG_FILE" \
        --experiment-type "$EXPERIMENT_TYPE" \
        --num-gpus 4 \
        --queue gpu \
        --walltime "12:00" \
        --memory 64 \
        --output-dir "$OUTPUT_DIR" \
        --job-name "rat_lsf_example" \
        --dry-run
fi

echo ""
echo "=== Job Monitoring Commands ==="

if [[ "$CLUSTER_TYPE" == "SLURM" ]]; then
    echo "Monitor jobs:     squeue -u $USER"
    echo "Job details:      scontrol show job <job_id>"
    echo "Cancel job:       scancel <job_id>"
    echo "View output:      tail -f $OUTPUT_DIR/slurm_logs/rat_*.out"

elif [[ "$CLUSTER_TYPE" == "LSF" ]]; then
    echo "Monitor jobs:     bjobs -u $USER"
    echo "Job details:      bjobs -l <job_id>"
    echo "Cancel job:       bkill <job_id>"
    echo "View output:      tail -f $OUTPUT_DIR/lsf_logs/rat_*.out"
fi

echo ""
echo "=== Advanced Usage ==="
echo ""
echo "1. Resume from checkpoint:"
echo "   python cluster_launcher.py --config $CONFIG_FILE --experiment-type $EXPERIMENT_TYPE --resume /path/to/checkpoint.pth"
echo ""
echo "2. Multiple experiments:"
echo "   for config in configs/*.yaml; do"
echo "     python cluster_launcher.py --config \$config --experiment-type $EXPERIMENT_TYPE --num-gpus 4"
echo "   done"
echo ""
echo "3. Parameter sweeps:"
echo "   for lr in 1e-4 2e-4 5e-4; do"
echo "     python cluster_launcher.py --config $CONFIG_FILE --experiment-type $EXPERIMENT_TYPE --job-name \"rat_lr_\$lr\""
echo "   done"
echo ""
echo "4. Debug on single GPU:"
echo "   python cluster_launcher.py --config $CONFIG_FILE --experiment-type $EXPERIMENT_TYPE --num-gpus 1 --debug"