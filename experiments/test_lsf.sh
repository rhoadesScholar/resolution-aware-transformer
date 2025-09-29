#!/bin/bash
# Simple test to verify LSF system is working

echo "Testing LSF submission..."

bsub <<EOF
#BSUB -J test_lsf
#BSUB -n 1
#BSUB -q gpu_h100
#BSUB -gpu "num=1"
#BSUB -R "rusage[mem=1000,ngpus_excl_p=1]"
#BSUB -o /nrs/cellmap/rhoadesj/resolution-aware-transformer/experiments/results/lsf_logs/test_%J.out
#BSUB -e /nrs/cellmap/rhoadesj/resolution-aware-transformer/experiments/results/lsf_logs/test_%J.err
#BSUB -W 0:05

echo "=== LSF Test Job ==="
echo "Job ID: \$LSB_JOBID"
echo "Hostname: \$(hostname)"
echo "Date: \$(date)"
echo "Working directory: \$(pwd)"
echo "Environment:"
env | grep -E "(LSB_|CUDA_|GPU)"
echo "=== Test Complete ==="
EOF