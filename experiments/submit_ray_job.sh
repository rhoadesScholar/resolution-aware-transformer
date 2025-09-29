#!/bin/bash
# submit_ray_job.sh
# Usage: ./submit_ray_job.sh <num_nodes> [--queue=QUEUE] [args...]
set -euo pipefail

NUM_NODES=${1:?Usage: $0 <num_nodes> [--queue=...]}; shift || true
QUEUE="gpu_h100"
TRAINING_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in --queue=*) QUEUE="${1#*=}";; *) TRAINING_ARGS+=("$1");; esac; shift
done

NUM_CPUS=$((NUM_NODES * 96))
NUM_GPUS=$((NUM_NODES * 8))
WALLTIME="4:00"; [[ $NUM_NODES -gt 2 ]] && WALLTIME="8:00"

# Ensure the log directory exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/results/lsf_logs"
mkdir -p "${LOG_DIR}"

echo "Submitting Ray job with ${NUM_NODES} nodes (${NUM_GPUS} GPUs) to queue ${QUEUE}"
echo "Log files will be in: ${LOG_DIR}"

bsub <<EOF
#BSUB -J ray_${NUM_NODES}n
#BSUB -n ${NUM_CPUS}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4000,ngpus_excl_p=8]"
#BSUB -q ${QUEUE}
#BSUB -gpu "num=8"
#BSUB -o ${LOG_DIR}/ray_job_%J.out
#BSUB -e ${LOG_DIR}/ray_job_%J.err
#BSUB -W ${WALLTIME}

# Change to the correct directory
cd /nrs/cellmap/rhoadesj/resolution-aware-transformer

# Create log directory if it doesn't exist (redundant safety)
mkdir -p experiments/results/lsf_logs

echo "=== Job started at \$(date) ==="
echo "Job ID: \$LSB_JOBID"
echo "Working directory: \$(pwd)"

# NCCL ethernet-optimized settings
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_BUFFSIZE=8388608
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

# Additional NCCL settings for cluster stability
export NCCL_TIMEOUT_S=7200
export NCCL_BLOCKING_WAIT=1

# Per-user Ray temp dir
export RAY_TMPDIR="/scratch/\$(whoami)/ray_\$(whoami)"; mkdir -p "\$RAY_TMPDIR"

# Pick a free TCP port for the Ray head
getfreeport() { local p c=1; while [[ \$c -eq 1 ]]; do p=\$(( (RANDOM % 40000) + 20000 )); (netstat -a | grep -q "\$p") && c=1 || c=0; done; echo \$p; }

hosts=(); while read -r h; do [[ -n "\$h" ]] && hosts+=("\$h"); done < <(cat \$LSB_DJOB_HOSTFILE | uniq)
head_node="\${hosts[0]}"; port=\$(getfreeport)

echo "=== Ray Cluster Setup ==="
echo "Head node: \$head_node"
echo "Port: \$port"
echo "Hosts: \${hosts[@]}"
echo "Working directory: \$(pwd)"

for h in "\${hosts[@]}"; do blaunch -z "\$h" "mkdir -p /scratch/\$(whoami)/ray_\$(whoami)"; done

# Start head
echo "Starting Ray head on \$head_node:\$port"
blaunch -z "\$head_node" "cd \$(pwd); \
  export NCCL_DEBUG=INFO NCCL_IB_DISABLE=1 NCCL_NET_GDR_LEVEL=0 NCCL_BUFFSIZE=8388608 NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0 NCCL_TIMEOUT_S=7200 NCCL_BLOCKING_WAIT=1; \
  nohup ray start --head --port \$port --num-cpus=96 --num-gpus=8 --temp-dir=/scratch/\$(whoami)/ray_\$(whoami) > /tmp/ray_head_\$port.log 2>&1 &"

echo "Waiting for Ray head to initialize..."
# Give Ray a moment to fully initialize  
sleep 10

# Test connection to Ray head
echo "Testing Ray head connection..."
if python -c "import ray; ray.init(address='\$head_node:\$port'); print('Connected!'); ray.shutdown()" 2>/dev/null; then
    echo "Ray head started successfully!"
else
    echo "Warning: Could not connect to Ray head, but proceeding anyway..."
fi

# Start workers
if [[ ${NUM_NODES} -gt 1 ]]; then
  echo "Starting Ray workers..."
  for h in "\${hosts[@]:1}"; do
    echo "Starting worker on \$h"
    blaunch -z "\$h" "cd \$(pwd); \
      export NCCL_DEBUG=INFO NCCL_IB_DISABLE=1 NCCL_NET_GDR_LEVEL=0 NCCL_BUFFSIZE=8388608 NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0 NCCL_TIMEOUT_S=7200 NCCL_BLOCKING_WAIT=1; \
      ray start --address \$head_node:\$port --num-cpus=96 --num-gpus=8 --temp-dir=/scratch/\$(whoami)/ray_\$(whoami)"
    # Give each worker time to connect, but don't wait for status check
    sleep 5
  done
fi

# Expose world size for Python
export RAY_ADDRESS="\$head_node:\$port"
export WORLD_GPUS=${NUM_GPUS}

echo "=== Ray Cluster Ready ==="
# Skip the problematic ray status check
echo "Ray cluster address: \$RAY_ADDRESS"
echo "Proceeding to training..."

echo "=== Starting Training ==="
python experiments/cluster_bootstrap.py \${TRAINING_ARGS[@]}
ec=\$?

echo "=== Cleaning Up Ray Cluster ==="
for h in "\${hosts[@]}"; do blaunch -z "\$h" "ray stop" || true; done
exit \$ec
EOF