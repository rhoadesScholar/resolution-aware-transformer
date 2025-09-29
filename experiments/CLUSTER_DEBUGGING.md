Ray on LSF with NCCL — Debugging Playbook (task-agnostic)

This guide documents cluster-related setup only (no dataset or model specifics). Use it with VS Code Copilot so it can ground its suggestions in the correct Ray + LSF + NCCL environment.

⸻

1. Ray client joins the LSF-launched cluster

Use this Python entrypoint to attach to the cluster and configure Ray Train.

# cluster_bootstrap.py
import os
import ray
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer
from ray.train.torch.config import TorchConfig

def train_loop_per_worker(cfg):
    """Your task-specific train loop (data/model not shown).
    Cluster logic stays minimal here."""
    import torch
    from torch.nn.parallel import DistributedDataParallel as DDP
    ctx = ray.train.get_context()
    local_rank = ctx.get_local_rank()
    torch.cuda.set_device(local_rank)
    # Initialize your model/optimizer/dataloaders here…

    # Call ray.train.report({...}) inside your loop

if __name__ == "__main__":
    ray.init(address="auto")  # Always attach to the LSF-launched cluster
    scaling = ScalingConfig(
        num_workers=int(os.environ.get("WORLD_GPUS")),
        use_gpu=True,
        resources_per_worker={"CPU": 11, "GPU": 1},
    )
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={},
        scaling_config=scaling,
        torch_config=TorchConfig(backend="nccl", timeout_s=1800),
        run_config=RunConfig(
            name="job_name_here",
            storage_path=f"/scratch/{os.getenv('USER','unknown')}/ray_results"
        ),
    )
    result = trainer.fit()
    print("Result:", result)

Key points:
	•	ray.init(address="auto") — never start a local cluster.
	•	resources_per_worker={"CPU": 11, "GPU": 1} — matches 96c/8g per node.
	•	TorchConfig(... timeout_s=1800) — avoids NCCL timeouts on Ethernet.

⸻

2. LSF submission script to start the Ray cluster

This script sets up the Ray head + workers, applies NCCL environment vars, and runs your entrypoint.

#!/bin/bash
# submit_ray_job.sh
# Usage: ./submit_ray_job.sh <num_nodes> [--queue=QUEUE] [args...]
set -euo pipefail

NUM_NODES=${1:?Usage: $0 <num_nodes> [--queue=...]}; shift || true
QUEUE="gpu_h200_parallel"
TRAINING_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in --queue=*) QUEUE="${1#*=}";; *) TRAINING_ARGS+=("$1");; esac; shift
done

NUM_CPUS=$((NUM_NODES * 96))
NUM_GPUS=$((NUM_NODES * 8))
WALLTIME="4:00"; [[ $NUM_NODES -gt 2 ]] && WALLTIME="8:00"

bsub <<EOF
#BSUB -J ray_${NUM_NODES}n
#BSUB -n ${NUM_CPUS}
#BSUB -app parallel-96
#BSUB -q ${QUEUE}
#BSUB -gpu "num=8:mode=exclusive_process"
#BSUB -R "span[ptile=96]"
#BSUB -o ray_job_%J.out
#BSUB -e ray_job_%J.err
#BSUB -W ${WALLTIME}

source ~/ray_env/bin/activate

# NCCL ethernet-optimized settings
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_BUFFSIZE=8388608
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

# Per-user Ray temp dir
export RAY_TMPDIR="/scratch/\$(whoami)/ray_\$(whoami)"; mkdir -p "\$RAY_TMPDIR"

# Pick a free TCP port for the Ray head
getfreeport() { local p c=1; while [[ \$c -eq 1 ]]; do p=\$(( (RANDOM % 40000) + 20000 )); (netstat -a | grep -q "\$p") && c=1 || c=0; done; echo \$p; }

hosts=(); while read -r h; do [[ -n "\$h" ]] && hosts+=("\$h"); done < <(cat \$LSB_DJOB_HOSTFILE | uniq)
head_node="\${hosts[0]}"; port=\$(getfreeport)

for h in "\${hosts[@]}"; do blaunch -z "\$h" "mkdir -p /scratch/\$(whoami)/ray_\$(whoami)"; done

# Start head
blaunch -z "\$head_node" "source ~/ray_env/bin/activate; \
  export NCCL_DEBUG=INFO NCCL_IB_DISABLE=1 NCCL_NET_GDR_LEVEL=0 NCCL_BUFFSIZE=8388608 NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0; \
  ray start --head --port \$port --num-cpus=96 --num-gpus=8 --temp-dir=/scratch/\$(whoami)/ray_\$(whoami)"

until ray status --address "\$head_node:\$port"; do echo "waiting for head…"; sleep 3; done

# Start workers
if [[ ${NUM_NODES} -gt 1 ]]; then
  for h in "\${hosts[@]:1}"; do
    blaunch -z "\$h" "source ~/ray_env/bin/activate; \
      export NCCL_DEBUG=INFO NCCL_IB_DISABLE=1 NCCL_NET_GDR_LEVEL=0 NCCL_BUFFSIZE=8388608 NCCL_P2P_DISABLE=0 NCCL_SHM_DISABLE=0; \
      ray start --address \$head_node:\$port --num-cpus=96 --num-gpus=8 --temp-dir=/scratch/\$(whoami)/ray_\$(whoami)"
    until blaunch -z "\$h" ray status --address "\$head_node:\$port"; do echo "waiting for \$h…"; sleep 3; done
  done
fi

# Expose world size for Python
export RAY_ADDRESS="\$head_node:\$port"
export WORLD_GPUS=${NUM_GPUS}

python cluster_bootstrap.py \${TRAINING_ARGS[@]}
ec=\$?

for h in "\${hosts[@]}"; do blaunch -z "\$h" "source ~/ray_env/bin/activate; ray stop" || true; done
exit \$ec
EOF

Key features:
	•	Exports NCCL ethernet-optimized environment vars.
	•	Dynamic port assignment + readiness checks.
	•	Head node first, then workers.
	•	Per-user scratch dirs under /scratch/$USER/.
	•	Graceful shutdown of all Ray processes.

⸻

3. How to run

# One node (8 GPUs)
./submit_ray_job.sh 1 --queue=gpu_h200_parallel

# Two nodes (16 GPUs)
./submit_ray_job.sh 2 --queue=gpu_h100_parallel

# Monitor
bjobs
bpeek <job_id>
cat ray_job_<job_id>.out


⸻

4. Debugging guide for Copilot

When you ask Copilot for help, prompt it to:
	•	Verify ray.init(address="auto") exists in Python.
	•	Confirm TorchConfig(backend="nccl", timeout_s=1800) is present.
	•	Ensure resources_per_worker={"CPU": 11, "GPU": 1} in ScalingConfig.
	•	Check NCCL env block in the LSF script.
	•	Confirm ray status loops are in place.
	•	Ensure cleanup with ray stop runs on all nodes.

⸻

5. Common failures
	•	NCCL timeouts → increase timeout_s, check NCCL env.
	•	Port in use → regenerate free port with getfreeport.
	•	CUDA device errors → let Ray handle CUDA_VISIBLE_DEVICES.
	•	Permission denied → verify /scratch/$USER dirs.

⸻

This document contains only cluster-related boilerplate. Add your own dataset/task code in the train loop.