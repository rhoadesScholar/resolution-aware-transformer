#!/bin/bash
# Kill any hanging Ray processes and submit a new job

echo "Checking for running Ray jobs..."

# Check for any ray_job processes
if pgrep -f "ray_job" > /dev/null; then
    echo "Found running Ray job processes. Please kill them manually if needed."
    pgrep -f "ray_job"
fi

echo "Submitting new Ray job with timeout fixes..."
cd /nrs/cellmap/rhoadesj/resolution-aware-transformer

# Submit the updated Ray job
./experiments/submit_ray_job.sh 1 --config experiments/configs/medical_segmentation.yaml --num-gpus 8