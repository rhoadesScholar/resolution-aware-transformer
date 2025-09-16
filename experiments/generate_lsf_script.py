#!/usr/bin/env python3
"""
Generate LSF job scripts using configuration file.
"""

import sys
import os
import argparse
from pathlib import Path

# Add experiments directory to path
local_path = str(Path(__file__).resolve().parent)
sys.path.insert(0, local_path)
os.chdir(local_path)
from config_manager import load_config


def generate_parallel_job_scripts(config, output_dir="cluster"):
    """Generate separate LSF job scripts for each experiment type."""

    experiments_config = config.get_experiments_config()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_scripts = []
    job_dependencies = []

    # Define experiment types and their configurations
    experiment_types = []
    if experiments_config["medical_segmentation"]:
        experiment_types.append(
            {
                "name": "medical_segmentation",
                "config_dir": "medical_segmentation/configs",
                "job_suffix": "medseg",
            }
        )

    if experiments_config["object_detection"]:
        experiment_types.append(
            {
                "name": "object_detection",
                "config_dir": "object_detection/configs",
                "job_suffix": "objdet",
            }
        )

    if experiments_config["ablation_studies"]:
        experiment_types.append(
            {
                "name": "ablation_studies",
                "config_dir": "ablations/configs",
                "job_suffix": "ablation",
            }
        )

    if experiments_config["robustness_tests"]:
        experiment_types.append(
            {
                "name": "robustness_tests",
                "config_dir": "robustness",
                "job_suffix": "robust",
            }
        )

    # Generate script for each experiment type
    for i, exp_type in enumerate(experiment_types):
        # Create modified config for this experiment
        exp_config = config

        # Generate script content for this specific experiment
        script_content = generate_single_experiment_script(
            config,
            exp_type,
            dependency_job_name=(
                job_dependencies[-1]
                if job_dependencies and experiments_config["job_dependencies"]
                else None
            ),
        )

        # Write script file
        script_file = output_dir / f"submit_{exp_type['job_suffix']}.lsf"
        with open(script_file, "w") as f:
            f.write(script_content)
        script_file.chmod(0o755)

        generated_scripts.append(script_file)
        job_dependencies.append(f"rat_{exp_type['job_suffix']}")

        print(f"Generated parallel job script: {script_file}")

    # Generate master submission script
    master_script = generate_master_submission_script(
        config, experiment_types, experiments_config["job_dependencies"]
    )
    master_file = output_dir / "submit_all_parallel.lsf"
    with open(master_file, "w") as f:
        f.write(master_script)
    master_file.chmod(0o755)
    generated_scripts.append(master_file)

    return generated_scripts


def generate_data_setup_for_experiment(exp_type):
    """Generate experiment-specific data setup code."""
    
    if exp_type["name"] == "medical_segmentation":
        return [
            "# Data setup - Medical segmentation only needs ISIC dataset",
            'echo "Setting up ISIC dataset for medical segmentation..."',
            "",
            "# Add small random delay to reduce I/O contention if multiple jobs start simultaneously",
            "sleep $((RANDOM % 30 + 10))  # Random delay between 10-40 seconds",
            "",
            'DATASET_CONFIG=$(python3 config_manager.py --dump)',
            'USE_SAMPLE_DATA=$(echo "$DATASET_CONFIG" | grep "use_sample_data" | cut -d\'=\' -f2 | xargs)',
            'ISIC_SOURCE=$(echo "$DATASET_CONFIG" | grep "isic_source_dir" | cut -d\'=\' -f2 | xargs)',
            'ISIC_SAMPLE_SIZE=$(echo "$DATASET_CONFIG" | grep "isic_sample_size" | cut -d\'=\' -f2 | xargs)',
            "",
            'if [ "$USE_SAMPLE_DATA" = "true" ] || [ -z "$ISIC_SOURCE" ]; then',
            '    echo "Creating sample ISIC2018 dataset..."',
            "    python scripts/setup_isic.py \\",
            "        --sample_only \\",
            '        --output_dir "$LOCAL_DATA_DIR/sample_ISIC2018" \\',
            "        --num_samples ${ISIC_SAMPLE_SIZE:-100}",
            "else",
            '    if [ -d "$ISIC_SOURCE" ]; then',
            '        echo "Copying ISIC2018 from $ISIC_SOURCE to local storage..."',
            '        rsync -av "$ISIC_SOURCE/" "$LOCAL_DATA_DIR/ISIC2018/"',
            "    else",
            '        echo "Warning: ISIC source directory not found: $ISIC_SOURCE"',
            '        echo "Falling back to sample dataset..."',
            "        python scripts/setup_isic.py \\",
            "            --sample_only \\",
            '            --output_dir "$LOCAL_DATA_DIR/sample_ISIC2018" \\',
            "            --num_samples ${ISIC_SAMPLE_SIZE:-100}",
            "    fi",
            "fi",
            "",
            'echo "ISIC dataset setup completed."',
        ]
    
    elif exp_type["name"] == "object_detection":
        return [
            "# Data setup - Object detection only needs COCO dataset",
            'echo "Setting up COCO dataset for object detection..."',
            "",
            "# Add small random delay to reduce I/O contention if multiple jobs start simultaneously",
            "sleep $((RANDOM % 30 + 10))  # Random delay between 10-40 seconds",
            "",
            'DATASET_CONFIG=$(python3 config_manager.py --dump)',
            'USE_SAMPLE_DATA=$(echo "$DATASET_CONFIG" | grep "use_sample_data" | cut -d\'=\' -f2 | xargs)',
            'COCO_SOURCE=$(echo "$DATASET_CONFIG" | grep "coco_source_dir" | cut -d\'=\' -f2 | xargs)',
            'COCO_SAMPLE_SIZE=$(echo "$DATASET_CONFIG" | grep "coco_sample_size" | cut -d\'=\' -f2 | xargs)',
            "",
            'if [ "$USE_SAMPLE_DATA" = "true" ] || [ -z "$COCO_SOURCE" ]; then',
            '    echo "Creating sample COCO2017 dataset..."',
            "    python scripts/setup_coco.py \\",
            "        --sample_only \\",
            '        --output_dir "$LOCAL_DATA_DIR/sample_COCO2017" \\',
            "        --num_samples ${COCO_SAMPLE_SIZE:-50}",
            "else",
            '    if [ -d "$COCO_SOURCE" ]; then',
            '        echo "Copying COCO2017 from $COCO_SOURCE to local storage..."',
            '        rsync -av "$COCO_SOURCE/" "$LOCAL_DATA_DIR/COCO2017/"',
            "    else",
            '        echo "Warning: COCO source directory not found: $COCO_SOURCE"',
            '        echo "Falling back to sample dataset..."',
            "        python scripts/setup_coco.py \\",
            "            --sample_only \\",
            '            --output_dir "$LOCAL_DATA_DIR/sample_COCO2017" \\',
            "            --num_samples ${COCO_SAMPLE_SIZE:-50}",
            "    fi",
            "fi",
            "",
            'echo "COCO dataset setup completed."',
        ]
    
    elif exp_type["name"] == "ablation_studies":
        return [
            "# Data setup - Ablation studies use ISIC dataset (medical segmentation focus)",
            'echo "Setting up ISIC dataset for ablation studies..."',
            "",
            "# Add small random delay to reduce I/O contention if multiple jobs start simultaneously",
            "sleep $((RANDOM % 30 + 10))  # Random delay between 10-40 seconds",
            "",
            'DATASET_CONFIG=$(python3 config_manager.py --dump)',
            'USE_SAMPLE_DATA=$(echo "$DATASET_CONFIG" | grep "use_sample_data" | cut -d\'=\' -f2 | xargs)',
            'ISIC_SOURCE=$(echo "$DATASET_CONFIG" | grep "isic_source_dir" | cut -d\'=\' -f2 | xargs)',
            'ISIC_SAMPLE_SIZE=$(echo "$DATASET_CONFIG" | grep "isic_sample_size" | cut -d\'=\' -f2 | xargs)',
            "",
            'if [ "$USE_SAMPLE_DATA" = "true" ] || [ -z "$ISIC_SOURCE" ]; then',
            '    echo "Creating sample ISIC2018 dataset..."',
            "    python scripts/setup_isic.py \\",
            "        --sample_only \\",
            '        --output_dir "$LOCAL_DATA_DIR/sample_ISIC2018" \\',
            "        --num_samples ${ISIC_SAMPLE_SIZE:-100}",
            "else",
            '    if [ -d "$ISIC_SOURCE" ]; then',
            '        echo "Copying ISIC2018 from $ISIC_SOURCE to local storage..."',
            '        rsync -av "$ISIC_SOURCE/" "$LOCAL_DATA_DIR/ISIC2018/"',
            "    else",
            '        echo "Warning: ISIC source directory not found: $ISIC_SOURCE"',
            '        echo "Falling back to sample dataset..."',
            "        python scripts/setup_isic.py \\",
            "            --sample_only \\",
            '            --output_dir "$LOCAL_DATA_DIR/sample_ISIC2018" \\',
            "            --num_samples ${ISIC_SAMPLE_SIZE:-100}",
            "    fi",
            "fi",
            "",
            'echo "ISIC dataset setup completed."',
        ]
    
    else:
        # Default setup for other experiment types
        return [
            "# Data setup - Default: setup both datasets",
            'echo "Setting up datasets for experiment..."',
            ". scripts/setup_experiment_data.sh",
        ]


def generate_single_experiment_script(config, exp_type, dependency_job_name=None):
    """Generate LSF script for a single experiment type."""

    lsf_config = config.get_lsf_job_config(quick_test=False)
    paths_config = config.get_paths_config()
    dataset_config = config.get_dataset_config()
    training_config = config.get_training_config()

    # Modify job name for this experiment
    job_name = f"rat_{exp_type['job_suffix']}"

    script_lines = [
        "#!/bin/bash",
        f"#BSUB -J {job_name}",
        f"#BSUB -n {lsf_config['total_cpus']}",
        f"#BSUB -gpu \"num={lsf_config['num_gpus']}:mode={lsf_config['gpu_mode']}\"",
    ]

    # Add walltime only if specified
    if lsf_config["walltime_hours"] is not None:
        script_lines.append(f"#BSUB -W {lsf_config['walltime_hours']}:00")

    # Add dependency if specified
    if dependency_job_name:
        script_lines.append(f'#BSUB -w "done({dependency_job_name})"')

    script_lines.extend(
        [
            f"#BSUB -M {lsf_config['total_memory']}",
            f"#BSUB -R \"span[hosts={lsf_config['span_hosts']}]\"",
            f"#BSUB -R \"rusage[mem={lsf_config['total_memory']}]\"",
            f"#BSUB -o {paths_config['results_dir']}/lsf_logs/{job_name}_%J.out",
            f"#BSUB -e {paths_config['results_dir']}/lsf_logs/{job_name}_%J.err",
            f"#BSUB -q {lsf_config['queue']}",
        ]
    )

    if lsf_config["exclusive_node"]:
        script_lines.append("#BSUB -x")

    # Add email notifications if configured
    email = config.get("notifications", "email")
    if email:
        script_lines.append(f"#BSUB -u {email}")
        script_lines.append("#BSUB -N")

    script_lines.extend(
        [
            "",
            f"# RAT {exp_type['name'].replace('_', ' ').title()} Experiment",
            "# Auto-generated for parallel execution",
            "",
            "set -e",
            "",
            'echo "======================================================"',
            f'echo "RAT {exp_type["name"].replace("_", " ").title()} - Training"',
            'echo "Job ID: $LSB_JOBID"',
            'echo "Host: $LSB_HOSTS"',
            'echo "======================================================"',
            "",
            "# Configuration from .config file",
            f"LOCAL_DATA_DIR=\"{config.get_local_data_dir('$LSB_JOBID')}\"",
            f"NETWORK_RESULTS_DIR=\"{paths_config['results_dir']}\"",
            f"NETWORK_CHECKPOINTS_DIR=\"{paths_config['checkpoints_dir']}\"",
            f"NUM_GPUS={lsf_config['num_gpus']}",
            "",
            "# Environment setup",
            'echo "Setting up environment..."',
            "source ~/.bashrc",
        ]
    )

    # Add conda environment if specified
    conda_env = config.get("environment", "conda_env")
    if conda_env:
        script_lines.append(f"conda activate {conda_env}")

    # Add module loading if specified
    modules = config.get("environment", "modules_to_load")
    if modules:
        for module in modules.split(","):
            module = module.strip()
            if module:
                script_lines.append(f"module load {module}")

    script_lines.extend(
        [
            "",
            "# Create directories and setup data",
            'mkdir -p "$LOCAL_DATA_DIR"',
            'mkdir -p "$NETWORK_RESULTS_DIR/lsf_logs"',
            'mkdir -p "$NETWORK_RESULTS_DIR/tensorboard_logs"',
            'mkdir -p "$NETWORK_CHECKPOINTS_DIR"',
            "",
        ]
    )

    # Add experiment-specific data setup
    data_setup_lines = generate_data_setup_for_experiment(exp_type)
    script_lines.extend(data_setup_lines)

    script_lines.extend(
        [
            "",
            'echo "Local data directory contents:"',
            'du -sh "$LOCAL_DATA_DIR"/* 2>/dev/null || echo "No data directories found"',
            "",
            f"# Update cluster configurations for this experiment",
            f'echo "Updating configurations for {exp_type["name"]}..."',
            "python scripts/update_cluster_configs.py \\",
            '    --data_dir "$LOCAL_DATA_DIR" \\',
            '    --results_dir "$NETWORK_RESULTS_DIR" \\',
            '    --checkpoint_dir "$NETWORK_CHECKPOINTS_DIR" \\',
            '    --num_gpus "$NUM_GPUS"',
            "",
            "# Set distributed training environment",
            f"export MASTER_ADDR=$(hostname)",
            f"export MASTER_PORT={training_config['master_port_base']}",
            "export WORLD_SIZE=$NUM_GPUS",
            "",
            f"# Run {exp_type['name']} experiment",
            f'echo "Running {exp_type["name"]} experiment..."',
        ]
    )

    # Add experiment-specific training command
    if exp_type["name"] == "robustness_tests":
        script_lines.extend(
            [
                "python robustness/resolution_transfer.py \\",
                '    --results_dir "$NETWORK_RESULTS_DIR" \\',
                '    --checkpoint_dir "$NETWORK_CHECKPOINTS_DIR"',
            ]
        )
    else:
        script_lines.extend(
            [
                "torchrun \\",
                "    --nnodes=1 \\",
                "    --nproc_per_node=$NUM_GPUS \\",
                "    --master_addr=$MASTER_ADDR \\",
                "    --master_port=$MASTER_PORT \\",
                "    train_distributed.py \\",
                f"    --config_dir {exp_type['config_dir']} \\",
                '    --results_dir "$NETWORK_RESULTS_DIR" \\',
                '    --checkpoint_dir "$NETWORK_CHECKPOINTS_DIR"',
            ]
        )

    script_lines.extend(
        [
            "",
            'echo "Experiment completed successfully!"',
            "",
            "# Cleanup local data if using local storage",
            'if [ "$LOCAL_DATA_DIR" != "$NETWORK_RESULTS_DIR" ]; then',
            '    echo "Cleaning up local data..."',
            '    rm -rf "$LOCAL_DATA_DIR"',
            "fi",
            "",
            'echo "======================================================"',
            f'echo "RAT {exp_type["name"].replace("_", " ").title()} Completed!"',
            'echo "======================================================"',
        ]
    )

    return "\n".join(script_lines)


def generate_master_submission_script(config, experiment_types, use_dependencies):
    """Generate master script to submit all parallel jobs."""

    script_lines = [
        "#!/bin/bash",
        "# Master submission script for RAT parallel experiments",
        "# Auto-generated from configuration",
        "",
        "set -e",
        "",
        'echo "======================================================"',
        'echo "RAT Parallel Experiments - Job Submission"',
        'echo "======================================================"',
        "",
    ]

    if use_dependencies:
        script_lines.extend(
            [
                "# Submit jobs with dependencies (sequential execution)",
                'echo "Submitting jobs with dependencies..."',
                "",
            ]
        )

        for i, exp_type in enumerate(experiment_types):
            script_lines.append(f'echo "Submitting {exp_type["name"]} experiment..."')
            script_lines.append(
                f'JOB_{i+1}=$(bsub < cluster/submit_{exp_type["job_suffix"]}.lsf | grep -oE "[0-9]+")'
            )
            script_lines.append(f'echo "Job ID: $JOB_{i+1}"')
            script_lines.append("")
    else:
        script_lines.extend(
            [
                "# Submit all jobs in parallel (independent execution)",
                'echo "Submitting all jobs in parallel..."',
                "",
            ]
        )

        for i, exp_type in enumerate(experiment_types):
            script_lines.append(f'echo "Submitting {exp_type["name"]} experiment..."')
            script_lines.append(
                f'JOB_{i+1}=$(bsub < cluster/submit_{exp_type["job_suffix"]}.lsf | grep -oE "[0-9]+")'
            )
            script_lines.append(f'echo "Job ID: $JOB_{i+1}"')
            script_lines.append("")

    script_lines.extend(
        [
            'echo "All jobs submitted successfully!"',
            'echo "Monitor with: bjobs"',
            'echo "View logs in: results/lsf_logs/"',
            'echo "======================================================"',
        ]
    )

    return "\n".join(script_lines)


def generate_lsf_script(config, quick_test=False, output_file=None):
    """Generate LSF job script content."""

    # Get configurations
    lsf_config = config.get_lsf_job_config(quick_test=quick_test)
    paths_config = config.get_paths_config()
    cluster_config = config.get_cluster_config()
    dataset_config = config.get_dataset_config()
    training_config = config.get_training_config()
    experiments_config = config.get_experiments_config()

    # Script header
    script_lines = [
        "#!/bin/bash",
        f"#BSUB -J {lsf_config['job_name']}",
        f"#BSUB -n {lsf_config['total_cpus']}",
        f"#BSUB -gpu \"num={lsf_config['num_gpus']}:mode={lsf_config['gpu_mode']}\"",
    ]

    # Add walltime only if specified
    if lsf_config["walltime_hours"] is not None:
        script_lines.append(f"#BSUB -W {lsf_config['walltime_hours']}:00")

    script_lines.extend(
        [
            f"#BSUB -M {lsf_config['total_memory']}",
            f"#BSUB -R \"span[hosts={lsf_config['span_hosts']}]\"",
            f"#BSUB -R \"rusage[mem={lsf_config['total_memory']}]\"",
            f"#BSUB -o {lsf_config['output_file']}",
            f"#BSUB -e {lsf_config['error_file']}",
            f"#BSUB -q {lsf_config['queue']}",
        ]
    )

    # Add exclusive node if configured
    if lsf_config["exclusive_node"]:
        script_lines.append("#BSUB -x")

    # Add email notifications if configured
    email = config.get("notifications", "email")
    if email:
        script_lines.append(f"#BSUB -u {email}")
        events = config.get("notifications", "notification_events", "END,FAIL")
        script_lines.append(f"#BSUB -N -B -E")

    script_lines.extend(
        [
            "",
            f"# RAT Experiments - {'Quick Test' if quick_test else 'Full Experiments'}",
            "# Auto-generated from configuration",
            "",
            "set -e",
            "",
            'echo "======================================================"',
            f'echo "RAT {"Quick Test" if quick_test else "Experiments"} - Node Setup and Training"',
            'echo "Job ID: $LSB_JOBID"',
            'echo "Host: $LSB_HOSTS"',
            'echo "Queue: $LSB_QUEUE"',
            'echo "======================================================"',
            "",
            "# Configuration from .config file",
            f"LOCAL_DATA_DIR=\"{config.get_local_data_dir('$LSB_JOBID')}\"",
            f"NETWORK_RESULTS_DIR=\"{paths_config['results_dir']}\"",
            f"NETWORK_CHECKPOINTS_DIR=\"{paths_config['checkpoints_dir']}\"",
            f"NUM_GPUS={lsf_config['num_gpus']}",
            "",
            "# Environment setup",
            'echo "Setting up environment..."',
            "source ~/.bashrc",
        ]
    )

    # Add conda environment activation if specified
    conda_env = config.get("environment", "conda_env")
    if conda_env:
        script_lines.append(f"conda activate {conda_env}")

    # Add module loading if specified
    modules = config.get("environment", "modules_to_load")
    if modules:
        for module in modules.split(","):
            module = module.strip()
            if module:
                script_lines.append(f"module load {module}")

    script_lines.extend(
        [
            "",
            'echo "LSF GPU allocation:"',
            'echo "LSB_HOSTS: $LSB_HOSTS"',
            'echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"',
            "",
            'echo "Python version: $(python --version)"',
            "echo \"PyTorch version: $(python -c 'import torch; print(torch.__version__)')\"",
            "echo \"CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')\"",
            "echo \"CUDA devices: $(python -c 'import torch; print(torch.cuda.device_count())')\"",
            "",
            "# Create necessary directories",
            'echo "Creating directories..."',
            'mkdir -p "$LOCAL_DATA_DIR"',
            'mkdir -p "$NETWORK_RESULTS_DIR/lsf_logs"',
            'mkdir -p "$NETWORK_RESULTS_DIR/tensorboard_logs"',
            'mkdir -p "$NETWORK_RESULTS_DIR/experiment_logs"',
            'mkdir -p "$NETWORK_CHECKPOINTS_DIR"',
            "",
        ]
    )

    # Data setup section
    script_lines.extend(
        [
            "# Setup data",
            'echo "Setting up data..."',
        ]
    )

    # ISIC dataset setup
    isic_source = dataset_config["isic_source_dir"]
    if isic_source and not dataset_config["use_sample_data"]:
        script_lines.extend(
            [
                f'if [ -d "{isic_source}" ]; then',
                '    echo "Copying ISIC2018 to local storage..."',
                f'    rsync -av "{isic_source}/" "$LOCAL_DATA_DIR/ISIC2018/"',
                "else",
            ]
        )

    script_lines.extend(
        [
            '    echo "Creating sample ISIC2018 dataset..."',
            "    python scripts/setup_isic.py \\",
            "        --sample_only \\",
            '        --output_dir "$LOCAL_DATA_DIR/sample_ISIC2018" \\',
            f'        --num_samples {dataset_config["isic_sample_size"]}',
        ]
    )

    if isic_source and not dataset_config["use_sample_data"]:
        script_lines.append("fi")

    script_lines.append("")

    # COCO dataset setup
    coco_source = dataset_config["coco_source_dir"]
    if coco_source and not dataset_config["use_sample_data"]:
        script_lines.extend(
            [
                f'if [ -d "{coco_source}" ]; then',
                '    echo "Copying COCO2017 to local storage..."',
                f'    rsync -av "{coco_source}/" "$LOCAL_DATA_DIR/COCO2017/"',
                "else",
            ]
        )

    script_lines.extend(
        [
            '    echo "Creating sample COCO2017 dataset..."',
            "    python scripts/setup_coco.py \\",
            "        --sample_only \\",
            '        --output_dir "$LOCAL_DATA_DIR/sample_COCO2017" \\',
            f'        --num_samples {dataset_config["coco_sample_size"]}',
        ]
    )

    if coco_source and not dataset_config["use_sample_data"]:
        script_lines.append("fi")

    script_lines.extend(
        [
            "",
            'echo "Local data setup completed. Directory size:"',
            'du -sh "$LOCAL_DATA_DIR"',
            "",
            "# Update configurations for cluster",
            'echo "Updating configurations for cluster..."',
            "python scripts/update_cluster_configs.py \\",
            '    --data_dir "$LOCAL_DATA_DIR" \\',
            '    --results_dir "$NETWORK_RESULTS_DIR" \\',
            '    --checkpoint_dir "$NETWORK_CHECKPOINTS_DIR" \\',
            '    --num_gpus "$NUM_GPUS"',
            "",
            "# Set distributed training environment variables",
            f"export MASTER_ADDR=$(hostname)",
            f"export MASTER_PORT={training_config['master_port_base']}",
            "export WORLD_SIZE=$NUM_GPUS",
            "",
            'echo "Starting distributed training..."',
            'echo "Master address: $MASTER_ADDR"',
            'echo "Master port: $MASTER_PORT"',
            'echo "World size: $WORLD_SIZE"',
            "",
        ]
    )

    # Training sections based on configuration
    port_base = training_config["master_port_base"]
    port_increment = training_config["port_increment"]
    current_port = port_base

    if experiments_config["medical_segmentation"]:
        script_lines.extend(
            [
                "# Run medical segmentation experiments",
                'echo "Running medical segmentation experiments..."',
                "torchrun \\",
                "    --nnodes=1 \\",
                "    --nproc_per_node=$NUM_GPUS \\",
                "    --master_addr=$MASTER_ADDR \\",
                f"    --master_port={current_port} \\",
                "    train_distributed.py \\",
                "    --config_dir medical_segmentation/configs \\",
                '    --results_dir "$NETWORK_RESULTS_DIR" \\',
                '    --checkpoint_dir "$NETWORK_CHECKPOINTS_DIR"',
                "",
            ]
        )
        current_port += port_increment

    if experiments_config["object_detection"]:
        script_lines.extend(
            [
                "# Run object detection experiments (if configs exist)",
                'if [ -d "object_detection/configs" ]; then',
                '    echo "Running object detection experiments..."',
                "    torchrun \\",
                "        --nnodes=1 \\",
                "        --nproc_per_node=$NUM_GPUS \\",
                "        --master_addr=$MASTER_ADDR \\",
                f"        --master_port={current_port} \\",
                "        train_distributed.py \\",
                "        --config_dir object_detection/configs \\",
                '        --results_dir "$NETWORK_RESULTS_DIR" \\',
                '        --checkpoint_dir "$NETWORK_CHECKPOINTS_DIR"',
                "fi",
                "",
            ]
        )
        current_port += port_increment

    if experiments_config["ablation_studies"]:
        script_lines.extend(
            [
                "# Run ablation studies",
                'if [ -d "ablations/configs" ]; then',
                '    echo "Running ablation studies..."',
                "    torchrun \\",
                "        --nnodes=1 \\",
                "        --nproc_per_node=$NUM_GPUS \\",
                "        --master_addr=$MASTER_ADDR \\",
                f"        --master_port={current_port} \\",
                "        train_distributed.py \\",
                "        --config_dir ablations/configs \\",
                '        --results_dir "$NETWORK_RESULTS_DIR" \\',
                '        --checkpoint_dir "$NETWORK_CHECKPOINTS_DIR"',
                "fi",
                "",
            ]
        )
        current_port += port_increment

    if experiments_config["robustness_tests"]:
        script_lines.extend(
            [
                "# Run robustness tests",
                'if [ -d "robustness" ]; then',
                '    echo "Running robustness tests..."',
                "    python robustness/resolution_transfer.py \\",
                '        --results_dir "$NETWORK_RESULTS_DIR" \\',
                '        --checkpoint_dir "$NETWORK_CHECKPOINTS_DIR"',
                "fi",
                "",
            ]
        )

    # Summary and cleanup
    script_lines.extend(
        [
            'echo "Training completed successfully!"',
            "",
            "# Save experiment summary",
            'echo "Saving experiment summary..."',
            'SUMMARY_FILE="$NETWORK_RESULTS_DIR/experiment_summary_$LSB_JOBID.txt"',
            'cat > "$SUMMARY_FILE" << EOF',
            f"RAT {'Quick Test' if quick_test else 'Experiments'} Summary",
            "======================",
            "Job ID: $LSB_JOBID",
            "Host: $LSB_HOSTS",
            "Queue: $LSB_QUEUE",
            "Start Time: $(date)",
            "GPUs Used: $NUM_GPUS",
            "Local Data Dir: $LOCAL_DATA_DIR",
            "Results Dir: $NETWORK_RESULTS_DIR",
            "Checkpoints Dir: $NETWORK_CHECKPOINTS_DIR",
            "",
            "Configuration:",
            f"- Use sample data: {dataset_config['use_sample_data']}",
            f"- Mixed precision: {training_config['mixed_precision']}",
            f"- Distributed backend: {training_config['distributed_backend']}",
            "",
            "Datasets:",
            '$(ls -la "$LOCAL_DATA_DIR")',
            "",
            "GPU Information:",
            "$(nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv)",
            "EOF",
            "",
            'echo "Experiment summary saved to: $SUMMARY_FILE"',
            "",
        ]
    )

    # Cleanup if using local storage
    if config.get("paths", "use_local_storage"):
        script_lines.extend(
            [
                "# Clean up local data to free space",
                'echo "Cleaning up local data..."',
                'rm -rf "$LOCAL_DATA_DIR"',
                "",
            ]
        )

    script_lines.extend(
        [
            'echo "======================================================"',
            f'echo "RAT {"Quick Test" if quick_test else "Experiments"} Completed Successfully!"',
            'echo "Results: $NETWORK_RESULTS_DIR"',
            'echo "Checkpoints: $NETWORK_CHECKPOINTS_DIR"',
            'echo "TensorBoard: tensorboard --logdir $NETWORK_RESULTS_DIR/tensorboard_logs"',
            'echo "======================================================"',
        ]
    )

    script_content = "\n".join(script_lines)

    if output_file:
        with open(output_file, "w") as f:
            f.write(script_content)
        print(f"LSF script generated: {output_file}")
    else:
        print(script_content)

    return script_content


def main():
    parser = argparse.ArgumentParser(
        description="Generate LSF job scripts from configuration"
    )
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument(
        "--quick", action="store_true", help="Generate quick test script"
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Generate parallel job scripts"
    )
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")

    args = parser.parse_args()

    try:
        config = load_config(args.config)
        experiments_config = config.get_experiments_config()

        # Check if parallel jobs are requested or configured
        use_parallel = args.parallel or experiments_config.get("parallel_jobs", False)

        if use_parallel and not args.quick:
            # Generate parallel job scripts
            print("Generating parallel job scripts...")
            generated_scripts = generate_parallel_job_scripts(config)
            print(f"Generated {len(generated_scripts)} parallel job scripts")
            return

        # Generate single script (original behavior)

        # Determine output file if not specified
        if not args.output:
            if args.quick:
                args.output = "cluster/submit_quick_test.lsf"
            else:
                args.output = "cluster/submit_experiments.lsf"

        # Check if file exists
        if Path(args.output).exists() and not args.force:
            response = input(f"File {args.output} exists. Overwrite? (y/N): ")
            if response.lower() != "y":
                print("Cancelled.")
                return

        # Create cluster directory if it doesn't exist
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)

        # Generate script
        generate_lsf_script(config, quick_test=args.quick, output_file=args.output)

        # Make script executable
        Path(args.output).chmod(0o755)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
