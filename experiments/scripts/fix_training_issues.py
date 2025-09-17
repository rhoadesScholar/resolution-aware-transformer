#!/usr/bin/env python3
"""
Script to fix common training issues in the Resolution Aware Transformer experiments.

This script identifies and fixes common issues that can cause training failures:
1. COCO annotation file path issues
2. Logger initialization issues in distributed training
3. Import path issues
4. Configuration validation
"""

import os
import re
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any


def fix_coco_annotation_paths(datasets_file: Path) -> bool:
    """Fix the COCO annotation file path construction."""
    if not datasets_file.exists():
        print(f"Warning: {datasets_file} not found")
        return False

    content = datasets_file.read_text()

    # Fix the double "2017" issue in annotation filename
    old_pattern = r'f"annotations/instances_{self\.split}2017\.json"'
    new_pattern = r'f"annotations/instances_{self.split}.json"'

    if old_pattern.replace("\\", "") in content:
        content = re.sub(old_pattern, new_pattern, content)
        datasets_file.write_text(content)
        print(f"✓ Fixed COCO annotation path in {datasets_file}")
        return True
    else:
        print(f"✓ COCO annotation path already correct in {datasets_file}")
        return False


def fix_ablation_logger_issues(ablation_file: Path) -> bool:
    """Fix logger usage in ablation study for distributed training."""
    if not ablation_file.exists():
        print(f"Warning: {ablation_file} not found")
        return False

    content = ablation_file.read_text()

    # Pattern to find logger calls without None checks
    patterns_to_fix = [
        (r"(\s+)logger\.info\(", r"\1if logger:\n\1    logger.info("),
        (r"(\s+)logger\.warning\(", r"\1if logger:\n\1    logger.warning("),
        (r"(\s+)logger\.error\(", r"\1if logger:\n\1    logger.error("),
        (r"(\s+)logger\.debug\(", r"\1if logger:\n\1    logger.debug("),
    ]

    changes_made = False
    for pattern, replacement in patterns_to_fix:
        # Only replace if not already wrapped in if logger check
        matches = re.finditer(pattern, content)
        for match in matches:
            start_pos = max(0, match.start() - 50)
            context = content[start_pos : match.start()]

            # Skip if already has 'if logger:' check nearby
            if "if logger:" not in context:
                content = re.sub(pattern, replacement, content, count=1)
                changes_made = True

    if changes_made:
        ablation_file.write_text(content)
        print(f"✓ Fixed logger usage in {ablation_file}")
    else:
        print(f"✓ Logger usage already correct in {ablation_file}")

    return changes_made


def validate_config_files(config_dir: Path) -> List[str]:
    """Validate configuration files for common issues."""
    issues = []

    if not config_dir.exists():
        return [f"Config directory {config_dir} not found"]

    yaml_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))

    for config_file in yaml_files:
        try:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            # Check for required sections
            required_sections = ["model", "training", "data"]
            for section in required_sections:
                if section not in config:
                    issues.append(
                        f"{config_file.name}: Missing required section '{section}'"
                    )

            # Check for data paths
            if "data" in config:
                data_dir = config["data"].get("data_dir")
                if data_dir and not Path(data_dir).exists():
                    # Only warn if it's not a template path
                    if "/path/to" not in data_dir and "/tmp/" not in data_dir:
                        issues.append(
                            f"{config_file.name}: Data directory not found: {data_dir}"
                        )

            # Check for realistic batch sizes
            if "training" in config:
                batch_size = config["training"].get("batch_size", 1)
                if batch_size > 64:
                    issues.append(
                        f"{config_file.name}: Large batch size ({batch_size}) may cause OOM"
                    )

            # Check model parameters for memory feasibility
            if "model" in config:
                feature_dims = config["model"].get("feature_dims", 128)
                num_blocks = config["model"].get("num_blocks", 4)
                if feature_dims > 512 or num_blocks > 8:
                    issues.append(
                        f"{config_file.name}: Large model (dims={feature_dims}, blocks={num_blocks}) may need DeepSpeed"
                    )

        except yaml.YAMLError as e:
            issues.append(f"{config_file.name}: YAML parsing error: {e}")
        except Exception as e:
            issues.append(f"{config_file.name}: Validation error: {e}")

    return issues


def check_import_paths(experiments_dir: Path) -> List[str]:
    """Check for potential import path issues."""
    issues = []

    # Check if common modules exist
    common_dir = experiments_dir / "common"
    if not common_dir.exists():
        issues.append("Common directory not found")
        return issues

    required_files = ["datasets.py", "models.py", "metrics.py", "utils.py"]
    for req_file in required_files:
        if not (common_dir / req_file).exists():
            issues.append(f"Required common file missing: {req_file}")

    # Check for __init__.py files for proper Python module structure
    for subdir in ["common", "medical_segmentation", "object_detection", "ablations"]:
        init_file = experiments_dir / subdir / "__init__.py"
        if not init_file.exists():
            issues.append(f"Missing __init__.py in {subdir}")

    return issues


def fix_distributed_training_issues(train_files: List[Path]) -> int:
    """Fix common distributed training issues."""
    fixes_applied = 0

    for train_file in train_files:
        if not train_file.exists():
            continue

        content = train_file.read_text()
        original_content = content

        # Fix: Add proper error handling for distributed initialization
        if "dist.init_process_group" in content and "try:" not in content:
            init_pattern = r"(\s+)(dist\.init_process_group\([^)]+\))"
            replacement = r'\1try:\n\1    \2\n\1except Exception as e:\n\1    print(f"Failed to initialize distributed training: {e}")\n\1    raise'
            content = re.sub(init_pattern, replacement, content)

        # Fix: Ensure proper rank checking for logging setup
        if "rank == 0" in content and "logger = None" not in content:
            # Add logger = None for non-rank 0 processes if not present
            if "else:" in content and "logger = None" not in content:
                content = content.replace(
                    "else:\n        tracker = None",
                    "else:\n        tracker = None\n        logger = None",
                )

        if content != original_content:
            train_file.write_text(content)
            print(f"✓ Fixed distributed training issues in {train_file}")
            fixes_applied += 1

    return fixes_applied


def main():
    """Main function to run all fixes."""
    experiments_dir = Path(__file__).parent.parent
    print(f"Checking experiments directory: {experiments_dir}")

    total_fixes = 0

    # Fix 1: COCO annotation paths
    datasets_file = experiments_dir / "common" / "datasets.py"
    if fix_coco_annotation_paths(datasets_file):
        total_fixes += 1

    # Fix 2: Ablation logger issues
    ablation_file = experiments_dir / "ablations" / "ablation_study.py"
    if fix_ablation_logger_issues(ablation_file):
        total_fixes += 1

    # Fix 3: Distributed training issues
    train_files = [
        experiments_dir / "medical_segmentation" / "train.py",
        experiments_dir / "object_detection" / "train.py",
    ]
    fixes_applied = fix_distributed_training_issues(train_files)
    total_fixes += fixes_applied

    # Check 4: Config file validation
    config_dirs = [
        experiments_dir / "medical_segmentation" / "configs",
        experiments_dir / "object_detection" / "configs",
        experiments_dir / "ablations" / "configs",
    ]

    all_config_issues = []
    for config_dir in config_dirs:
        issues = validate_config_files(config_dir)
        all_config_issues.extend(issues)

    if all_config_issues:
        print("\n⚠️ Configuration Issues Found:")
        for issue in all_config_issues:
            print(f"  - {issue}")
    else:
        print("✓ All configuration files validated successfully")

    # Check 5: Import paths
    import_issues = check_import_paths(experiments_dir)
    if import_issues:
        print("\n⚠️ Import Path Issues Found:")
        for issue in import_issues:
            print(f"  - {issue}")
    else:
        print("✓ All import paths validated successfully")

    print(f"\n✅ Applied {total_fixes} fixes total")

    if total_fixes > 0:
        print("\nRecommendations:")
        print("1. Re-run the failed experiments to verify fixes")
        print("2. Check LSF logs for any remaining issues")
        print("3. Ensure dataset paths are correctly configured")
        print("4. Verify DeepSpeed is properly configured for large models")


if __name__ == "__main__":
    main()
