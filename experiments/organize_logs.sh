#!/bin/bash
# Log File Organization Script
# Moves misplaced log files to their proper directories

EXPERIMENTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$EXPERIMENTS_DIR/results"
EXPERIMENT_LOGS_DIR="$RESULTS_DIR/experiment_logs"

echo "🧹 Organizing log files in experiments directory..."

# Create directories if they don't exist
mkdir -p "$EXPERIMENT_LOGS_DIR"
mkdir -p "$RESULTS_DIR/lsf_logs"
mkdir -p "$RESULTS_DIR/tensorboard_logs"

# Function to move log files
move_logs() {
    local source_dir="$1"
    local dest_dir="$2"
    local pattern="$3"
    local description="$4"
    
    if find "$source_dir" -maxdepth 1 -name "$pattern" -type f | grep -q .; then
        echo "  Moving $description from $source_dir to $dest_dir"
        local before_count=$(find "$dest_dir" -maxdepth 1 -name "$pattern" -type f | wc -l)
        local move_count=$(find "$source_dir" -maxdepth 1 -name "$pattern" -type f | wc -l)
        find "$source_dir" -maxdepth 1 -name "$pattern" -type f -exec mv {} "$dest_dir/" \;
        local after_count=$(find "$dest_dir" -maxdepth 1 -name "$pattern" -type f | wc -l)
        local actual_moved=$((after_count - before_count))
        echo "    ✓ Moved $actual_moved files"
    fi
}

# Move application log files scattered in various directories
move_logs "$EXPERIMENTS_DIR/checkpoints" "$EXPERIMENT_LOGS_DIR" "*.log" "training logs"
move_logs "$EXPERIMENTS_DIR" "$EXPERIMENT_LOGS_DIR" "*_train_*.log" "training logs"
move_logs "$EXPERIMENTS_DIR" "$EXPERIMENT_LOGS_DIR" "*_study_*.log" "study logs"

# Move any LSF output files to proper location
move_logs "$EXPERIMENTS_DIR" "$RESULTS_DIR/lsf_logs" "*.out" "LSF output files"
move_logs "$EXPERIMENTS_DIR" "$RESULTS_DIR/lsf_logs" "*.err" "LSF error files"

# Report final status
echo ""
echo "📊 Log file organization complete!"
echo "   Experiment logs: $EXPERIMENT_LOGS_DIR ($(find "$EXPERIMENT_LOGS_DIR" -name "*.log" | wc -l) files)"
echo "   LSF logs: $RESULTS_DIR/lsf_logs ($(find "$RESULTS_DIR/lsf_logs" -name "*.out" -o -name "*.err" | wc -l) files)"
echo "   TensorBoard logs: $RESULTS_DIR/tensorboard_logs"
echo ""
echo "✨ All logs are now properly organized!"