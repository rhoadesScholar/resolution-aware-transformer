#!/bin/bash
# LSF Deployment Helper for RAT Experiments
# Usage: ./deploy_lsf.sh [quick|full|status|generate|config]
# Ensure the script runs from its own directory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$SCRIPT_DIR"
set -e

# Configuration
CLUSTER_SCRIPTS_DIR="cluster"
CONFIG_FILE=".config"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}======================================================"
    echo -e "RAT LSF Deployment Helper"
    echo -e "======================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

check_lsf() {
    if ! command -v bsub &> /dev/null; then
        print_error "LSF not found. Please ensure LSF is installed and configured."
        exit 1
    fi
    print_success "LSF is available"
}

check_directories() {
    # Use config to get and create directories
    python3 config_manager.py --create-dirs
    print_success "Directories created/verified from configuration"
}

show_config() {
    print_header
    echo -e "${BLUE}Current Configuration:${NC}"
    
    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "Configuration file not found: $CONFIG_FILE"
        return 1
    fi
    
    python3 config_manager.py --dump
    
    echo ""
    echo -e "${BLUE}LSF Job Configuration (Full):${NC}"
    python3 config_manager.py --lsf-config
    
    echo ""
    echo -e "${BLUE}LSF Job Configuration (Quick):${NC}"
    python3 config_manager.py --lsf-config --quick
}

generate_scripts() {
    print_header
    echo -e "${YELLOW}Generating LSF scripts from configuration...${NC}"
    
    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "Configuration file not found: $CONFIG_FILE"
        print_error "Please create $CONFIG_FILE first"
        return 1
    fi
    
    # Check if parallel jobs are configured
    PARALLEL_JOBS=$(python3 config_manager.py --dump | grep "parallel_jobs" | cut -d'=' -f2 | xargs)
    
    if [ "$PARALLEL_JOBS" = "true" ]; then
        echo "Generating parallel job scripts..."
        python3 generate_lsf_script.py --parallel --force
        
        echo ""
        echo -e "${BLUE}Generated parallel job files:${NC}"
        ls -la $CLUSTER_SCRIPTS_DIR/submit_*.lsf 2>/dev/null | grep -v "submit_experiments.lsf\|submit_quick_test.lsf" || echo "No parallel scripts generated"
        echo "- $CLUSTER_SCRIPTS_DIR/submit_all_parallel.lsf (master submission script)"
    else
        # Generate both standard scripts
        echo "Generating sequential job scripts..."
        python3 generate_lsf_script.py --force
        
        echo "Generating quick test script..."
        python3 generate_lsf_script.py --quick --force
        
        echo ""
        echo -e "${BLUE}Generated sequential job files:${NC}"
        echo "- $CLUSTER_SCRIPTS_DIR/submit_experiments.lsf"
        echo "- $CLUSTER_SCRIPTS_DIR/submit_quick_test.lsf"
    fi
    
    print_success "LSF scripts generated successfully"
}

show_status() {
    print_header
    echo -e "${BLUE}Current LSF Job Status:${NC}"
    
    # Show all RAT jobs
    echo "All RAT experiments:"
    bjobs -w | grep -E "(rat_|RAT)" || echo "No RAT jobs found"
    
    echo ""
    echo -e "${BLUE}GPU Queue Status:${NC}"
    bqueues gpu || echo "GPU queue not available"
    
    echo ""
    echo -e "${BLUE}Recent Results:${NC}"
    # Get results directory from config
    RESULTS_DIR=$(python3 config_manager.py --dump | grep "results_dir" | cut -d'=' -f2 | xargs)
    if [ -d "$RESULTS_DIR" ]; then
        ls -la "$RESULTS_DIR"
    else
        echo "Results directory not found: $RESULTS_DIR"
    fi
}

submit_quick_test() {
    print_header
    echo -e "${YELLOW}Submitting quick test (2 GPUs, 1 hour)...${NC}"
    
    check_lsf
    check_directories
    
    if [ ! -f "$CLUSTER_SCRIPTS_DIR/submit_quick_test.lsf" ]; then
        print_error "Quick test script not found: $CLUSTER_SCRIPTS_DIR/submit_quick_test.lsf"
        exit 1
    fi
    
    # Submit job
    JOB_ID=$(bsub < "$CLUSTER_SCRIPTS_DIR/submit_quick_test.lsf" | grep -oE '[0-9]+')
    
    if [ -n "$JOB_ID" ]; then
        print_success "Quick test submitted with Job ID: $JOB_ID"
        
        echo ""
        echo -e "${BLUE}Monitor progress:${NC}"
        echo "bjobs $JOB_ID"
        
        # Get results directory from config
        RESULTS_DIR=$(python3 config_manager.py --dump | grep "results_dir" | cut -d'=' -f2 | xargs)
        echo "tail -f $RESULTS_DIR/lsf_logs/rat_quick_${JOB_ID}.out"
        
        echo ""
        echo -e "${BLUE}View results when complete:${NC}"
        echo "cat $RESULTS_DIR/quick_test_summary_${JOB_ID}.txt"
        
    else
        print_error "Failed to submit quick test"
        exit 1
    fi
}

submit_full_experiments() {
    print_header
    echo -e "${YELLOW}Submitting full experiments...${NC}"
    
    check_lsf
    check_directories
    
    # Check if parallel jobs are configured
    PARALLEL_JOBS=$(python3 config_manager.py --dump | grep "parallel_jobs" | cut -d'=' -f2 | xargs)
    
    if [ "$PARALLEL_JOBS" = "true" ]; then
        if [ ! -f "$CLUSTER_SCRIPTS_DIR/submit_all_parallel.lsf" ]; then
            print_error "Parallel submission script not found. Run './deploy_lsf.sh generate' first."
            exit 1
        fi
        
        echo -e "${YELLOW}This will submit multiple parallel jobs for different experiment types.${NC}"
        read -p "Are you sure you want to proceed? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Cancelled."
            exit 0
        fi
        
        # Submit parallel jobs
        echo "Submitting parallel experiments..."
        bash $CLUSTER_SCRIPTS_DIR/submit_all_parallel.lsf
        
        print_success "Parallel experiments submitted successfully"
        
        echo ""
        echo -e "${BLUE}Monitor progress:${NC}"
        echo "bjobs"
        echo ""
        echo -e "${BLUE}View logs:${NC}"
        RESULTS_DIR=$(python3 config_manager.py --dump | grep "results_dir" | cut -d'=' -f2 | xargs)
        echo "ls $RESULTS_DIR/lsf_logs/"
        
    else
        if [ ! -f "$CLUSTER_SCRIPTS_DIR/submit_experiments.lsf" ]; then
            print_error "Full experiment script not found: $CLUSTER_SCRIPTS_DIR/submit_experiments.lsf"
            exit 1
        fi
        
        # Ask for confirmation
        echo -e "${YELLOW}This will submit a long-running job using 8 GPUs.${NC}"
        read -p "Are you sure you want to proceed? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Cancelled."
            exit 0
        fi
        
        # Submit job
        JOB_ID=$(bsub < "$CLUSTER_SCRIPTS_DIR/submit_experiments.lsf" | grep -oE '[0-9]+')
        
        if [ -n "$JOB_ID" ]; then
            print_success "Full experiments submitted with Job ID: $JOB_ID"
            
            echo ""
            echo -e "${BLUE}Monitor progress:${NC}"
            echo "bjobs $JOB_ID"
            
            # Get results directory from config
            RESULTS_DIR=$(python3 config_manager.py --dump | grep "results_dir" | cut -d'=' -f2 | xargs)
            echo "tail -f $RESULTS_DIR/lsf_logs/rat_experiments_${JOB_ID}.out"
            
            echo ""
            echo -e "${BLUE}View TensorBoard:${NC}"
            echo "tensorboard --logdir $RESULTS_DIR/tensorboard_logs"
            
        else
            print_error "Failed to submit full experiments"
            exit 1
        fi
    fi
}

show_help() {
    print_header
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  config   - Show current configuration"
    echo "  generate - Generate LSF scripts from configuration"
    echo "  quick    - Submit quick test (2 GPUs, no time limit)"
    echo "  full     - Submit full experiments (8 GPUs, no time limit)"
    echo "  status   - Show current job status and results"
    echo "  help     - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 config       # Show current configuration"
    echo "  $0 generate     # Generate LSF scripts"
    echo "  $0 quick        # Run quick validation"
    echo "  $0 status       # Check job status"
    echo "  $0 full         # Run full experiments"
    echo ""
    echo -e "${YELLOW}Configuration:${NC}"
    echo "Edit experiments/.config to customize:"
    echo "- LSF queue and resource allocation"
    echo "- Data and results directories"
    echo "- Experiment selection and parallel execution"
    echo "- Training parameters"
    echo ""
    echo -e "${YELLOW}Key Configuration Options:${NC}"
    echo "- data_dir: For local development (default: repo/data)"
    echo "- local_temp_dir: Where data is processed on cluster nodes (default: /tmp/rat_data)"
    echo "- parallel_jobs: true = separate job per experiment, false = sequential"
    echo "- walltime limits: Removed - jobs run without time constraints"
    echo ""
    echo -e "${YELLOW}First time setup:${NC}"
    echo "1. Edit 'experiments/.config' for your cluster"
    echo "2. Run './deploy_lsf.sh generate' to create scripts"
    echo "3. Run './deploy_lsf.sh quick' to validate setup"
    echo "4. Check results and logs"
    echo "5. Run './deploy_lsf.sh full' for complete experiments"
}

# Main script logic
case "${1:-help}" in
    "config")
        show_config
        ;;
    "generate")
        generate_scripts
        ;;
    "quick")
        submit_quick_test
        ;;
    "full")
        submit_full_experiments
        ;;
    "status")
        show_status
        ;;
    "help"|*)
        show_help
        ;;
esac