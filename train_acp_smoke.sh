#!/bin/bash
# =============================================================================
# ACP Small-Scale Training Test Script
# =============================================================================
# This script trains the model with a small subset of data for quick testing.
#
# Usage:
#   ./train_acp_smoke.sh        # Run smoke test (100 samples, 10 steps)
#   ./train_acp_smoke.sh quick  # Quick test (500 samples, 50 steps)
#   ./train_acp_smoke.sh full   # Full training (all data, 100000 steps)
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_DIR="$REPO_ROOT/Ancient-Chinese-Phonology"
cd "$TRAIN_DIR"

# Default: smoke test
MODE="${1:-smoke}"

# GPU settings
GPU_ID=0

# =============================================================================
# Functions
# =============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        log "ERROR: nvidia-smi not found. Is NVIDIA driver installed?"
        exit 1
    fi
    
    local gpu_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader -i $GPU_ID | awk '{print $1}')
    log "GPU $GPU_ID Memory: ${gpu_mem}MB"
}

# Create small subset of training data
create_subset() {
    local subset_size=$1
    local train_file="corpus/train.txt"
    local dev_file="corpus/dev.txt"
    local train_subset="corpus/train_small_${subset_size}.txt"
    local dev_subset="corpus/dev_small_${subset_size}.txt"
    
    # Create training subset (first N lines)
    head -n $subset_size "$train_file" > "$train_subset"
    
    # Create dev subset (proportional: ~10% of train size)
    local dev_size=$((subset_size / 10))
    head -n $dev_size "$dev_file" > "$dev_subset"
    
    echo "$train_subset $dev_subset"
}

# Run training
run_training() {
    local train_file=$1
    local dev_file=$2
    local total_steps=$3
    local batch_size=$4
    local seed=$5
    local output_dir=$6
    
    log "=========================================="
    log "Training config:"
    log "  Train file: $train_file"
    log "  Dev file: $dev_file"
    log "  Total steps: $total_steps"
    log "  Batch size: $batch_size"
    log "  Seed: $seed"
    log "  Output: $output_dir"
    log "=========================================="
    
    mkdir -p "$output_dir/model"
    
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    
    python -u train/train.py \
        --output_dir "$output_dir" \
        --train_dir "$train_file" \
        --dev_dir "$dev_file" \
        --char_dict data/han_seq.json \
        --name "smoke_${seed}" \
        --seed $seed \
        --total_train_steps $total_steps \
        --num_train_epochs 2 \
        --learning_rate 0.0001 \
        --batch_size $batch_size \
        --gradient_accumulation_steps 1 \
        --logging_steps 10 \
        --saving_steps $total_steps \
        2>&1 | tee "$output_dir/training.log"
    
    log "Training complete!"
}

# =============================================================================
# Main
# =============================================================================

case "$MODE" in
    smoke)
        log "=== SMOKE TEST MODE ==="
        log "Using 100 samples, 10 steps for quick validation"
        
        check_gpu
        
        # Create small subset
        train_subset="corpus/train_small_100.txt"
        dev_subset="corpus/dev_small_100.txt"
        
        head -n 100 corpus/train.txt > "$train_subset"
        head -n 10 corpus/dev.txt > "$dev_subset"
        
        log "Created subset: train=$train_subset (100), dev=$dev_subset (10)"
        
        # Count lines
        log "Train samples: $(wc -l < $train_subset)"
        log "Dev samples: $(wc -l < $dev_subset)"
        
        run_training \
            "$train_subset" \
            "$dev_subset" \
            10 \
            32 \
            42 \
            "repro/smoke_test"
        ;;
        
    quick)
        log "=== QUICK TEST MODE ==="
        log "Using 500 samples, 50 steps"
        
        check_gpu
        
        train_subset="corpus/train_small_500.txt"
        dev_subset="corpus/dev_small_500.txt"
        
        head -n 500 corpus/train.txt > "$train_subset"
        head -n 50 corpus/dev.txt > "$dev_subset"
        
        log "Created subset: train=$train_subset (500), dev=$dev_subset (50)"
        
        run_training \
            "$train_subset" \
            "$dev_subset" \
            50 \
            32 \
            42 \
            "repro/quick_test"
        ;;
        
    full)
        log "=== FULL TRAINING MODE ==="
        
        check_gpu
        
        # Use full corpus
        TOTAL_TRAIN_STEPS=100000
        BATCH_SIZE=128
        GRAD_ACCUM_STEPS=8
        SAVING_STEPS=10000
        SEEDS=(42 43 44 45 46)
        
        log "Training with full dataset:"
        log "  Total steps: $TOTAL_TRAIN_STEPS"
        log "  Batch size: $BATCH_SIZE (effective: $((BATCH_SIZE * GRAD_ACCUM_STEPS)))"
        
        for seed in "${SEEDS[@]}"; do
            run_training \
                "corpus/train.txt" \
                "corpus/dev.txt" \
                $TOTAL_TRAIN_STEPS \
                $BATCH_SIZE \
                $seed \
                "repro/seed_runs_native/$seed"
        done
        
        log "All seeds training complete!"
        ;;
        
    help|--help|-h)
        echo "Usage: $0 [mode]"
        echo ""
        echo "Modes:"
        echo "  smoke   - Smoke test (100 samples, 10 steps) - for quick validation"
        echo "  quick   - Quick test (500 samples, 50 steps)"
        echo "  full    - Full training (15333 samples, 100000 steps)"
        echo "  help    - Show this help"
        ;;
        
    *)
        echo "Unknown mode: $MODE"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac

log "Done!"
