#!/bin/bash
# =============================================================================
# ACP (Ancient Chinese Phonology) Full Training Script
# =============================================================================
# This script trains the PGDN model for Ancient Chinese Phonology research.
# 
# Usage:
#   ./train_acp_full.sh          # Train all seeds (42, 43, 44, 45, 46)
#   ./train_acp_full.sh 42      # Train single seed
#   ./train_acp_full.sh dry    # Dry run (1 step only for testing)
#
# Requirements:
#   - GPU with >= 16GB VRAM (recommended 24GB)
#   - Python 3.10+
#   - PyTorch with CUDA
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_DIR="$REPO_ROOT/Ancient-Chinese-Phonology"
cd "$TRAIN_DIR"

# Training parameters
TOTAL_TRAIN_STEPS=100000      # Total training steps (default: 100000)
BATCH_SIZE=128                # Batch size per GPU
GRAD_ACCUM_STEPS=8           # Gradient accumulation (effective batch: 128 * 8 = 1024)
LEARNING_RATE=0.0001         # Adam learning rate
NUM_EPOCHS=2                # Number of epochs
LOGGING_STEPS=100            # Log every N steps
SAVING_STEPS=10000          # Save checkpoint every N steps

# Seeds to train
SEEDS=(42 43 44 45 46)

# GPU settings
GPU_ID=0  # Set to GPU device ID

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
    
    if [ "$gpu_mem" -lt 15000 ]; then
        log "WARNING: Less than 16GB VRAM. Consider reducing BATCH_SIZE."
    fi
}

train_single_seed() {
    local seed=$1
    local output_dir="repro/seed_runs_native/$seed"
    
    log "=========================================="
    log "Training seed $seed"
    log "Output directory: $output_dir"
    log "=========================================="
    
    # Create output directory
    mkdir -p "$output_dir/model"
    
    # Set CUDA device
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    
    # Run training
    python -u train/train.py \
        --output_dir "$output_dir" \
        --train_dir corpus/train.txt \
        --dev_dir corpus/dev.txt \
        --char_dict data/han_seq.json \
        --name pretrain_native \
        --seed $seed \
        --total_train_steps $TOTAL_TRAIN_STEPS \
        --num_train_epochs $NUM_EPOCHS \
        --learning_rate $LEARNING_RATE \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
        --logging_steps $LOGGING_STEPS \
        --saving_steps $SAVING_STEPS \
        2>&1 | tee "$output_dir/training.log"
    
    log "Seed $seed training complete!"
}

train_all() {
    check_gpu
    
    log "Starting full training for seeds: ${SEEDS[*]}"
    log "Parameters:"
    log "  Total steps: $TOTAL_TRAIN_STEPS"
    log "  Batch size: $BATCH_SIZE (effective: $((BATCH_SIZE * GRAD_ACCUM_STEPS)))"
    log "  Learning rate: $LEARNING_RATE"
    log "  Checkpoint interval: $SAVING_STEPS steps"
    
    for seed in "${SEEDS[@]}"; do
        train_single_seed $seed
    done
    
    log "=========================================="
    log "ALL TRAINING COMPLETE!"
    log "=========================================="
}

dry_run() {
    log "DRY RUN MODE - Training for 1 step only"
    
    for seed in 42; do
        local output_dir="repro/seed_runs_native/$seed"
        mkdir -p "$output_dir/model"
        
        export CUDA_VISIBLE_DEVICES=$GPU_ID
        
        python -u train/train.py \
            --output_dir "$output_dir" \
            --train_dir corpus/train.txt \
            --dev_dir corpus/dev.txt \
            --char_dict data/han_seq.json \
            --name pretrain_native \
            --seed $seed \
            --total_train_steps 1 \
            --num_train_epochs 2 \
            --learning_rate $LEARNING_RATE \
            --batch_size $BATCH_SIZE \
            --gradient_accumulation_steps 1 \
            --logging_steps 1 \
            --saving_steps 1 \
            2>&1 | tee "$output_dir/dryrun.log"
    done
    
    log "Dry run complete!"
}

# =============================================================================
# Main
# =============================================================================

case "${1:-all}" in
    dry)
        log "Running dry run test..."
        dry_run
        ;;
    42|43|44|45|46)
        log "Training single seed: $1"
        check_gpu
        train_single_seed $1
        ;;
    all)
        log "Training all seeds..."
        train_all
        ;;
    help|--help|-h)
        echo "Usage: $0 [mode]"
        echo ""
        echo "Modes:"
        echo "  (none)   - Train all seeds (42-46)"
        echo "  dry      - Dry run (1 step for testing)"
        echo "  42-46    - Train specific seed"
        echo "  help     - Show this help"
        ;;
    *)
        echo "Unknown mode: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
