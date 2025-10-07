#!/bin/bash

# Get the directory where the script is located
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$SCRIPTPATH"

# Script paths
FINETUNE_SCRIPT="${SCRIPTPATH}/finetune.py"
PRUNING_SCRIPT="${SCRIPTPATH}/prune.py"
METRICS_SCRIPT="${SCRIPTPATH}/metrics.py"
VENV_PATH="${SCRIPTPATH}/ultravenv"
DATA_PATH="${SCRIPTPATH}/BCCD/data.yaml"

# Logging setup
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/yolo11n_experiment_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

# Print session information
echo "========================================" | tee "$LOG_FILE"
echo "YOLO11n Pruning Experiment Session" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Create and activate virtual environment
echo "Setting up virtual environment..." | tee -a "$LOG_FILE"
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating new virtual environment..." | tee -a "$LOG_FILE"
    python -m venv "$VENV_PATH"
fi

# Activate virtual environment (Windows-specific)
if [ -f "$VENV_PATH/Scripts/activate" ]; then
    source "$VENV_PATH/Scripts/activate"
    
    # Install required packages if needed
    pip install ultralytics torch torch-pruning | tee -a "$LOG_FILE"
else
    echo "Error: Virtual environment activation failed" | tee -a "$LOG_FILE"
    exit 1
fi

# First run finetuning
echo -e "\n=== Starting Finetuning ===" | tee -a "$LOG_FILE"
python "$FINETUNE_SCRIPT" \
            --model "yolo11n.pt" \
            --data "$DATA_PATH" \
            --epochs 100 \
            --batch_size 4 2>&1 | tee -a "$LOG_FILE"
FINETUNE_STATUS=$?

if [ $FINETUNE_STATUS -eq 0 ]; then
    # Check if model saved correctly after finetuning
    FINETUNED_MODEL="${SCRIPTPATH}/runs/fine_tuned_yolo11n/weights/best.pt"
    echo -e " Weights file after finetuning found at $FINETUNED_MODEL" | tee -a "$LOG_FILE"
    if [ -f "$FINETUNED_MODEL" ]; then
        echo -e "\n=== Starting Pruning ===" | tee -a "$LOG_FILE"
        python "$PRUNING_SCRIPT" \
            --model "$FINETUNED_MODEL" \
            --data "$DATA_PATH" \
            --cfg "default.yaml" \
            --iterative_steps 2 \
            --postprune_epochs 50 \
            --target_prune_rate 0.3 2>&1 | tee -a "$LOG_FILE"
        PRUNING_STATUS=$?
    else
        echo "Error: Weights file not found at $FINETUNED_MODEL" | tee -a "$LOG_FILE"
        exit 1
    fi
else
    echo "Finetuning failed with status: $FINETUNE_STATUS" | tee -a "$LOG_FILE"
    exit 1
fi

# Summary
echo -e "\n========================================" | tee -a "$LOG_FILE"
echo "Experiment Summary:" | tee -a "$LOG_FILE"
echo "Completed: $(date)" | tee -a "$LOG_FILE"
echo "Finetuning status: $FINETUNE_STATUS" | tee -a "$LOG_FILE"
[ $FINETUNE_STATUS -eq 0 ] && echo "Pruning status: $PRUNING_STATUS" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Total log size: $(du -h "$LOG_FILE" | cut -f1)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Deactivate virtual environment
deactivate