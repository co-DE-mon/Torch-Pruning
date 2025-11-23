#!/bin/bash

###############################################
# YOLO11n FULL PRUNING EXPERIMENT
# For each batch size:
#   1. Train Model A for 25 epochs (NO pruning)
#   2. Save metrics
#   3. For each prune ratio:
#         a. Prune Model A
#         b. Finetune pruned model for 25 epochs
#         c. Save metrics
###############################################

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$SCRIPTPATH"

# Paths
FINETUNE_SCRIPT="${SCRIPTPATH}/finetune.py"
PRUNING_SCRIPT="${SCRIPTPATH}/prune.py"
VENV_PATH="${SCRIPTPATH}/ultravenv"
DATA_PATH="${SCRIPTPATH}/27073186/TXL-PBC/TXL-PBC/data.yaml"
BASE_MODEL="yolo11n.pt"

# Logging
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="${LOG_DIR}/main_experiment_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

echo "===============================================" | tee "$MASTER_LOG"
echo "     YOLO11n PRUNING EXPERIMENT" | tee -a "$MASTER_LOG"
echo "===============================================" | tee -a "$MASTER_LOG"
echo "Started: $(date)" | tee -a "$MASTER_LOG"

###############################################
# Setup Virtual Environment
###############################################
echo -e "\n[INFO] Setting up ultravenv..." | tee -a "$MASTER_LOG"

if [ ! -d "$VENV_PATH" ]; then
    python -m venv "$VENV_PATH"
fi

source "$VENV_PATH/Scripts/activate"
pip install ultralytics torch torch-pruning | tee -a "$MASTER_LOG"


###############################################
# EXPERIMENT PARAMETER
###############################################
EPOCHS_PRE=1
EPOCHS_POST=1
batch_sizes=(2 4 8 16)
PRUNE_LIST=(5 10 20 30 40 50 75)


###############################################
# MAIN LOOP — iterate over BATCH SIZE
###############################################
for BATCH in "${batch_sizes[@]}"; do

    echo -e "\n===================================================" | tee -a "$MASTER_LOG"
    echo "[GROUP] Starting experiments for BATCH SIZE = ${BATCH}" | tee -a "$MASTER_LOG"
    echo "===================================================" | tee -a "$MASTER_LOG"

    #############################
    # 1) TRAIN MODEL A (NO PRUNING)
    #############################
    RUN1_LOG="${LOG_DIR}/batch${BATCH}_run1_pretrain_${TIMESTAMP}.log"

    echo -e "\n[STAGE 1] Training Model A (25 epochs, no pruning)" | tee -a "$MASTER_LOG"

    python "$FINETUNE_SCRIPT" \
        --model "$BASE_MODEL" \
        --data "$DATA_PATH" \
        --epochs "$EPOCHS_PRE" \
        --batch_size "$BATCH" 2>&1 | tee "$RUN1_LOG"

    FINETUNE_STATUS=$?
    if [ $FINETUNE_STATUS -ne 0 ]; then
        echo "[ERROR] Base training failed for batch ${BATCH}. Skipping batch." | tee -a "$MASTER_LOG"
        continue
    fi

    MODEL_A_WEIGHTS="${SCRIPTPATH}/runs/finetune_b${BATCH}/weights/best.pt"
    
    if [ ! -f "$MODEL_A_WEIGHTS" ]; then
        echo "[ERROR] Model A weights missing for batch ${BATCH}" | tee -a "$MASTER_LOG"
        continue
    fi

    #############################
    # 2) LOOP OVER PRUNING RATIOS
    #############################
    for PRUNE_PCT in "${PRUNE_LIST[@]}"; do
        
        PRUNE_RATE=$(awk "BEGIN{printf \"%.4f\", ${PRUNE_PCT}/100}")

        echo -e "\n-----------------------------------------------" | tee -a "$MASTER_LOG"
        echo "[STAGE 2] Pruning ${PRUNE_PCT}% for batch ${BATCH}" | tee -a "$MASTER_LOG"
        echo "-----------------------------------------------" | tee -a "$MASTER_LOG"

        PRUNE_LOG="${LOG_DIR}/batch${BATCH}_prune_${PRUNE_PCT}pct_${TIMESTAMP}.log"

        ###########################################
        # 2a) PRUNE MODEL A
        ###########################################
        python "$PRUNING_SCRIPT" \
            --model "$MODEL_A_WEIGHTS" \
            --data "$DATA_PATH" \
            --cfg "default.yaml" \
            --iterative_steps 2 \
            --postprune_epochs "$EPOCHS_POST" \
            --batch_size "$BATCH" \
            --target_prune_rate "$PRUNE_RATE" 2>&1 | tee "$PRUNE_LOG"


        PRUNE_STATUS=$?
        if [ $PRUNE_STATUS -ne 0 ]; then
            echo "[ERROR] Pruning ${PRUNE_PCT}% failed for batch ${BATCH}" | tee -a "$MASTER_LOG"
            continue
        fi

        echo "[Completed] Completed prune+retrain for ${PRUNE_PCT}% (batch ${BATCH})" | tee -a "$MASTER_LOG"

    done  # END PRUNE LOOP

done  # END BATCH LOOP


###############################################
# SUMMARY
###############################################
echo -e "\n===============================================" | tee -a "$MASTER_LOG"
echo "              EXPERIMENT DONE" | tee -a "$MASTER_LOG"
echo "Finished: $(date)" | tee -a "$MASTER_LOG"
echo "Logs: $LOG_DIR" | tee -a "$MASTER_LOG"
echo "===============================================" | tee -a "$MASTER_LOG"

deactivate
