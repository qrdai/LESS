# 3-2. calculate the influence score for all training data points
# Note that: the final influence score for selection is the maximum value across all validation subtasks
# So we need extra operations to recover the influence score for EACH subtask
# TRAIN_FILE_NAMES="dolly"    # training point gradients calculated in step2
TRAIN_FILE_NAMES="summarization brainstorming closed_qa creative_writing information_extraction classification open_qa general_qa"
TARGET_TASK_NAMES="mmlu"    # validation point gradients calculated in step3-1
DIM=8192                    # decide which dimension to use
# CKPTS="422 845 1268 1688" # checkpoing index
CKPTS="469 938 1407 1876"
# CHECKPOINT_WEIGHTS="1.6877e-05 1.2859e-05 7.7030e-06 2.5616e-06" # average lr of the epoch; batch_size=128 (420 steps altogether)
# CHECKPOINT_WEIGHTS="1.6775e-05 1.2889e-05 7.7337e-06 2.5779e-06" # batch_size=32 (1688 steps altogether)
CHECKPOINT_WEIGHTS="1.6804e-05 1.2887e-05 7.7320e-06 2.5773e-06" # midpoints of continuous lines

GRADIENT_PATH=../grads/TinyLlama-1.1B-p1.0-lora-seed3-fsdp-dollyonly/{}-ckpt{}-adam/dim${DIM}
VALIDATION_GRADIENT_PATH=../grads/TinyLlama-1.1B-p1.0-lora-seed3-fsdp-dollyonly/{}-ckpt{}-sgd/dim${DIM}
REFERENCE_MODEL=TinyLlama-1.1B-p1.0-lora-seed3-fsdp-dollyonly
SELECTED_DATA_OUTPUT_PATH=../selected_data/${REFERENCE_MODEL}

./less/scripts/data_selection/matching.sh "$GRADIENT_PATH" "$TRAIN_FILE_NAMES" "$CKPTS" "$CHECKPOINT_WEIGHTS" "$VALIDATION_GRADIENT_PATH" "$TARGET_TASK_NAMES" "$SELECTED_DATA_OUTPUT_PATH"