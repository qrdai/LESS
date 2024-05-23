# 3-2. calculate the influence score for all training data points
# Note that: the final influence score for selection is the maximum value across all validation subtasks
# TODO: need extra operations to recover the influence score for EACH subtask
TRAIN_FILE_NAMES="dolly"    # training point gradients calculated in step2
TARGET_TASK_NAMES="mmlu"    # validation point gradients to be calculated in step3
DIM=8192                    # decide which dimension to use
CKPTS="422 845 1268 1688" # checkpoing index
# CHECKPOINT_WEIGHTS="1.6877e-05 1.2859e-05 7.7030e-06 2.5616e-06" # average lr of the epoch; batch_size=128 (420 steps altogether)
CHECKPOINT_WEIGHTS="1.6775e-05 1.2889e-05 7.7337e-06 2.5779e-06" # batch_size=32 (1688 steps altogether)

GRADIENT_PATH=../grads/TinyLlama-1.1B-p0.05-lora-seed3-fsdp/{}-ckpt{}-adam/dim${DIM}
VALIDATION_GRADIENT_PATH=../grads/TinyLlama-1.1B-p0.05-lora-seed3-fsdp/{}-ckpt{}-sgd/dim${DIM}
SELECTED_DATA_OUTPUT_PATH="../selected_data"

./less/scripts/data_selection/matching.sh "$GRADIENT_PATH" "$TRAIN_FILE_NAMES" "$CKPTS" "$CHECKPOINT_WEIGHTS" "$VALIDATION_GRADIENT_PATH" "$TARGET_TASK_NAMES" "$SELECTED_DATA_OUTPUT_PATH"