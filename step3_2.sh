# 3-2. calculate the influence score for all training data points
# Note that: the final influence score for selection is the maximum value across all validation subtasks
# So we need extra operations to recover the influence score for EACH subtask

TRAIN_FILE_NAMES="cot dolly flan_v2 oasst1" # training point gradients calculated in step2
TARGET_TASK_NAMES="mmlu bbh tydiqa" # validation point gradients calculated in step3-1
# TARGET_TASK_NAMES="tydiqa"
DIM=8192                    # decide which dimension to use

CKPTS="422 845 1268 1688"   # original 4 datasets-p0.05-bsz32
# CKPTS="211 422 634 844"     # original 4 datasets-p0.05-bsz32-fsdp on 2 GPUs (for mistral-7B)

# CHECKPOINT_WEIGHTS="1.6877e-05 1.2859e-05 7.7030e-06 2.5616e-06" # average lr of the epoch; batch_size=128 (420 steps altogether)
# CHECKPOINT_WEIGHTS="1.6775e-05 1.2889e-05 7.7337e-06 2.5779e-06" # batch_size=32 (1688 steps altogether)
CHECKPOINT_WEIGHTS="1.6804e-05 1.2887e-05 7.7320e-06 2.5773e-06" # midpoints of continuous lines

REFERENCE_MODEL=Llama-2-7b-p0.05-lora-seed3-bsz32
# REFERENCE_MODEL=Mistral-7B-v0.1-p0.05-lora-seed3-bsz32

GRADIENT_PATH=../grads/${REFERENCE_MODEL}/{}-ckpt{}-adam/dim${DIM}
VALIDATION_GRADIENT_PATH=../grads/${REFERENCE_MODEL}/{}-ckpt{}-sgd/dim${DIM}
SELECTED_DATA_OUTPUT_PATH=../selected_data/${REFERENCE_MODEL}

./less/scripts/data_selection/matching.sh "$GRADIENT_PATH" "$TRAIN_FILE_NAMES" "$CKPTS" "$CHECKPOINT_WEIGHTS" "$VALIDATION_GRADIENT_PATH" "$TARGET_TASK_NAMES" "$SELECTED_DATA_OUTPUT_PATH"