# 3-1. obtain SGD gradient features of all validation data, for all model checkpoints
TASK=mmlu
DATA_DIR=../data
DIMS="8192" # We use 8192 as our default projection dimension

CKPT_list=("422" "845" "1268" "1688")
for CKPT in "${CKPT_list[@]}"
do
    MODEL_PATH=../out/TinyLlama-1.1B-p0.05-lora-seed3-fsdp/checkpoint-${CKPT}   # should be fsdp-trained ckpt
    OUTPUT_PATH=../grads/TinyLlama-1.1B-p0.05-lora-seed3-fsdp/${TASK}-ckpt${CKPT}-sgd # for validation data, we always use sgd
done

./less/scripts/get_info/get_eval_lora_grads.sh "$TASK" "$DATA_DIR" "$MODEL_PATH" $OUTPUT_PATH "$DIMS"



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



# 3-3. select the top-k training data points with the highest influence scores
python3 -m less.data_selection.write_selected_data \
--target_task_names ${TARGET_TASK_NAMES} \
--train_file_names ${TRAIN_FILE_NAMES} \
--train_files ../data/train/processed/dolly/dolly_data.jsonl \
--output_path $SELECTED_DATA_OUTPUT_PATH \
--percentage 0.05