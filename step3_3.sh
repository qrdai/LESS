# 3-3. select the top-k training data points with the highest influence scores
TRAIN_FILE_NAMES="cot dolly flan_v2 oasst1" # training point gradients calculated in step2
TARGET_TASK_NAMES="mmlu bbh tydiqa"    # validation point gradients calculated in step3-1
# TARGET_TASK_NAMES="tydiqa"

train_files=(
    "../data/train/processed/cot/cot_data.jsonl"
    "../data/train/processed/dolly/dolly_data.jsonl"
    "../data/train/processed/flan_v2/flan_v2_data.jsonl"
    "../data/train/processed/oasst1/oasst1_data.jsonl"
)   # must be in same order as TRAIN_FILE_NAMES

REFERENCE_MODEL=Llama-2-7b-p0.05-lora-seed3-bsz32
# REFERENCE_MODEL=Mistral-7B-v0.1-p0.05-lora-seed3-bsz32
SELECTED_DATA_OUTPUT_PATH=../selected_data/${REFERENCE_MODEL}


python3 -m less.data_selection.write_selected_data \
--target_task_names ${TARGET_TASK_NAMES} \
--train_file_names ${TRAIN_FILE_NAMES} \
--train_files "${train_files[@]}" \
--output_path $SELECTED_DATA_OUTPUT_PATH \
--percentage 0.05