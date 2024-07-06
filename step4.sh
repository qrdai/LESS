# the final lora-tuning step just uses the default seed 0

# TARGET_TASK_NAME="mmlu"
TARGET_TASK_NAME="bbh"
# TARGET_TASK_NAME="tydiqa"
PERCENTAGE=0.05
REFERENCE_MODEL=Llama-2-7b-p0.05-lora-seed3-bsz32

TRAIN_FILES=../selected_data/${REFERENCE_MODEL}/${TARGET_TASK_NAME}/top_p${PERCENTAGE}.jsonl

MODEL_PATH=meta-llama/Llama-2-7b-hf
# MODEL_PATH=mistralai/Mistral-7B-v0.1

JOB_NAME=Llama-2-7b-p${PERCENTAGE}-lora-bsz32-selected_data-${TARGET_TASK_NAME}
# JOB_NAME=Mistral-7B-v0.1-p${PERCENTAGE}-lora-bsz32-selected_data-${TARGET_TASK_NAME}

./less/scripts/train/lora_train.sh "$TRAIN_FILES" "$MODEL_PATH" "$JOB_NAME" 