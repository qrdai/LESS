CKPT=1688

TRAINING_DATA_NAME=dolly
TRAINING_DATA_FILE=../data/train/processed/dolly/dolly_data.jsonl # when changing data name, change the data path accordingly
GRADIENT_TYPE="adam"
MODEL_PATH=../out/TinyLlama-1.1B-p0.05-lora-seed3-fsdp/checkpoint-${CKPT}   # should be fsdp-trained ckpt
OUTPUT_PATH=../grads/TinyLlama-1.1B-p0.05-lora-seed3-fsdp/${TRAINING_DATA_NAME}-ckpt${CKPT}-${GRADIENT_TYPE}    # when changing warmup model name, change output dir accordingly
DIMS="8192"

./less/scripts/get_info/grad/get_train_lora_grads.sh "$TRAINING_DATA_FILE" "$MODEL_PATH" "$OUTPUT_PATH" "$DIMS" "$GRADIENT_TYPE"