# fixed arguments
GRADIENT_TYPE="adam"
DIMS="8192"

# looping args
CKPT_list=("469" "938" "1407" "1876")
subset_list=("summarization" "brainstorming" "closed_qa" "creative_writing" "information_extraction" "classification" "open_qa" "general_qa")

# nested loops
for CKPT in "${CKPT_list[@]}"
do
    for TRAINING_DATA_NAME in "${subset_list[@]}"
    do
        TRAINING_DATA_FILE=../data/train/processed/dolly/dolly_data_${TRAINING_DATA_NAME}.jsonl # when changing data name, change the data path accordingly
        MODEL_PATH=../out/TinyLlama-1.1B-p1.0-lora-seed3-fsdp-dollyonly/checkpoint-${CKPT}   # should be fsdp-trained ckpt
        OUTPUT_PATH=../grads/TinyLlama-1.1B-p1.0-lora-seed3-fsdp-dollyonly/${TRAINING_DATA_NAME}-ckpt${CKPT}-${GRADIENT_TYPE}    # when changing warmup model name, change output dir accordingly
        ./less/scripts/get_info/grad/get_train_lora_grads.sh "$TRAINING_DATA_FILE" "$MODEL_PATH" "$OUTPUT_PATH" "$DIMS" "$GRADIENT_TYPE"
    done
done