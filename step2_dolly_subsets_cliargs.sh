# export CUDA_VISIBLE_DEVICES=0

# fixed arguments
GRADIENT_TYPE="adam"
DIMS="8192"

# looping args
subset_list=("summarization" "brainstorming" "closed_qa" "creative_writing" "information_extraction" "classification" "open_qa" "general_qa")

# assigned by external script
# CKPT_list=("422" "845" "1268" "1688")   # TinyLlama-1.1B-p0.05-lora-seed3-fsdp
# CKPT_list=("469" "938" "1407" "1876")   # TinyLlama-1.1B-p1.0-lora-seed3-fsdp-dollyonly

# all but last argument are CKPT_list
CKPT_list=("${@:1:$#-1}")

# REFERENCE_MODEL=TinyLlama-1.1B-p0.05-lora-seed3-fsdp
# REFERENCE_MODEL=TinyLlama-1.1B-p1.0-lora-seed3-fsdp-dollyonly

# last argument is REFERENCE_MODEL
REFERENCE_MODEL="${@: -1}"

# nested loops
for CKPT in "${CKPT_list[@]}"
do
    for TRAINING_DATA_NAME in "${subset_list[@]}"
    do
        TRAINING_DATA_FILE=../data/train/processed/dolly/dolly_data_${TRAINING_DATA_NAME}.jsonl # when changing data name, change the data path accordingly
        MODEL_PATH=../out/${REFERENCE_MODEL}/checkpoint-${CKPT}   # should be fsdp-trained ckpt
        OUTPUT_PATH=../grads/${REFERENCE_MODEL}/${TRAINING_DATA_NAME}-ckpt${CKPT}-${GRADIENT_TYPE}    # when changing warmup model name, change output dir accordingly
        ./less/scripts/get_info/grad/get_train_lora_grads.sh "$TRAINING_DATA_FILE" "$MODEL_PATH" "$OUTPUT_PATH" "$DIMS" "$GRADIENT_TYPE"
    done
done