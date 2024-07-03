# fixed arguments
GRADIENT_TYPE="adam"
DIMS="8192"

# looping args
CKPT_list=("1688")   # original 4 datasets-p0.05-bsz32
# CKPT_list=("211" "422" "634" "844") # original 4 datasets-p0.05-bsz32-fsdp on 2 GPUs (for mistral-7B)
# dataset_list=("cot" "dolly" "flan_v2" "oasst1")
dataset_list=("flan_v2")

REFERENCE_MODEL=Llama-2-7b-p0.05-lora-seed3-bsz32
# REFERENCE_MODEL=Mistral-7B-v0.1-p0.05-lora-seed3-bsz32


# nested loops for 4_datasets-ckpt-1688
for CKPT in "${CKPT_list[@]}"
do
    for TRAINING_DATA_NAME in "${dataset_list[@]}"
    do
        TRAINING_DATA_FILE=../data/train/processed/${TRAINING_DATA_NAME}/${TRAINING_DATA_NAME}_data.jsonl # when changing data name, change the data path accordingly
        MODEL_PATH=../out/${REFERENCE_MODEL}/checkpoint-${CKPT}   # should be fsdp-trained ckpt
        OUTPUT_PATH=../grads/${REFERENCE_MODEL}/${TRAINING_DATA_NAME}-ckpt${CKPT}-${GRADIENT_TYPE}    # when changing warmup model name, change output dir accordingly
        ./less/scripts/get_info/grad/get_train_lora_grads.sh "$TRAINING_DATA_FILE" "$MODEL_PATH" "$OUTPUT_PATH" "$DIMS" "$GRADIENT_TYPE"
    done
done