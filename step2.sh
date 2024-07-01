# fixed arguments
GRADIENT_TYPE="adam"
DIMS="8192"

# looping args
CKPT_list=("422" "845" "1268" "1688")   # original 4 datasets-p0.05-bsz32
dataset_list=("cot" "dolly" "flan_v2" "oasst1")

REFERENCE_MODEL=Llama-2-7b-p0.05-lora-seed3-bsz32

# nested loops
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