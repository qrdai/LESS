# 3-1. obtain SGD gradient features of all validation data, for all model checkpoints
TASK=mmlu
DATA_DIR=../data
DIMS="8192" # We use 8192 as our default projection dimension

# CKPT_list=("422" "845" "1268" "1688")
CKPT_list=("469" "938" "1407" "1876")
for CKPT in "${CKPT_list[@]}"
do
    MODEL_PATH=../out/TinyLlama-1.1B-p1.0-lora-seed3-fsdp-dollyonly/checkpoint-${CKPT}   # should be fsdp-trained ckpt
    OUTPUT_PATH=../grads/TinyLlama-1.1B-p1.0-lora-seed3-fsdp-dollyonly/${TASK}-ckpt${CKPT}-sgd # for validation data, we always use sgd
    ./less/scripts/get_info/grad/get_eval_lora_grads.sh "$TASK" "$DATA_DIR" "$MODEL_PATH" $OUTPUT_PATH "$DIMS"
done
