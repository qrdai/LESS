# 3-1. obtain SGD gradient features of all validation data, for all model checkpoints
TASK=mmlu
DATA_DIR=../data
DIMS="8192" # We use 8192 as our default projection dimension

# CKPT_list=("422" "845" "1268" "1688") # original 4 datasets-p0.05-bsz32
CKPT_list=("211" "422" "634" "844") # original 4 datasets-p0.05-bsz32-fsdp on 2 GPUs (for mistral-7B)
# REFERENCE_MODEL=Llama-2-7b-p0.05-lora-seed3-bsz32
REFERENCE_MODEL=Mistral-7B-v0.1-p0.05-lora-seed3-bsz32

for CKPT in "${CKPT_list[@]}"
do
    MODEL_PATH=../out/${REFERENCE_MODEL}/checkpoint-${CKPT}   # should be fsdp-trained ckpt
    OUTPUT_PATH=../grads/${REFERENCE_MODEL}/${TASK}-ckpt${CKPT}-sgd # for validation data, we always use sgd
    ./less/scripts/get_info/grad/get_eval_lora_grads.sh "$TASK" "$DATA_DIR" "$MODEL_PATH" $OUTPUT_PATH "$DIMS"
done
