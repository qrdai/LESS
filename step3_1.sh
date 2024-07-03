# 3-1. obtain SGD gradient features of all validation data, for all model checkpoints
DATA_DIR=../data
DIMS="8192" # We use 8192 as our default projection dimension

CKPT_list=("422" "845" "1268" "1688") # original 4 datasets-p0.05-bsz32
task_list=("mmlu" "bbh" "tydiqa")

REFERENCE_MODEL=Llama-2-7b-p0.05-lora-seed3-bsz32

for CKPT in "${CKPT_list[@]}"
do
    for TASK in "${task_list[@]}"
    do
        MODEL_PATH=../out/${REFERENCE_MODEL}/checkpoint-${CKPT}   # should be fsdp-trained ckpt
        OUTPUT_PATH=../grads/${REFERENCE_MODEL}/${TASK}-ckpt${CKPT}-sgd # for validation data, we always use sgd
        ./less/scripts/get_info/grad/get_eval_lora_grads.sh "$TASK" "$DATA_DIR" "$MODEL_PATH" $OUTPUT_PATH "$DIMS"
    done
done
