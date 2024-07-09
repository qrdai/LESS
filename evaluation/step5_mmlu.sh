# 1. LESS 5%
# model_name="Llama-2-7b-p0.05-lora-bsz32-selected_data-mmlu/checkpoint-1688" # must specify checkpoint dir here
# 2. random 5%
# model_name="Llama-2-7b-p0.05-lora-seed3-bsz32/checkpoint-1688"
# subjects=("nutrition" "miscellaneous" "sociology" "prehistory" "marketing")  # use all subjects when unspecified


CKPT_list=("1268")
eval_bsz_list=(1)
model_list=(
    "Llama-2-7b-p0.05-lora-bsz32-selected_data-mmlu" # 1. LESS 5%
    "Llama-2-7b-p0.05-lora-seed3-bsz32"              # 2. random 5%
)   # model_list is not the path to ckpts!


# Source the script to use its functions
source eval_mmlu.sh


for eval_bsz in "${eval_bsz_list[@]}"
do
    for CKPT in "${CKPT_list[@]}"
    do
        for model in "${model_list[@]}"
        do
            # Call the evaluation function
            eval_mmlu "${model}/checkpoint-${CKPT}" "$eval_bsz" # "${subjects[@]}"

            # Extract and print results
            extract_mmlu "${model}/checkpoint-${CKPT}" "$eval_bsz"
        done
    done
done