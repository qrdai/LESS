# Source the script to use its functions
source eval_bbh.sh

CKPT_list=("1688" "1268")
eval_seed_list=(42)
eval_bsz_list=(10)
model_list=(
    "Llama-2-7b-p0.05-lora-bsz32-selected_data-bbh" # 1. LESS 5%
    "Llama-2-7b-p0.05-lora-seed3-bsz32"              # 2. random 5%
)   # model_list is not the path to ckpts!


for eval_bsz in "${eval_bsz_list[@]}"
do
    for eval_seed in "${eval_seed_list[@]}"
    do
        for CKPT in "${CKPT_list[@]}"
        do
            for model in "${model_list[@]}"
            do
                model_name="${model}/checkpoint-${CKPT}"

                # Call the evaluation function
                eval_bbh "$model_name" "$eval_seed" "$eval_bsz"

                # Extract and print results
                extract_bbh "$model_name" "$eval_seed" "$eval_bsz"
            done
        done
    done
done