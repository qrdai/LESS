# Source the script to use its functions
source eval_tydiqa.sh

CKPT_list=("1688" "1268")
eval_seed_list=(3407)   # 1 42 3407
eval_bsz_list=(20)     # 1 20
model_list=(
    "Llama-2-7b-p0.05-lora-bsz32-selected_data-tydiqa" # 1. LESS 5%
    "Llama-2-7b-p0.05-lora-seed3-bsz32"                # 2. random 5%
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
                eval_tydiqa "$model_name" "$eval_seed" "$eval_bsz"

                # Extract and print results
                extract_tydiqa "$model_name" "$eval_seed" "$eval_bsz"
            done
        done
    done
done