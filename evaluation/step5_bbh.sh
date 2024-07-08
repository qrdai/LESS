# Source the script to use its functions
source eval_bbh.sh

CKPT_list=("1688" "1268")
eval_seed_list=(1 3407)
# eval_bsz_list=(10)
eval_bsz_list=(1)


for eval_bsz in "${eval_bsz_list[@]}"
do
    for eval_seed in "${eval_seed_list[@]}"
    do
        for CKPT in "${CKPT_list[@]}"
        do
            model_name="Llama-2-7b-p0.05-lora-bsz32-selected_data-bbh/checkpoint-${CKPT}" # must specify checkpoint dir here

            # Call the evaluation function
            eval_bbh "$model_name" "$CKPT" "$eval_seed" "$eval_bsz"

            # Extract and print results
            extract_bbh "$model_name" "$CKPT" "$eval_seed" "$eval_bsz"
        done
    done
done