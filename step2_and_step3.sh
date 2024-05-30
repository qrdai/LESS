# step2_dolly_subsets + step3_1 + step3_2
export CUDA_VISIBLE_DEVICES=1   # assign only one GPU for gradient calculation

model_list=(
    # "TinyLlama-1.1B-n4500_3750_4500_2250-lora-seed3-fsdp"
    # "TinyLlama-1.1B-n3000_7500_3000_1500-lora-seed3-fsdp"
    "TinyLlama-1.1B-n1500_11250_1500_750-lora-seed3-fsdp"
)
ckpt_list=("468" "937" "1406" "1872")
ckpts="468 937 1406 1872"

for REFERENCE_MODEL in "${model_list[@]}"
do
    ./step2_dolly_subsets_cliargs.sh "${ckpt_list[@]}" "$REFERENCE_MODEL"
    ./step3_1_cliargs.sh "${ckpt_list[@]}" "$REFERENCE_MODEL"
    ./step3_2_cliargs.sh "$ckpts" "$REFERENCE_MODEL"
done