model_name="Llama-2-7b-p0.05-lora-bsz32-selected_data-mmlu/checkpoint-1688" # must specify checkpoint dir here
# subjects=("nutrition" "miscellaneous" "sociology" "prehistory" "marketing")  # use all subjects when unspecified

# Source the script to use its functions
source eval_mmlu.sh

# Call the evaluation function
eval_mmlu "$model_name" # "${subjects[@]}"

# Extract and print results
extract_mmlu "$model_name"