DATA_DIR=../data
# MODEL_PATH=meta-llama/Llama-2-7b-hf
MODEL_PATH=TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
PERCENTAGE=0.05 # percentage of the full data to train, you can specify the training file you want to use in the script
DATA_SEED=3
JOB_NAME=TinyLlama-1.1B-p${PERCENTAGE}-lora-seed${DATA_SEED}-fsdp   # activate pytorch FSDP with default fsdp_config in training_arguments.py

./less/scripts/train/warmup_lora_train.sh "$DATA_DIR" "$MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" "$JOB_NAME"