DATA_DIR=../data
# MODEL_PATH=meta-llama/Llama-2-7b-hf
MODEL_PATH=mistralai/Mistral-7B-v0.1
PERCENTAGE=0.05 # percentage of the full data to train, you can specify the training file you want to use in the script
DATA_SEED=3
# JOB_NAME=Llama-2-7b-p${PERCENTAGE}-lora-seed${DATA_SEED}-bsz32
JOB_NAME=Mistral-7B-v0.1-p${PERCENTAGE}-lora-seed${DATA_SEED}-bsz32

./less/scripts/train/warmup_lora_train.sh "$DATA_DIR" "$MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" "$JOB_NAME"