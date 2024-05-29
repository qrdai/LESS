export CUDA_VISIBLE_DEVICES=1

DATA_DIR=../data
# MODEL_PATH=meta-llama/Llama-2-7b-hf
MODEL_PATH=TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T

# NUM_SAMPLES=4500_3750_4500_2250 # 25%
# NUM_SAMPLES=3000_7500_3000_1500 # 50%
NUM_SAMPLES=1500_11250_1500_750 # 75%

DATA_SEED=3
JOB_NAME=TinyLlama-1.1B-n${NUM_SAMPLES}-lora-seed${DATA_SEED}-fsdp   # activate pytorch FSDP with default fsdp_config in training_arguments.py

./less/scripts/train/warmup_lora_train_num_samples_cuda1.sh "$DATA_DIR" "$MODEL_PATH" "$NUM_SAMPLES" "$DATA_SEED" "$JOB_NAME"