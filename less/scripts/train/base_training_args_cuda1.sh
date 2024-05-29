#!/bin/bash

ID=$RANDOM
# PORT=$((29000 + ID % 1000))  # This generates a port number between 29000 and 29999
export header="torchrun --nproc_per_node 1 --nnodes 1 \
--rdzv-endpoint=localhost:29500 \
--rdzv-id=$ID --rdzv_backend c10d \
-m less.train.train"

# delete --percentage since it's set in `warmup_lora_train.sh`
export base_training_args="--do_train True \
--max_seq_length 2048 \
--use_fast_tokenizer True \
--lr_scheduler_type linear \
--warmup_ratio 0.03 \
--weight_decay 0.0 \
--evaluation_strategy no \
--logging_steps 1 \
--save_strategy no \
--num_train_epochs 4 \
--bf16 True \
--tf32 False \
--fp16 False \
--overwrite_output_dir True \
--report_to wandb \
--optim adamw_torch \
--seed 0 \
--save_strategy epoch \
--lora True \
--lora_r 128 \
--lora_alpha 512 \
--lora_dropout 0.1 \
--lora_target_modules q_proj k_proj v_proj o_proj \
--learning_rate 2e-05 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 32"