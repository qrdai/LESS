#!/bin/bash

source less/scripts/train/base_training_args.sh

data_dir=$1
model_path=$2
percentage=$3
data_seed=$4
job_name=$5

output_dir=../out/${job_name}
if [[ ! -d $output_dir ]]; then
    mkdir -p $output_dir
fi

train_files=(
    "$data_dir/train/processed/cot/cot_data.jsonl"
    "$data_dir/train/processed/dolly/dolly_data.jsonl"  # finetune a tinyllama on the whole dolly dataset only
    "$data_dir/train/processed/flan_v2/flan_v2_data.jsonl"
    "$data_dir/train/processed/oasst1/oasst1_data.jsonl"
)


# use fsdp for all types of models, in order to obtain `optimizer.bin` with str-based keys
if [[ $model_path == *"llama"* ]]; then
    base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config llama_finetune"
elif [[ $model_path == *"mistral"* ]]; then
    base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config mistral_7b_finetune"
fi


training_args="$base_training_args \
--model_name_or_path $model_path \
--output_dir $output_dir \
--percentage $percentage \
--data_seed $data_seed \
--train_files ${train_files[@]} 2>&1 | tee $output_dir/train.log"

eval "$header" "$training_args"