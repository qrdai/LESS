source eval.sh

# main evaluation function
# current batch_size -> 1: avoid padding problems (need further review before scaling it)
eval_mmlu() {
    mdir=$1
    evalbsz=$2
    # shift 2 # This shifts the positional parameters, dropping the first two and moving the rest down
    # subjects="$@"  # This captures all additional arguments as subjects

    set_save_dir $mdir mmlu
    save_dir=${save_dir}_evalbsz${evalbsz}
    mkdir -p $save_dir

    cmd="python -m eval.mmlu.run_eval \
    --ntrain 5 \
    --data_dir $DATA_DIR/mmlu \
    --save_dir $save_dir \
    --model_name_or_path $mdir \
    --tokenizer_name_or_path $mdir \
    --eval_batch_size $evalbsz \
    --convert_to_bf16"
    # --n_instances 4
    # --subjects $subjects
    eval "$cmd" 2>&1 | tee $save_dir/eval.log
}

# # evaluate the validation set, which is not supported yet
# valid_mmlu() {
#     mdir=$1
#     type=$2
#     set_valid_dir $mdir mmlu
#     mkdir -p $save_dir
#     cmd="python -m eval.mmlu.run_eval \
#     --ntrain 5 \
#     --eval_valid \
#     --data_dir $DATA_DIR/mmlu \
#     --save_dir $save_dir \
#     --model_name_or_path $mdir \
#     --tokenizer_name_or_path $mdir \
#     --eval_batch_size 4 \
#     --convert_to_bf16"
#     eval "$cmd" 2>&1 | tee $save_dir/log.txt
# }

# extract the results
extract_mmlu() {
    mdir=$1
    evalbsz=$2

    set_save_dir $mdir mmlu
    save_dir=${save_dir}_evalbsz${evalbsz}

    result=$(jq .average_acc $save_dir/metrics.json)
    result=$(echo "$result * 100" | bc)
    echo $result
}

# # extract the results for the validation set
# extract_valid_mmlu() {
#     mdir=$1
#     set_valid_dir $mdir mmlu
#     result=$(jq .average_acc $save_dir/metrics.json)
#     result=$(echo "$result * 100" | bc)
#     echo $result
# }

export -f eval_mmlu
# export -f valid_mmlu
export -f extract_mmlu
# export -f extract_valid_mmlu