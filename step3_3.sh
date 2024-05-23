# 3-3. select the top-k training data points with the highest influence scores
python3 -m less.data_selection.write_selected_data \
--target_task_names ${TARGET_TASK_NAMES} \
--train_file_names ${TRAIN_FILE_NAMES} \
--train_files ../data/train/processed/dolly/dolly_data.jsonl \
--output_path $SELECTED_DATA_OUTPUT_PATH \
--percentage 0.05