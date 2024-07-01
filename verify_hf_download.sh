# input_text is assigned inside the python script

# for load_mode in "cuda" "cpu"
# do
#     python verify_hf_download.py \
#         --model_name_or_path "meta-llama/Llama-2-7b-hf" \
#         --cache_dir "/root/autodl-tmp/huggingface/transformers" \
#         --load_mode $load_mode
# done

python verify_hf_download.py \
    --model_name_or_path "mistralai/Mistral-7B-v0.1" \
    --cache_dir "/projects/illinois/eng/cs/haopeng/qirundai/.cache/huggingface/transformers" \
    --load_mode "cuda"