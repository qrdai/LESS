import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Download
# model_name_or_path = 'meta-llama/Llama-2-7b-hf'
# model_name_or_path = 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T'
model_name_or_path = 'mistralai/Mistral-7B-v0.1'
# cache_dir = '/root/autodl-tmp/huggingface/transformers'   # for autodl
cache_dir = '/projects/illinois/eng/cs/haopeng/qirundai/.cache/huggingface/transformers'   # for CC at UIUC

os.makedirs(cache_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, cache_dir=cache_dir)
