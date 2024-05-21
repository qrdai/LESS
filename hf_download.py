import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoTokenizer, AutoModelForCausalLM

# Define the model name and the target directory for caching
# model_name_or_path = 'meta-llama/Llama-2-7b-hf'  # replace with your model name
model_name_or_path = 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T'
cache_dir = '/root/autodl-tmp/huggingface/transformers'

# Ensure the cache directory exists
os.makedirs(cache_dir, exist_ok=True)

# Download and cache the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, cache_dir=cache_dir)
