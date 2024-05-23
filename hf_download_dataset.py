import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datasets import load_dataset
cache_dir = '/root/autodl-tmp/huggingface/datasets'
dataset_name_or_path = 'databricks/databricks-dolly-15k'

# Ensure the cache directory exists
os.makedirs(cache_dir, exist_ok=True)

# Load the dataset from the specified mirror and cache it in the custom directory
dataset = load_dataset(dataset_name_or_path, cache_dir=cache_dir)

print(dataset["train"][:10])
