import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datasets import load_dataset

dataset_name_or_path = 'databricks/databricks-dolly-15k'
# cache_dir = '/root/autodl-tmp/huggingface/datasets' # for autodl
cache_dir = '/projects/illinois/eng/cs/haopeng/qirundai/.cache/huggingface/datasets'    # for CC at UIUC

os.makedirs(cache_dir, exist_ok=True)

dataset = load_dataset(dataset_name_or_path, cache_dir=cache_dir)

print(dataset["train"][:10])
