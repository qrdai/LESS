from datasets import load_dataset


cache_dir = '/root/autodl-tmp/huggingface/datasets'
dataset_name_or_path = 'databricks/databricks-dolly-15k'

dataset = load_dataset(dataset_name_or_path, cache_dir=cache_dir)

first_ten_samples = dataset["train"][:10]
responses = first_ten_samples["response"]
for idx, resp in enumerate(responses):
    print(f"dolly_{idx}: {resp}\n")
