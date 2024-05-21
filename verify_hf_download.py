import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'   # only needed for the first download
# os.environ['HF_HOME'] = '/root/autodl-tmp/huggingface'    # doesn't work; should still use `cache_dir`
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import time


def verify_and_measure_time(args):
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if args.load_mode == "cuda":
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir, device_map="auto") # 已经自动做了 device_map, 那就不需要 .to(device)
        # model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto")
    elif args.load_mode == "cpu":
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir).to(args.load_mode)
        # model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(args.load_mode)
    end_time_1 = time.time()    # to measure time Loading checkpoint shards

    # Ensure the model is in evaluation mode
    model.eval()
    inputs = tokenizer(args.input_text, return_tensors="pt").to(args.load_mode)

    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, max_length=100)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    end_time_2 = time.time()

    print("LOAD_MODE: ", args.load_mode)
    print("Generated text:", generated_text)
    print("Time Cost for loading shards: ", end_time_1 - start_time)
    print("Time Cost for generating text: ", end_time_2 - end_time_1)
    print("Total time cost: ", end_time_2 - start_time)
    print('\n')


def main():
    parser = argparse.ArgumentParser(description='args for verifying models downloaded from hf')
    parser.add_argument('--model_name_or_path', type=str, help='path to hf model')
    parser.add_argument('--cache_dir', type=str, help='path to cache dir containing model weights', default="/root/autodl-tmp/huggingface/transformers")
    parser.add_argument('--load_mode', type=str, help='loading with cuda or cpu', choices=["cuda", "cpu"])
    parser.add_argument('--input_text', type=str, help='prompt for starting completion', default="I am a friendly assistant called ")

    args = parser.parse_args()

    args.input_text = r'''I am a friendly assistant called '''
    verify_and_measure_time(args)


if __name__ == "__main__":
    main()