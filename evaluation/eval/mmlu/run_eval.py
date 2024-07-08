import argparse
import json
import os
import time
# import sys
# print(sys.path)   # inspect search paths for import statements

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from eval.mmlu.categories import categories, subcategories
from eval.utils import (dynamic_import_function, get_next_word_predictions,
                        load_hf_lm_and_tokenizer, query_openai_chat_model)

choices = ["A", "B", "C", "D"]  # correspond to indices 0,1,2,3


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)   # include_answer default to True
    return prompt


@torch.no_grad()
def eval_hf_model(args, subject, model, tokenizer, dev_df, test_df, batch_size=1, k=5):
    prompts = []
    chat_formatting_function = dynamic_import_function(
        args.chat_formatting_function) if args.use_chat_format else None
    for i in range(0, test_df.shape[0]):
        prompt_end = format_example(test_df, i, include_answer=False)   # test question from test_df
        train_prompt = gen_prompt(dev_df, subject, k)   # few-shot examples from dev_df
        prompt = train_prompt + prompt_end

        # if i == 0:
        #     print(f"Few-shot examples:\n\n{train_prompt}\n")
        # print(f"Test set Question {i+1}:\n{prompt_end}\n")

        if args.use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            prompt = chat_formatting_function(messages, add_bos=False)
            if prompt[-1] in ["\n", " "]:
                prompt += "The answer is:"
            else:
                prompt += " The answer is:"

        tokenized_prompt = tokenizer(
            prompt, truncation=False, add_special_tokens=False).input_ids
        # make sure every prompt is less than 2048 tokens, by iteratively deleting few-shot examples
        while len(tokenized_prompt) > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

            if args.use_chat_format:
                messages = [{"role": "user", "content": prompt}]
                prompt = chat_formatting_function(messages, add_bos=False)
                if prompt[-1] in ["\n", " "]:
                    prompt += "The answer is:"
                else:
                    prompt += " The answer is:"

            tokenized_prompt = tokenizer(
                prompt, truncation=False, add_special_tokens=False).input_ids

        prompts.append(prompt) # len(prompts) == test_df.shape[0], storing prompts for all test questions

    # get the answer for all examples
    # adding a prefix space here, as that's expected from the prompt
    # TODO: should raise a warning if this returns more than one token
    answer_choice_ids = [tokenizer.encode(
        " " + answer_choice, add_special_tokens=False)[-1] for answer_choice in choices]
    pred_indices, all_probs = get_next_word_predictions(
        model, tokenizer, prompts, candidate_token_ids=answer_choice_ids, return_token_predictions=False, batch_size=batch_size
    )

    # get the metrics
    cors = []
    groud_truths = test_df.iloc[:, -1].values

    print(f"\nSubject {subject} - Pred & GT:\n")
    print(f"pred_indices:\n{pred_indices}\n")
    print(f"groud_truths:\n{groud_truths}\n")

    for i in range(len(pred_indices)):
        prediction = choices[pred_indices[i]]
        ground_truth = groud_truths[i]
        cors.append(prediction == ground_truth)

    acc = np.mean(cors)
    cors = np.array(cors)   # [0,1,1,0,0,1, ...] arranged in the same order as `test_df`

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))
    return cors, acc, all_probs


def eval_openai_chat_engine(args, subject, engine, dev_df, test_df, batch_size=1):

    import tiktoken
    gpt_tokenizer = tiktoken.get_encoding("cl100k_base")
    # be careful, the tokenizer will tokenize " A" and "A" differently.
    answer_choice_ids = [gpt_tokenizer.encode(" " + x)[0] for x in choices]

    prompts = []
    for i in range(0, test_df.shape[0]):
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        prompts.append(prompt)

    instances = [{"id": prompt, "prompt": prompt}
                 for _, prompt in enumerate(prompts)]
    results = query_openai_chat_model(
        engine=args.openai_engine,
        instances=instances,
        batch_size=args.eval_batch_size if args.eval_batch_size else 10,
        output_path=os.path.join(
            args.save_dir, f"{subject}_openai_results.jsonl"),
        logit_bias={token_id: 100 for token_id in answer_choice_ids},
        max_tokens=1,
    )

    # get the metrics
    cors = []
    groud_truths = test_df.iloc[:, -1].values
    for i in range(len(test_df)):
        prediction = results[i]["output"].strip()
        ground_truth = groud_truths[i]
        cors.append(prediction == ground_truth)

    acc = np.mean(cors)
    cors = np.array(cors)

    # dummy probs, just don't want to dig into the openai probs
    all_probs = np.array([[0.25, 0.25, 0.25, 0.25]
                         for _ in range(len(test_df))])

    print("Average accuracy {:.3f} - {}".format(acc, subject))
    return cors, acc, all_probs


def main(args):

    if args.model_name_or_path:
        print(f"Loading model and tokenizer from {args.model_name_or_path}")
        model, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            load_in_8bit=args.load_in_8bit, # False
            device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
            gptq_model=args.gptq,   # False
            use_fast_tokenizer=not args.use_slow_tokenizer, # False
            convert_to_bf16=args.convert_to_bf16,   # True
            convert_to_half=args.convert_to_half,   # False
        )

    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    if args.subjects:   # loaded as a list; can include only part of all the 57 mmlu tasks
        assert all(
            subj in subjects for subj in args.subjects), f"Some of the subjects you specified are not valid: {args.subjects}"
        subjects = args.subjects

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}  # cat in ["STEM", "humanities", "social sciences", "other (business, health, misc.)"]

    for subject in tqdm(subjects, desc=f"Evaluating subjects: "):

        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]    # args.ntrain: # of few-shot examples
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )
        if args.n_instances and args.n_instances < test_df.shape[0]:    # args.n_instances: max # of test questions
            # test_df = test_df.sample(args.n_instances, random_state=42) # random_seed for test data subsampling fixed to 42
            test_df = test_df.head(args.n_instances)
            # print(test_df)

        if args.model_name_or_path:
            if args.eval_valid: # evaluate the validation set, which is not supported yet
                test_df = dev_df
            
            cors, acc, probs = eval_hf_model(
                args, subject, model, tokenizer, dev_df, test_df, args.eval_batch_size, k=args.ntrain if not args.eval_valid else 0)
        else:
            cors, acc, probs = eval_openai_chat_engine(
                args, subject, args.openai_engine, dev_df, test_df, args.eval_batch_size)

        print(f"\nSubject {subject} - Cors and Probs:\n")
        print(f"cors:\n{cors}\n")
        print(f"probs:\n{probs}\n")

        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)    # subcat_cors[subcat]: [[0,1,0,...], [1,0,1,...], ...  ]
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)  # cat_cors[key]: [[0,1,0,...], [1,0,1,...], ...  ]
        all_cors.append(cors)   # all_cors: [[0,1,0,...], [1,0,1,...], ...  ]

        test_df["correct"] = cors   # test_df and cors both point to the same subject
        for j in range(probs.shape[1]): # probs.shape[1] == 4
            choice = choices[j]
            test_df["choice{}_probs".format(choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(
                args.save_dir, "{}.csv".format(subject)
            ),
            index=None,
        )

    # print(f"subcat_cors:\n{subcat_cors}\n")
    for subcat in subcat_cors:
        if subcat_cors[subcat] == []:
            continue
        # print(f"subcat:\n{subcat}\n")
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        if cat_cors[cat] == []:
            continue
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))

    # save results
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "average_acc": weighted_acc,
                "subcat_acc": {
                    subcat: np.mean(np.concatenate(subcat_cors[subcat]))
                    for subcat in subcat_cors if subcat_cors[subcat] != []
                },
                "cat_acc": {
                    cat: np.mean(np.concatenate(cat_cors[cat]))
                    for cat in cat_cors if cat_cors[cat] != []
                },
            },
            f,
            indent=4
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ntrain",
        type=int,
        default=5,
        help="# few-shot examples in dev"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/mmlu"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/mmlu/llama-7B/"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the tokenizer from here."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--openai_engine",
        type=str,
        default=None,
        help="if specified, we will use the OpenAI API to generate the predictions."
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        help="which subjects to evaluate. If not specified, all the 57 subjects will be evaluated."
    )
    parser.add_argument(
        "--n_instances",
        type=int,
        help="if specified, a maximum of n_instances per subject will be used for the evaluation."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference."
    )
    parser.add_argument(
        "--gptq",
        action="store_true",
        help="If given, we're evaluating a 4-bit quantized GPTQ model."
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function",
        type=str,
        default="eval.templates.create_prompt_with_tulu_chat_format",
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    parser.add_argument(
        "--convert_to_half",
        action="store_true",
        help="Load model in half.",
    )
    parser.add_argument(
        "--convert_to_bf16",
        action="store_true",
        help="Load model in bf16.",
    )
    parser.add_argument(
        "--eval_valid",
        action="store_true",
        help="If given, we will use gpu for inference.")  # not supported yet

    args = parser.parse_args()

    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None) != (
        args.openai_engine is None), "Either model_name_or_path or openai_engine should be specified."

    # add prefix to model_name_or_path and tokenizer_name_or_path
    base_path = "/projects/illinois/eng/cs/haopeng/qirundai"
    args.model_name_or_path = f"{base_path}/out/{args.model_name_or_path}"
    args.tokenizer_name_or_path = args.model_name_or_path

    main(args)
