from datasets import load_dataset, Dataset
import os
import argparse

def make_map_fn(split):
    def process_fn(example, idx):
        question = example.pop("problem")
        # question = f"Human: {question_raw}\nAssistant:"
        question = question + "\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        # solution = example.pop("qwen_7B_solution")
        answer = example.pop("gt_answer")
        data = {
            "data_source": "math",
            "prompt": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "solution": " " + solution,
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": split,
                "index": idx,
            },
        }
        return data

    return process_fn

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="gsm8k")
    parser.add_argument(
        "--local_save_dir", default="data/evolm", help="The save directory for the preprocessed dataset."
    )
    args = parser.parse_args()

    ds = load_dataset("ZhentingNLP/mathaug-disjoint", split="sampled_500k_balanced_last_100k", cache_dir="hf_cache")
    # test_ds = load_dataset("ZhentingNLP/mathaug-disjoint", split="sampled_500k_balanced_last_400k", cache_dir="hf_cache")
    # test_ds = test_ds.select(range(512))

    ds = ds.shuffle(seed=42)
    ds = ds.select(range(40_000))
    ds = ds.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    # test_ds = test_ds.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    # print(ds['prompt'][0])
    # print('\n')
    # print(test_ds['prompt'][0])
    ds.to_parquet(os.path.join(args.local_save_dir, "train.parquet"))
    # test_ds.to_parquet(os.path.join(args.local_save_dir, "test.parquet"))