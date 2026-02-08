from datasets import load_dataset, Dataset
import random
random.seed(42)
import os
import argparse

def make_map_fn(split):
    def process_fn(example, idx):
        messages = example.pop("messages")
        question = messages[0]['content'].replace("\n\nPresent the answer in LaTex format: \\boxed{Your answer}", "")
        # question = f"Human: {question_raw}\nAssistant:"
        question = question + "\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        answer = example.pop("ground_truth")
        data = {
            "data_source": "omega-math",
            "prompt": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
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
    parser.add_argument(
        "--local_save_dir", default="data", help="The save directory for the preprocessed dataset."
    )
    args = parser.parse_args()

    ds = load_dataset("hamishivi/omega-combined-no-boxed_filtered", split=f"train", cache_dir="hf_cache")
    num_samples = 40000
    ds = ds.shuffle(seed=42)
    ds = ds.select(range(num_samples))
    test_ds = load_dataset("allenai/omega-500", split=f"train")
    dataset = ds.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    test_dataset = test_ds.map(function=make_map_fn("test"), with_indices=True, num_proc=8)
    
    print(test_dataset['prompt'][0][0]['content'])
    print(dataset['prompt'][0][0]['content'])

    local_save_dir = os.path.join(args.local_save_dir, "omega")
    dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))