from datasets import load_dataset, Dataset
from fractions import Fraction
import os
import argparse

def make_map_fn(split):
    def process_fn(example, idx):
        question = example.pop("problem")
        question = question + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        # question = f"Human: {question_raw}\nAssistant:"
        answer = example.pop("answer")
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

    ds = load_dataset("POLARIS-Project/Polaris-Dataset-53K", split=f"train", cache_dir="hf_cache")
    # ds = ds.filter(lambda x: float(Fraction(x['difficulty'])) < 1/8)
    print(len(ds))
    dataset = ds.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    local_save_dir = os.path.join(args.local_save_dir, "polaris")
    print(dataset['prompt'][0][0]['content'])
    dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))