from datasets import load_dataset
import argparse
import os
import itertools
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams, PoolingParams
import json
from verl.utils.reward_score.deepscaler import rllm_reward_fn_math, remove_boxed, last_boxed_only_string
from typing import List, Union


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def process_jsonl_file(file_name):
    """
    Process a JSONL file and dynamically handle the number of problems.
    """
    results = []
    with open(file_name) as f:
        for line in f:
            data = json.loads(line)
            id = int(data["example_id"])
            while len(results) <= id:  # Ensure the list is large enough
                results.append({"gt": None, "responses": [], "prompt": None})
            gt = data["answer"]
            response = data["response"]
            prompt = data['prompt']
            results[id]["gt"] = gt
            results[id]["prompt"] = prompt
            results[id]["responses"].append(response)
    return results


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("-b", "--benchmark", type=str, default="AIME24")
    args = parser.parse_args()

    benchmark_dict = {
        "AIME24": {
            "n": 1024,
        },
        "AIME25": {
            "n": 1024,
        },
        "AMC23": {
            "n": 1024,
        },
        "Math-500": {
            "n": 128,
        },
        "GSM8K-Plat": {
            "n": 128,
        },
        "Olympiad-Bench": {
            "n": 128,
        },
        "EvoLM": {
            "n": 16,
        },
        "Omega": {
            "n": 16,
        },
        "Polaris": {
            "n": 16,
        },
        "Polaris-hard": {
            "n": 16,
        },

    }
    # file_path = f"gen_outputs/EvoLM-1B-160BT-MixedFW8FM42-400k-evolm-GRPO-step300/{args.benchmark.lower()}_t0.7_p0.95_n{benchmark_dict[args.benchmark]['n']}-MNT3072.jsonl"
    n = 16
    file_path = f"gen_outputs/{args.model_name}/{args.benchmark.lower()}_t0.6_p0.95_n{n}-MNT3072.jsonl"
    df = process_jsonl_file(file_path)
    tqdm_loader = tqdm(range(len(df)))
    threshold = 0.7
    total = []
    correct = []
    system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."

    prm_model = LLM("models/Qwen2.5-Math-PRM-7B", gpu_memory_utilization=0.7, task="reward")
    tokenizer = prm_model.get_tokenizer()
    pooling_params = PoolingParams(truncate_prompt_tokens=4096)
    
    for i in tqdm_loader:
        prompt = df[i]['prompt']
        responses = df[i]['responses']
        gt = df[i]['gt']
        if args.benchmark == "AIME24":
            gt = remove_boxed(gt)

        all_convs = []

        for i, resp in enumerate(responses):
            score = rllm_reward_fn_math("", resp, gt)
            if score == 1.0:
                resp_splitted = resp.split("\n\n")
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "<extra_0>".join(resp_splitted) + "<extra_0>"},
                ]
                conv = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                all_convs.append(conv)
        if len(all_convs) == 0:
            # correct.append(0)
            prm_scores = [0]
        else:
            outputs = prm_model.reward(all_convs, use_tqdm=False, pooling_params=pooling_params)
            prm_scores = [ele.outputs.data for ele in outputs]
            prm_scores = [float((ele[:, 1] > threshold).all().item()) for ele in prm_scores]
            # total.append(len(responses))
            total.append(len(all_convs))
            correct.append(sum(prm_scores))
        tqdm_loader.set_postfix(acc=np.mean(prm_scores).item())
        # else:
        
    
    output_dir = f"eval_results/{args.model_name}"
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, 'pass_prm_correct.jsonl')
    row_data = {
        'model_name': file_path.split("/")[1] + "-PRM",
        "threshold": threshold,
        'dataset': args.benchmark,
        'raw_scores': correct,
        'total': total,
    }

    # ks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    # total = np.array(total)
    # correct = np.array(correct)
    # pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean().item()
    #             for k in ks if (total >= k).all()}
    # print(pass_at_k)
    # for k, v in pass_at_k.items():
    #     row_data[k] = v
    print("JSON path:", json_path)
    # Write to CSV
    with open(json_path, 'a+') as f:
        json.dump(row_data, f)
        f.write('\n')
    