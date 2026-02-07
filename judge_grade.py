import numpy as np
import pandas as pd
import argparse
import os
import itertools
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams, PoolingParams
import json
from verl.utils.reward_score.deepscaler import rllm_reward_fn_math, remove_boxed, last_boxed_only_string
from typing import List, Union


CoT_Passk_Prompt='''You are an expert in mathematics and logical reasoning. Your task is to evaluate the correctness of a solution to a given math (or logical reasoning) problem, with a **strong emphasis on the reasoning process**, not just the final answer.
Below is the **Problem** and the **Solution (Provided by another AI model)**:
—
**Problem**:
{problem}
**Solution (Provided by another AI model)**:
{solution}
—
Please perform the following tasks:
1. **Analyze the solution step-by-step**, paying close attention to: - Computational accuracy - Logical consistency - Conceptual understanding - Whether the reasoning is valid and
complete
2. **Identify any issues or errors in the reasoning**, even if the final answer is correct. Classify them into the following categories (if applicable): - **Calculation Error**: Mistakes in
arithmetic, algebraic manipulation, or numerical computation. - **Logical Error**: Invalid
reasoning, flawed logic, or incorrect inference. - **Conceptual Error**: Misunderstanding
or misuse of mathematical concepts or definitions. - **Omission / Incompleteness**: Missing steps, incomplete justification, or not addressing all parts of the question. - **Other**:
Any other type of error that does not fit into the above categories.
3. **Provide a final judgment** on whether the solution is logically sound and free of errors
in reasoning.
Please format your response as follows:
—
**Issues Identified:**
- [Issue 1]: [Classification] - [Brief explanation] - [Issue 2]: [Classification] - [Brief explanation] - ...
Let’s think step by step and output your final judgment within \\boxed{{}}
\\boxed{{yes}} or \\boxed{{no}}'''

ProcessBenchPrompt='''The following is a math problem and a solution (split into paragraphs, enclosed with tags and indexed from 0):

[Math Problem]
{problem}

[Solution]

<paragraph_0>
</paragraph_n-1>

Your task is to review and critique the solution paragraph by paragraph. Once you identify an
error in a paragraph, return the index of the paragraph where the earliest error occurs. Otherwise,
return the index of -1 (which typically denotes "not found").

Please put your final answer (i.e., the index) in \\boxed{{}}.'''


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
    total = []
    correct = []
    llm = LLM("models/Qwen3-30B-A3B-Instruct-2507-FP8", gpu_memory_utilization=0.8, tensor_parallel_size=2, max_num_seqs=1024)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(truncate_prompt_tokens=32768, temperature=0.7, top_p=0.8, top_k=20, min_p=0, max_tokens=16384)

    tqdm_loader = tqdm(range(len(df)))
    all_idxs = []
    all_judge_prompts= []
    all_prompts = []

    for i in tqdm_loader:
        prompt = df[i]['prompt']
        responses = df[i]['responses']
        gt = df[i]['gt']
        if args.benchmark == "AIME24":
            gt = remove_boxed(gt)


        for j, resp in enumerate(responses):
            # resp_splitted = resp.split("\n\n")
            score = rllm_reward_fn_math("", resp, gt)
            if score == 1.0:
                judge_prompt = CoT_Passk_Prompt.format(problem=prompt, solution=resp)
                all_judge_prompts.append(tokenizer.apply_chat_template([{"role": "user", "content": judge_prompt}], tokenize=False, add_generation_prompt=True))
                all_idxs.append(i)
                all_prompts.append(prompt)
            # messages = [
            #     {"role": "system", "content": system_prompt},
            #     {"role": "user", "content": prompt},
            #     {"role": "assistant", "content": "<extra_0>".join(resp_splitted) + "<extra_0>"},
            # ]
            # conv = tokenizer.apply_chat_template(
            #     messages, 
            #     tokenize=False, 
            #     add_generation_prompt=False
            # )
    outputs = llm.generate(all_judge_prompts, use_tqdm=True, sampling_params=sampling_params)
    responses = [ele.outputs[0].text for ele in outputs]
    final_ans = [last_boxed_only_string(ele.outputs[0].text) for ele in outputs]
    all_scores = []
    for ele in final_ans:
        if ele is None:
            score = 0
        else:
            resp = remove_boxed(ele)
            if resp.strip().lower() == 'yes':
                score = 1
            else:
                score = 0
        all_scores.append(score)
    
    
    output_dir = f"eval_results/{args.model_name}"
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame()
    df['benchmark'] = [args.benchmark] * len(all_prompts)
    df['idx'] = all_idxs
    df['prompt'] = all_prompts
    df['responses'] = responses
    df['score'] = all_scores
    df.to_parquet(f"{output_dir}/pass_judge_{args.benchmark}.parquet")

    # json_path = os.path.join(output_dir, 'pass_judge.jsonl')
    # row_data = {
    #     'model_name': file_path.split("/")[1],
    #     "prompt_type": "CoT-Pass@k",
    #     'dataset': args.benchmark,
    #     'raw_scores': correct,
    #     'total': total[0],
    # }

    # ks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    # total = np.array(total)
    # correct = np.array(correct)
    # pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean().item()
    #             for k in ks if (total >= k).all()}
    # print(pass_at_k)
    # for k, v in pass_at_k.items():
    #     row_data[k] = v
    # print("JSON path:", json_path)
    # # Write to CSV
    # with open(json_path, 'a+') as f:
    #     json.dump(row_data, f)
    #     f.write('\n')
    