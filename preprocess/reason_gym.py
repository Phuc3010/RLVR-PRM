#!/usr/bin/env python3

"""
Script to generate Reasoning Gym datasets and save them to the Hugging Face Hub.
"""
from tqdm import tqdm
import argparse
import pandas as pd
import json
from typing import Dict, List, Optional, Any
import reasoning_gym
import yaml
from datasets import Dataset
from tqdm import tqdm
import os
from reasoning_gym.composite import DatasetSpec
from reasoning_gym.factory import DATASETS, create_dataset


def generate_dataset(
    dataset_names: List[str],
    dataset_size: int = 20000,
    seed: int = 42,
    split: str = "train",
    weights: Optional[Dict[str, float]] = None,
    configs: Optional[Dict[str, Dict]] = None,
) -> Dataset:
    """
    Generate a dataset from the specified Reasoning Gym datasets.

    Args:
        dataset_names: List of dataset names to include
        dataset_size: Total size of the dataset to generate
        seed: Random seed for dataset generation
        weights: Optional dictionary mapping dataset names to weights
        configs: Optional dictionary mapping dataset names to configurations

    Returns:
        A Hugging Face Dataset object
    """
    # Validate dataset names
    for name in dataset_names:
        if name not in DATASETS:
            raise ValueError(f"Dataset '{name}' not found. Available datasets: {sorted(DATASETS.keys())}")

    # Set default weights if not provided
    if weights is None:
        equal_weight = 1.0 / len(dataset_names)
        weights = {name: equal_weight for name in dataset_names}
    else:
        # Validate weights
        for name in dataset_names:
            if name not in weights:
                weights[name] = 0.0
                print(f"Warning: No weight provided for {name}, setting to 0.0")

    # Set default configs if not provided
    if configs is None:
        configs = {name: {} for name in dataset_names}
    else:
        # Add empty configs for missing datasets
        for name in dataset_names:
            if name not in configs:
                configs[name] = {}

    # Create dataset specs
    dataset_specs = [DatasetSpec(name=name, weight=weights[name], config=configs[name]) for name in dataset_names]

    # Create composite dataset
    data_source = create_dataset("composite", seed=seed, size=dataset_size, datasets=dataset_specs)

    # Generate all examples
    def map_process(example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        question = example.pop('question')
        answer = example.pop('answer')
        developer_prompt = reasoning_gym.utils.SYSTEM_PROMPTS["simple"]
        source_dataset = example['metadata']['source_dataset']
        try:
            metadata = json.dumps(example['metadata'])
        except Exception as e:
            variable_keys = example['metadata']['variables'].keys()
            for k in variable_keys:
                if "fraction" in k:
                    example['metadata']['variables'][k] = float(example['metadata']['variables'][k])
            metadata = json.dumps(example['metadata'])
        data = {
            "data_source": source_dataset,
            "prompt": [
                {
                    "role": "system",
                    "content": developer_prompt
                },
                {
                "role": "user",
                "content": question
            }],
            "ability": "math",
            "reward_model": {
                "answer": str(answer),
                "question": question,
                "metadata": metadata
            },
            "extra_info": {
                'split': "train" if split == "buffer" else split,
                'index': idx
            }
        }
        return data
    examples = []
    for idx in tqdm(range(dataset_size), desc="Generating examples"):
        example = map_process(data_source[idx], idx=idx)
        if example['reward_model']['answer'] != 'None':
            examples.append(example)

    # Convert to HF Dataset
    dataset = pd.DataFrame(examples)
    return dataset


def save_to_hub(
    dataset: Dataset,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False,
    commit_message: str = "Upload reasoning_gym dataset",
    split: Optional[str] = None,
) -> str:
    """
    Save the dataset to the Hugging Face Hub.

    Args:
        dataset: HF Dataset to save
        repo_id: Hugging Face repo ID (e.g., "username/dataset-name")
        token: HF API token
        private: Whether the repository should be private
        commit_message: Commit message
        split: Dataset split name

    Returns:
        URL of the uploaded dataset
    """
    # Push to the hub
    dataset.push_to_hub(
        repo_id,
        token=token,
        private=private,
        commit_message=commit_message,
    )

    print(f"Dataset pushed to https://huggingface.co/datasets/{repo_id}")
    return f"https://huggingface.co/datasets/{repo_id}"



def main():
    parser = argparse.ArgumentParser(description="Generate and upload Reasoning Gym datasets to HF Hub")
    parser.add_argument("--size", type=int, default=37000, help="Total dataset size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--split", type=str, choices=["train", "test"], default="train", help="Dataset split name"
    )

    # First parse args to check for config file
    args = parser.parse_args()

    # Validate repo_id is provided
    # Load configuration
    dataset_names = []
    weights = {}
    weights = None
    configs = {}

    # Load from config file if provided
    print(f"Dataset size: {args.size}")
    print(f"Dataset seed: {args.seed}")
    dataset_names = list(DATASETS.keys())[1:]
    os.makedirs(f"data/reasoning_gym/{args.split}", exist_ok=True)

    if args.split != 'test':
        dataset_names = [ele for ele in dataset_names if ele not in ["acre", "boxnet", "graph_color", "game_of_life_halting", "puzzle24"]]
        size = 4800 // len(dataset_names)
        all_ds = []
    else:
        size = 100

    for ds_name in tqdm(dataset_names):
        dataset = generate_dataset(
            dataset_names=[ds_name],
            dataset_size=size,
            seed=args.seed,
            weights=weights,
            split=args.split,
            configs=None,
        )
        if args.split == 'test':
            dataset.to_parquet(f"data/reasoning_gym/{args.split}/{ds_name}.parquet")
        else:
            all_ds.append(dataset)
    
    print(f"Total tasks: {len(all_ds)}")
    
    dataset = pd.concat(all_ds)
    dataset.to_parquet(f"data/reasoning_gym/{args.split}.parquet")

if __name__ == "__main__":
    main()