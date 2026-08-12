import json 
import os
import pandas as pd
from pathlib import Path
import argparse
import time

print("Base modules loaded at", pd.Timestamp.now())

from optimize import datasets, make_train_function

print("All modules loaded at", time.ctime(time.time())) # Fomatted into human readable time

default_config = {
    "embClust": {
        "clean_interval": 100,
        "cluster_interval": 1000,
        "decay_rate": 0.0005,
        "min_num_mc": 1,
        "adaptive_threshold": 0,
        "radius": 0.3,
        "threshold_momentum": 0.5,
        "use_heuristic": 1,
        "time_gap": 100,
        "model": "sentenceTransformers",
    },
    "textClust": {
        "clean_interval": 100,
        "cluster_interval": 1000,
        "decay_rate": 0.0005,
        "min_num_mc": 1,
        "adaptive_threshold": 0,
        "radius": 0.3,
        "threshold_momentum": 0.5,
        "use_heuristic": 1,
        "time_gap": 100,
    }
}


def _evaluate_task(task):
    config, dataset, config_name, config_id, loo_dataset, seed, algorithm = task
    print("EVALUATING:", dataset, config_name, config_id, loo_dataset, seed, algorithm)
    
    train = make_train_function(algorithm, max_samples=None, throughput_limit=0.0, num_points=1000)
    performance = train(config, instance=dataset)[1]
    print("FINISHED EVALUATION:", dataset, config_name, config_id, loo_dataset, seed, algorithm, performance)
    performance["dataset"] = dataset
    performance["config"] = config_name
    performance["config_id"] = config_id
    performance["loo_dataset"] = loo_dataset
    performance["seed"] = seed
    performance["algorithm"] = algorithm
    return performance


def load_config(config_path):
    if config_path == "default":
        return default_config

    with open(config_path, "r") as f:
        configs = json.load(f)

    if not configs:
        raise ValueError(f"No configurations found in {config_path}")

    return configs


def evaluate_config(config, config_name="loaded_config", output_csv="final_loo_performances.csv", algorithm="textClust"):
    tasks = []
    
    if config_name == "default":
        loo_dataset = "none"
        seed = "none"
    else:
        loo_dataset = config_name.split("_")[3]
        seed = config_name.split("_")[5].split(".")[0]

    for dataset_name in datasets.values():
        for i,c in enumerate(config if isinstance(config, list) else [config]):
            tasks.append((c, dataset_name, config_name, i, loo_dataset, seed, algorithm))

    results = [_evaluate_task(task) for task in tasks]
    print(f"Finished evaluation at {time.ctime(time.time())}")
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    return df


def main(config_path=None, output_csv="final_loo_performances.csv", use_default_config=False, algorithm="textClust"):
    #Get folder for output CSV 
    output_folder = Path(output_csv).parent
    os.makedirs(output_folder, exist_ok=True)

    if use_default_config:
        config = default_config[algorithm]
        config_name = "default"
    else:
        if config_path is None:
            raise ValueError("Provide a config_path or pass --use-default-config to evaluate the built-in configuration.")
        config = load_config(f"smac3_results/{algorithm}/{config_path}")
        config_name = Path(config_path).name
    print(f"Evaluating configuration '{config_name}' on all datasets...")
    print(f"Evaluation started at {time.ctime(time.time())}")
    return evaluate_config(config, config_name=config_name, output_csv=output_csv, algorithm=algorithm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", nargs="?", default=None, help="Path to a JSON file containing one or more configurations.")
    parser.add_argument("--algorithm", default="textClust", help="The algorithm to be optimized.")
    parser.add_argument("--use-default-config", action="store_true", help="Evaluate the built-in default configuration instead of loading a JSON file.")
    parser.add_argument("--output-csv", default="final_loo_performances.csv", help="Where to write the results CSV.")
    args = parser.parse_args()

    main(
        args.config_path,
        output_csv=args.output_csv,
        use_default_config=args.use_default_config,
        algorithm=args.algorithm
    )
