import argparse
import json
from pathlib import Path
import time

import pandas as pd
from optimize import datasets, make_train_function, make_train_function_ray

def _evaluate_task(algorithm, dataset, config):
    if algorithm == "inc_clust_ray":
        train = make_train_function_ray(max_samples=None)
    else:
        train = make_train_function(algorithm, max_samples=None, throughput_limit=0.0)
    performance = train(config, instance=dataset, seed=None)[1]
    performance["dataset"] = dataset
    performance["algorithm"] = algorithm
    return performance


def load_config(config_path):
    """Load one configuration from a JSON file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")

    with path.open() as config_file:
        config = json.load(config_file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration file must contain one JSON object, got {type(config).__name__}."
        )
    return config


def evaluate_config(config, config_name, output_csv, algorithm, dataset_names=None):
    """Evaluate a configuration on selected datasets and write a CSV."""
    results = []
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_datasets = (
        datasets
        if dataset_names is None
        else {name: datasets[name] for name in dataset_names}
    )
    for dataset_name, dataset in selected_datasets.items():
        print(f"Evaluating '{config_name}' on dataset: {dataset_name} ({dataset})")
        try:
            performance = _evaluate_task(algorithm, dataset, config)
        except Exception as error:
            print(f"Evaluation failed for dataset '{dataset_name}': {error}")
            continue
        performance["dataset_name"] = dataset_name
        performance["config"] = config_name
        results.append(performance)
        pd.DataFrame(results).to_csv(output_path, index=False)
        print(f"Saved results after dataset '{dataset_name}' to {output_path}")

    print(f"Finished writing evaluation results to {output_path}")
    return results


def main(config_path, algorithm, output_csv, dataset_names=None):
    config = load_config(config_path)
    config_name = Path(config_path).name
    print(f"Evaluating configuration '{config_name}' on all datasets...")
    print(f"Evaluation started at {time.ctime(time.time())}")
    return evaluate_config(
        config,
        config_name=config_name,
        output_csv=output_csv,
        algorithm=algorithm,
        dataset_names=dataset_names,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to a JSON file containing one configuration.")
    parser.add_argument(
        "algorithm",
        choices=["embClust", "textClust", "inc_clust_ray"],
        help="Algorithm used to evaluate the configuration.",
    )
    parser.add_argument("output_csv", help="Path where the evaluation CSV will be written.")
    parser.add_argument(
        "--dataset",
        dest="dataset_names",
        nargs="+",
        choices=list(datasets),
        help="Dataset name(s) to evaluate. If omitted, evaluate all datasets.",
    )
    args = parser.parse_args()

    main(args.config_path, args.algorithm, args.output_csv, args.dataset_names)
