import argparse
import json
from pathlib import Path
import time
from typing import Any

import pandas as pd
from optimize import make_train_function, make_train_function_ray

data_path = "data/textClust_stream_data/raw_data"

datasets = {
    "Arabic_7" : f"{data_path}/Arabic_7.npy",
    "Arabic_Twitter" : f"{data_path}/Arabic_Twitter.npy",
    "CrisisLexT6" : f"{data_path}/CrisisLexT6.npy",
    "CrisisLexT7" : f"{data_path}/CrisisLexT7.npy",
    "CrisisLexT26" : f"{data_path}/CrisisLexT26.npy",
    "CrisisMMD" : f"{data_path}/CrisisMMD.npy",
    "CrisisNLP" : f"{data_path}/CrisisNLP.npy",
    "Event2012" : f"{data_path}/Event2012.npy",
    "Event2018" : f"{data_path}/Event2018.npy",
    "HumAID" : f"{data_path}/HumAID.npy",
    "KBP" : f"{data_path}/KBP.npy",
    "MAVEN" : f"{data_path}/MAVEN.npy",
    "Mix_Data" : f"{data_path}/Mix_Data.npy",
    "Semantic5-T" : f"{data_path}/Semantic5-T.json",
    "Semantic10-T" : f"{data_path}/Semantic10-T.json",
    "Semantic40-T" : f"{data_path}/Semantic40-T.json",
    "NT" : f"{data_path}/NT.json",
    "NTS" : f"{data_path}/NTS.json",
    "NTSO-T" : f"{data_path}/NTSO-T.json",
    "SO-T" : f"{data_path}/SO-T.json",
    "News-T" : f"{data_path}/News-T.json",
    "Trends-T" : f"{data_path}/Trends-T.json",
    "Tweets-T" : f"{data_path}/Tweets-T.json",
}

def _evaluate_task(algorithm, dataset, config) -> dict[str, Any]:
    if algorithm == "inc_clust_ray":
        train = make_train_function_ray(max_samples=None)
    else:
        train = make_train_function(algorithm, max_samples=None, throughput_limit=0.0)
    performance: dict[str, Any] = train(config, instance=dataset, seed=None)[1]
    performance["dataset"] = dataset
    performance["algorithm"] = algorithm
    return performance


def load_config(config_path):
    """Load configurations from a JSON file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")

    with path.open() as config_file:
        config = json.load(config_file)

    if not isinstance(config, list):
        raise ValueError(
            f"Configuration file must contain a JSON list, got {type(config).__name__}."
        )
    if not all(isinstance(item, dict) for item in config):
        raise ValueError("Every item in the configuration list must be a JSON object.")
    return config


def evaluate_config(configs, config_name, output_csv, algorithm, dataset_names=None):
    """Evaluate configurations on selected datasets and write a CSV."""
    results = []
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_datasets = (
        datasets
        if dataset_names is None
        else {name: datasets[name] for name in dataset_names}
    )
    for config_index, config in enumerate(configs):
        current_config_name = f"{config_name}[{config_index}]"
        for dataset_name, dataset in selected_datasets.items():
            print(
                f"Evaluating '{current_config_name}' on dataset: "
                f"{dataset_name} ({dataset})"
            )
            try:
                performance = _evaluate_task(algorithm, dataset, config)
            except Exception as error:
                print(
                    f"Evaluation failed for config '{current_config_name}' "
                    f"on dataset '{dataset_name}': {error}"
                )
                continue
            performance["dataset_name"] = dataset_name
            performance["config"] = current_config_name
            performance["config_index"] = config_index
            results.append(performance)
            pd.DataFrame(results).to_csv(output_path, index=False)
            print(f"Saved results after dataset '{dataset_name}' to {output_path}")

    print(f"Finished writing evaluation results to {output_path}")
    return results


def main(config_path, algorithm, output_csv, dataset_names=None):
    configs = load_config(config_path)
    config_name = Path(config_path).name
    print(f"Evaluating {len(configs)} configurations from '{config_name}' on all datasets...")
    print(f"Evaluation started at {time.ctime(time.time())}")
    return evaluate_config(
        configs,
        config_name=config_name,
        output_csv=output_csv,
        algorithm=algorithm,
        dataset_names=dataset_names,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_path",
        help="Path to a JSON file containing a list of configuration objects.",
    )
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
