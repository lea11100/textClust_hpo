# Compare the performance of textClust, embClust, and fastClust

import argparse
import time
import numpy as np
import pandas as pd
from typing import Optional

from textClust.core.textClust import textClust
from textClust.data.wrapper import StaticJsonWrapper
from textClust.algorithms.presets import get_textClust, get_embClust, get_fastClust, get_fastLogClust


DEFAULT_DATASETS = [
    "examples/data/News-T.json",
    "examples/data/NT.json",
    "examples/data/NTS.json",
    "examples/data/NTSO-T.json",
    "examples/data/Semantic10-T.json",
    "examples/data/Semantic5-T.json",
    "examples/data/Semantic5-T_sorted.json",
    "examples/data/SO-T.json",
    "examples/data/Trends-T_modified.json",
    "examples/data/Trends-T.json",
    "examples/data/Tweets-T.json",
]

DEFAULT_ALGORITHMS = ["textClust", "embClust", "fastLogClust", "fastClust"]


def test_algorithm(algorithm: textClust, name: str = "textClust") -> dict:
    algorithm.verbose = False
    print(f"Starting {name}...")
    start_time = time.time()
    algorithm.start()
    end_time = time.time()
    print(f"{name} finished in {end_time - start_time:.2f} seconds.")
    
    flat_result = { 
                   "algorithm": name,
                   "duration": end_time - start_time,
                   "median_num_macroclusters": np.median([h.get("num_macro_clusters", 0.0) for h in algorithm.history_manager.eval_history]),
                   "median_v_meassure": np.median([h.get("v_measure", 0.0) for h in algorithm.history_manager.eval_history]),
                   "median_homogeneity": np.median([h.get("homogeneity", 0.0) for h in algorithm.history_manager.eval_history]),
                   "median_completeness": np.median([h.get("completeness", 0.0) for h in algorithm.history_manager.eval_history]),
                   "median_v_meassure_final": np.median([h.get("v_measure_final", 0.0) for h in algorithm.history_manager.eval_history]),
                   "median_homogeneity_final": np.median([h.get("homogeneity_final", 0.0) for h in algorithm.history_manager.eval_history]),
                   "median_completeness_final": np.median([h.get("completeness_final", 0.0) for h in algorithm.history_manager.eval_history])
                   }
    
    return flat_result


def test_dataset(
    dataset: str,
    max_samples: Optional[int],
    do_check: bool,
    clean_interval: int,
    cluster_interval: int,
    log_interval: int,
    decay_rate: float,
    sigma: float,
    embedding_model_name: str,
    device: str,
    algorithms: list[str],
):
    print(f"Testing dataset: {dataset}")
    data_loader = StaticJsonWrapper(dataset, max_samples=max_samples, do_check=do_check)

    clusterers = {}

    if "textClust" in algorithms:
        clusterers["textClust"] = get_textClust(
            data_loader=data_loader,
            clean_interval=clean_interval,
            cluster_interval=cluster_interval,
            log_interval=log_interval,
            decay_rate=decay_rate,
            sigma=sigma,
        )

    if "embClust" in algorithms:
        clusterers["embClust"] = get_embClust(
            data_loader=data_loader,
            embedding_model_name=embedding_model_name,
            clean_interval=clean_interval,
            cluster_interval=cluster_interval,
            log_interval=log_interval,
            decay_rate=decay_rate,
            sigma=sigma,
            device=device,
        )

    if "fastLogClust" in algorithms:
        clusterers["fastLogClust"] = get_fastLogClust(
            data_loader=data_loader,
            clean_interval=clean_interval,
            cluster_interval=cluster_interval,
            log_interval=log_interval,
            decay_rate=decay_rate,
            sigma=sigma,
        )

    if "fastClust" in algorithms:
        clusterers["fastClust"] = get_fastClust(
            data_loader=data_loader,
            clean_interval=clean_interval,
            cluster_interval=cluster_interval,
            log_interval=log_interval,
            decay_rate=decay_rate,
            sigma=sigma,
        )

    results = []

    for name in algorithms:
        clusterer = clusterers[name]
        result = test_algorithm(clusterer, name=name)
        result["dataset"] = dataset
        results.append(result)
    
    print(f"Finished {dataset}.\n")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare textClust variants over one or more datasets.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, help="Dataset file paths to evaluate.")
    parser.add_argument("--algorithms", nargs="+", choices=DEFAULT_ALGORITHMS, default=DEFAULT_ALGORITHMS, help="Algorithms to run.")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples loaded from each dataset. If omitted, load all samples.")
    parser.add_argument("--do-check", action=argparse.BooleanOptionalAction, default=True, help="Enable or disable wrapper validation checks.")
    parser.add_argument("--clean-interval", type=int, default=100, help="Microcluster clean interval.")
    parser.add_argument("--cluster-interval", type=int, default=1000, help="Macro clustering interval.")
    parser.add_argument("--log-interval", type=int, default=100, help="Evaluation logging interval.")
    parser.add_argument("--decay-rate", type=float, default=0.005, help="Decay rate for fading function.")
    parser.add_argument("--sigma", type=float, default=1.0, help="Sigma threshold for neighborhood checks.")
    parser.add_argument("--embedding-model-name", default="sentence-transformers/paraphrase-MiniLM-L3-v2", help="Embedding model for embClust.")
    parser.add_argument("--device", default="mps", help="Device for embClust embeddings (e.g. cpu, cuda, mps).")
    parser.add_argument("--output", default="textClust_comparison_results.csv", help="Path of the output CSV file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results = []
    for dataset in args.datasets:
        results.extend(
            test_dataset(
                dataset=dataset,
                max_samples=args.max_samples,
                do_check=args.do_check,
                clean_interval=args.clean_interval,
                cluster_interval=args.cluster_interval,
                log_interval=args.log_interval,
                decay_rate=args.decay_rate,
                sigma=args.sigma,
                embedding_model_name=args.embedding_model_name,
                device=args.device,
                algorithms=args.algorithms,
            )
        )

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
