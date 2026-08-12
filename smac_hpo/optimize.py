import argparse
import sys
import os
import json
import shutil
import sys
import signal
from pathlib import Path
import time
from typing import List, Optional, Tuple
import numpy as np
from sklearn import metrics

os.environ.setdefault("NLTK_DISABLE_IMPORT_SECURITY", "1")

import pandas as pd

sys.path.append("/Users/lms/Documents/Misc/local_packages/pygmo2/build")

from textClust.history.manager import HistoryManager
from textClust.offline.cluster import Macrocluster

from ConfigSpace import Configuration, ConfigurationSpace
from smac.facade import MultiObjectiveFacade
from smac import Scenario
from ConfigSpace.read_and_write import pcs_new as pcs

from textClust.algorithms.presets import get_embClust, get_fastLogClust
from textClust.data.wrapper import StaticJsonWrapper, SocialEDWrapper
from textClust.utils.eval import eval_against_truth
from textClust.offline.clusterer import SimpleMacroClusterer
from textClust.history.manager import HistoryManager
from textClust.online.clusterer import Microclusterer
from textClust.preprocessing.preprocessor import SimulatedEmbeddingPreprocessor

social_ed_path = "data/social_ed_data"
data_path = "data"

datasets = {
    "Arabic_7" : f"{social_ed_path}/Arabic_7.npy",
    "Arabic_Twitter" : f"{social_ed_path}/Arabic_Twitter.npy",
    "CrisisLexT6" : f"{social_ed_path}/CrisisLexT6.npy",
    "CrisisLexT7" : f"{social_ed_path}/CrisisLexT7.npy",
    "CrisisLexT26" : f"{social_ed_path}/CrisisLexT26.npy",
    "CrisisMMD" : f"{social_ed_path}/CrisisMMD.npy",
    "CrisisNLP" : f"{social_ed_path}/CrisisNLP.npy",
    "Event2012" : f"{social_ed_path}/Event2012.npy",
    "Event2018" : f"{social_ed_path}/Event2018.npy",
    "HumAID" : f"{social_ed_path}/HumAID.npy",
    "KBP" : f"{social_ed_path}/KBP.npy",
    "MAVEN" : f"{social_ed_path}/MAVEN.npy",
    "Mix_Data" : f"{social_ed_path}/Mix_Data.npy",
    "Semantic5-T" : f"{data_path}/Semantic5-T.json",
    "Semantic10-T" : f"{data_path}/Semantic10-T.json",
    "Semantic40-T" : f"{data_path}/Semantic40-T.json",
}

TIME_LIMIT = int((30_000 + 500) / 2)  # Maximum time limit in seconds (30,000 points at 2 points/sec + 500 seconds buffer)

def timeout_handler(signum, frame):
    raise RuntimeError("algorithm.start() exceeded time limit of {} seconds.".format(TIME_LIMIT))

class ThroughputLimitHistoryManager(HistoryManager):
    def __init__(self, lower_bound: float, num_points: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_time = time.time()
        self.lower_bound = lower_bound
        self.num_points = num_points

    def snapshot(self, current_time: float, clusterer: Microclusterer):
        super().snapshot(current_time, clusterer)
        elapsed_time = time.time() - self.current_time
        throughput = self.num_points / elapsed_time if elapsed_time > 0 else float('inf')
        if throughput < self.lower_bound:
            raise RuntimeError(f"Throughput {throughput:.2f} is below the lower bound of {self.lower_bound}. Terminating optimization.")
        self.current_time = time.time()  # Reset timer for next snapshot

def make_train_function(algorithm: str, max_samples:Optional[int]=30_000, throughput_limit: float = 2.0, log_after_n_points: int = 1000):
    
    def get_algorithm(config: Configuration, seed:Optional[int] = 1,instance:str = ""):
        if instance.endswith(".npy"):
            data_loader = SocialEDWrapper(instance, max_samples=max_samples, random_start_seed=seed)
        else:
            data_loader = StaticJsonWrapper(instance, max_samples=max_samples, random_start_seed=seed)
        if algorithm == "embClust":
            embedding_model_name = config.get("model", "sentenceTransformers")
            embClust = get_embClust(data_loader=data_loader,
                                embedding_model=None, # The embedding model will be simulated in the preprocessor
                                log_interval=log_after_n_points,
                                **config)
            dataset = Path(instance).stem
            embClust.preprocessor = SimulatedEmbeddingPreprocessor(embedding_model_name, dataset)
            return embClust
        elif algorithm == "textClust":
            return get_fastLogClust(data_loader=data_loader,
                                 log_interval=log_after_n_points,
                                 **config)
        else: 
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        
    def train(config: Configuration, seed: Optional[int] = 1, instance:str = "") -> Tuple[dict[str, float], dict[str, str]]:
        algorithm = get_algorithm(config, seed, instance)

        algorithm.verbose = False
        algorithm.history_manager = ThroughputLimitHistoryManager(lower_bound=throughput_limit, num_points=log_after_n_points)  # Set a lower bound for throughput (e.g., 10 points/sec)
        
        if(max_samples is not None):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(TIME_LIMIT)
        start_time = time.time()
        try:
            algorithm.start()
        except Exception as e:
            signal.alarm(0)  # cancel the alarm
            raise RuntimeError(f"Algorithm failed during execution: {e}")
        signal.alarm(0)  # cancel the alarm
        end_time = time.time()
        result = eval_against_truth(
            initial_text_assignments=algorithm.history_manager.initial_text_assignments,
            final_text_assignments=algorithm.history_manager.get_final_assignments(),
            true_assignments=algorithm.history_manager.true_assignments
        )
        avg_throughput = len(algorithm.history_manager.true_assignments) / (end_time - start_time)
        result["avg_throughput"] = avg_throughput
        result["total_time"] = end_time - start_time
        
        return {"-adj_mutual_info":-1 * result.get("adj_mutual_info", 0.0), 
                "-throughput":-np.log(avg_throughput)}, result # Negate because SMAC minimizes by default
    
    return train

def make_train_function_ray(max_samples:Optional[int]=30_000):
    
    sys.path.append("/Users/lms/Documents/Forschung/textClust_hpo/")
    sys.path.append("/Users/lms/Documents/Forschung/textClust_hpo/other_clusterer/twitter_incremental_clustering")
    from other_clusterer.twitter_incremental_clustering.run_detection import run as run_ray
    
    
    def train(config: Configuration, seed: Optional[int] = 1, instance:str = "") -> Tuple[dict[str, float], dict[str, str]]:
        if(max_samples is not None):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(TIME_LIMIT)
        start_time = time.time()
        try:
            config_dict = dict(config)
            config_dict["model"] = [config_dict.get("model", "")]
            config_dict["dataset"] = instance
            config_dict["seed"] = seed
            config_dict["max_samples"] = max_samples
            config_dict["output_dir"] = "."
            config_dict["lang"] = "en"
            config_dict["sub_model"] = "None"
            y_true, y_pred_final = run_ray(config_dict)
        except Exception as e:
            signal.alarm(0)  # cancel the alarm
            raise RuntimeError(f"Algorithm failed during execution: {e}")
        signal.alarm(0)  # cancel the alarm
        end_time = time.time()
        rand_final = metrics.rand_score(y_true, y_pred_final)
        adj_rand_final = metrics.adjusted_rand_score(y_true, y_pred_final)
        mi_final = metrics.mutual_info_score(y_true, y_pred_final)
        adj_mi_final = metrics.adjusted_mutual_info_score(y_true, y_pred_final)
        norm_mi_final = metrics.normalized_mutual_info_score(y_true, y_pred_final)
        homogeneity_final, completeness_final, v_measure_final = metrics.homogeneity_completeness_v_measure(y_true, y_pred_final)
        fowlkes_mallows_final = metrics.fowlkes_mallows_score(y_true, y_pred_final)
        
        num_evaluated_points = len(y_true)
        num_clusters_final = len(np.unique(y_pred_final))
        num_clusters_true = len(np.unique(y_true))
        
        result = {
            "rand_final": rand_final,
            "adj_rand_final": adj_rand_final,
            "mutual_info_final": mi_final,
            "adj_mutual_info_final": adj_mi_final,
            "normalized_mutual_info_final": norm_mi_final,
            "homogeneity_final": homogeneity_final,
            "completeness_final": completeness_final,
            "v_measure_final": v_measure_final,
            "fowlkes_mallows_final": fowlkes_mallows_final,
            "num_evaluated_points": num_evaluated_points,
            "num_clusters_final": num_clusters_final,
            "num_clusters_true": num_clusters_true
        }
        avg_throughput = num_evaluated_points / (end_time - start_time)
        result["avg_throughput"] = avg_throughput
        result["total_time"] = end_time - start_time
        
        return {"-adj_mutual_info":-1 * result.get("adj_mutual_info_final", 0.0), 
                "-throughput":-np.log(avg_throughput)}, result # Negate because SMAC minimizes by default
    
    return train

# Add args for seed and instance selection if needed
args = argparse.ArgumentParser()
args.add_argument("--seed", type=int, default=1, help="Random seed for SMAC (default: 1).")
args.add_argument("--loo_datasets", nargs="+", type=str, default="none", help="Datasets to leave out for evaluation (default: 'none').")
args.add_argument("--algorithm", type=str, default="embClust", choices=["embClust", "textClust", "inc_clust_ray"], help="Algorithm to optimize (default: 'embClust').")
    
if __name__ == "__main__":
    parsed_args = args.parse_args()
    
    print(f"Starting optimization with seed {parsed_args.seed}, algorithm {parsed_args.algorithm}, and leave-one-out datasets: {parsed_args.loo_datasets}")

    if parsed_args.loo_datasets == "none":
        selected_datasets = list(datasets.values())
    else:
        selected_datasets = [path for name, path in datasets.items() if name not in parsed_args.loo_datasets]

    with open(f'smac_hpo/params_{parsed_args.algorithm}.pcs', 'r') as f:
        configuration = pcs.read(f)

    output_path = Path(f"smac3_output/{parsed_args.algorithm}/loo_{parsed_args.loo_datasets}_seed_{parsed_args.seed}")
    print(f"Output path for SMAC results: {output_path}")
    # Scenario object specifying the optimization environment
    scenario = Scenario(
        configuration,
        objectives=["-adj_mutual_info", "-throughput"],  
        instances=selected_datasets,
        instance_features={name: [i] for i, name in enumerate(selected_datasets)},  # Map instance names to indices
        deterministic=False,
        walltime_limit=60 * 60 * 24 * 7,
        n_trials=50,
        seed=parsed_args.seed,
        crash_cost = [2.0, 2.0],  # Cost for crashing runs (e.g., due to low throughput)
        output_directory=output_path
    )

    # Use SMAC to find the best configuration/hyperparameters
    if parsed_args.algorithm == "inc_clust_ray":
        train_function = make_train_function_ray(max_samples=2_000)
    else:
        train_function = make_train_function(parsed_args.algorithm, max_samples=30_000, throughput_limit=2.0, log_after_n_points=1000)
    
    smac = MultiObjectiveFacade(scenario, train_function)  # Use multi-objective optimization if needed
    incumbents = smac.optimize()
    assert isinstance(incumbents, list), "Expected incumbents to be a list of configurations."

    print("Best found configuration:", incumbents)
    #Store the best configuration in a JSON file for later use
    config_name = f"best_config_loo_{parsed_args.loo_datasets}_seed_{parsed_args.seed}.json"
    best_config_path = f"smac3_best_configs/{parsed_args.algorithm}/{config_name}"
    os.makedirs(os.path.dirname(best_config_path), exist_ok=True)
    with open(best_config_path, 'w') as f:
        json.dump([dict(i.items()) for i in incumbents], f, indent=4)
    print(f"Best configuration saved to: {best_config_path}")
        
    print ("Optimization completed. Best configuration saved to:", best_config_path)
    print ("Start evaluation of the best configuration on full datasets...")
    
    results = []
    final_performance_path = Path(f"smac3_final_performance/{parsed_args.algorithm}/loo_{parsed_args.loo_datasets}_seed_{parsed_args.seed}.csv")
    final_performance_path.parent.mkdir(parents=True, exist_ok=True)
    
    for i, c in enumerate(incumbents):
        for d in datasets.values():
            try:
                print(f"Evaluating configuration on dataset: {d}")
                if parsed_args.algorithm == "inc_clust_ray":
                    train = make_train_function_ray()
                else:
                    train = make_train_function(parsed_args.algorithm)
                performance = train(c, seed=None, instance=d)[1]
                print("FINISHED EVALUATION:", d, config_name, i, parsed_args.loo_datasets, parsed_args.seed, parsed_args.algorithm, performance)
                performance["dataset"] = d
                performance["config"] = config_name
                performance["config_id"] = str(i)
                performance["loo_datasets"] = parsed_args.loo_datasets
                performance["seed"] = parsed_args.seed
                performance["algorithm"] = parsed_args.algorithm
                results.append(performance)
                pd.DataFrame(results).to_csv(final_performance_path, index=False)
            except Exception as e:
                print(f"Error evaluating configuration on dataset {d}: {e}")
                continue
            