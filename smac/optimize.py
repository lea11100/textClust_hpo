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

import pandas as pd


from textClust.history.manager import HistoryManager
from textClust.offline.cluster import Macrocluster

# ROOT = Path(__file__).resolve().parent  # directory of this file
# sys.path.append(str(ROOT / "../ext_lib/mosmac"))

from ConfigSpace import Configuration, ConfigurationSpace
from sentence_transformers import SentenceTransformer
from smac.facade import MultiObjectiveFacade
from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace.read_and_write import pcs_new as pcs

from textClust.algorithms.presets import get_embClust, get_fastLogClust
from textClust.data.wrapper import StaticJsonWrapper, SocialEDWrapper
from textClust.utils.eval import eval_against_truth
from textClust.offline.clusterer import SimpleMacroClusterer
from textClust.history.manager import HistoryManager
from textClust.online.clusterer import Microclusterer
from textClust.preprocessing.preprocessor import SimulatedEmbeddingPreprocessor

# datasets = {
#     "Semantic10-T": "examples/data/Semantic10-T.json",
#     "Semantic40-T": "examples/data/Semantic40-T.json",
#     "Semantic5-T": "examples/data/Semantic5-T.json",
#     "NT": "examples/data/NT.json",
#     "NTS": "examples/data/NTS.json",
#     "NTSO": "examples/data/NTSO-T.json",
#     "SO-T": "examples/data/SO-T.json",
#     "News-T": "examples/data/News-T.json",
#     "Trends-T": "examples/data/Trends-T.json",
#     "Tweets-T": "examples/data/Tweets-T.json",
# }

datasets = {
    "Arabic_7" : "data/social_ed_data/Arabic_7.npy",
    "Arabic_Twitter" : "data/social_ed_data/Arabic_Twitter.npy",
    "CrisisLexT6" : "data/social_ed_data/CrisisLexT6.npy",
    "CrisisLexT7" : "data/social_ed_data/CrisisLexT7.npy",
    "CrisisLexT26" : "data/social_ed_data/CrisisLexT26.npy",
    "CrisisMMD" : "data/social_ed_data/CrisisMMD.npy",
    "CrisisNLP" : "data/social_ed_data/CrisisNLP.npy",
    "Event2012" : "data/social_ed_data/Event2012.npy",
    "Event2018" : "data/social_ed_data/Event2018.npy",
    "HumAID" : "data/social_ed_data/HumAID.npy",
    "KBP" : "data/social_ed_data/KBP.npy",
    "MAVEN" : "data/social_ed_data/MAVEN.npy",
    "Mix_Data" : "data/social_ed_data/Mix_Data.npy",
    "Semantic5-T" : "data/Semantic5-T.json",
    "Semantic10-T" : "data/Semantic10-T.json",
    "Semantic40-T" : "data/Semantic40-T.json",
}

# embedding_model_name = 'ibm-granite/granite-embedding-107m-multilingual'
# embedding_model = SentenceTransformer(embedding_model_name, device="cpu")

models = {
    "NeuML": SentenceTransformer('NeuML/pubmedbert-base-embeddings-2M', device="cpu"),
    "minishlab_2M": SentenceTransformer('minishlab/potion-base-2M', device="cpu"),
    "minishlab_4M": SentenceTransformer('minishlab/potion-base-4M', device="cpu"),
    "minishlab_8M": SentenceTransformer('minishlab/potion-base-8M', device="cpu"),
    "minishlab_32M": SentenceTransformer('minishlab/potion-retrieval-32M', device="cpu"),
    "stephantulkens": SentenceTransformer('stephantulkens/NIFE-mxbai-embed-large-v1_as_router', device="cpu"),
    "Mihaiii_Bulbasaur": SentenceTransformer('Mihaiii/Bulbasaur', device="cpu"),
    "Mihaiii_gte_micro_v4": SentenceTransformer('Mihaiii/gte-micro-v4', device="cpu"),
    "sentence_transformers_all_MiniLM_L6_v2": SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device="cpu"),
    "Snowflake": SentenceTransformer('Snowflake/snowflake-arctic-embed-xs', device="cpu"),
    "MongoDB": SentenceTransformer('MongoDB/mdbr-leaf-ir', device="cpu"),
    "intfloat": SentenceTransformer('intfloat/e5-base-v2', device="cpu"),
    "Alibaba_NLP": SentenceTransformer('Alibaba-NLP/gte-modernbert-base', device="cpu"),
    "microsoft": SentenceTransformer('microsoft/harrier-oss-v1-270m', device="cpu"),
    }

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("algorithm.start() exceeded time limit")

TIME_LIMIT = int((30_000 + 500) / 2)  # Maximum time limit in seconds (30,000 points at 2 points/sec + 500 seconds buffer)

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
            embedding_model = models.get(embedding_model_name, models["sentenceTransformers"])
            embClust = get_embClust(data_loader=data_loader,
                                embedding_model=embedding_model,
                                log_interval=log_after_n_points,
                                **config)
            embClust.preprocessor = SimulatedEmbeddingPreprocessor(embedding_model_name, instance)
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
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TIME_LIMIT)
        start_time = time.time()
        try:
            algorithm.start()
        finally:
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
    

# Add args for seed and instance selection if needed
args = argparse.ArgumentParser()
args.add_argument("--seed", type=int, default=1, help="Random seed for SMAC (default: 1).")
args.add_argument("--loo_datasets", type=list, default="none", help="Dataset to leave out for evaluation (default: 'none').")
args.add_argument("--algorithm", type=str, default="embClust", choices=["embClust", "textClust"], help="Algorithm to optimize (default: 'embClust').")
    
if __name__ == "__main__":
    parsed_args = args.parse_args()

    if parsed_args.loo_datasets == "none":
        selected_datasets = list(datasets.values())
    else:
        selected_datasets = [path for name, path in datasets.items() if name not in parsed_args.loo_datasets]

    with open(f'experiments/smac/params_{parsed_args.algorithm}.pcs', 'r') as f:
        configuration = pcs.read(f)

    output_path = Path(f"smac3_output/{parsed_args.algorithm}/loo_{parsed_args.loo_datasets}_seed_{parsed_args.seed}")
    # Scenario object specifying the optimization environment
    scenario = Scenario(
        configuration,
        objectives=["-adj_mutual_info", "-throughput"],  
        instances=selected_datasets,
        instance_features={name: [i] for i, name in enumerate(selected_datasets)},  # Map instance names to indices
        deterministic=False,
        walltime_limit=60 * 60 * 24 * 7,
        n_trials=250,
        seed=parsed_args.seed,
        crash_cost = [2.0, 2.0],  # Cost for crashing runs (e.g., due to low throughput)
        output_directory=output_path
    )

    # Use SMAC to find the best configuration/hyperparameters
    smac = MultiObjectiveFacade(scenario, make_train_function(parsed_args.algorithm))  # Use multi-objective optimization if needed
    incumbents = smac.optimize()
    assert isinstance(incumbents, list), "Expected incumbents to be a list of configurations."

    print("Best found configuration:", incumbents)
    
    #Store the best configuration in a JSON file for later use
    config_name = f"best_config_loo_{parsed_args.loo_datasets}_seed_{parsed_args.seed}.json"
    best_config_path = f"smac3_best_configs/{parsed_args.algorithm}/{config_name}"
    os.makedirs(os.path.dirname(best_config_path), exist_ok=True)
    with open(best_config_path, 'w') as f:
        json.dump([dict(i.items()) for i in incumbents], f, indent=4)
        
    print ("Optimization completed. Best configuration saved to:", best_config_path)
    print ("Start evaluation of the best configuration on full datasets...")
    
    results = []
    
    for i, c in enumerate(incumbents):
        for d in datasets.values():
            print(f"Evaluating configuration on dataset: {d}")
            train = make_train_function(parsed_args.algorithm, max_samples=None, throughput_limit=0.0, log_after_n_points=1000)
            performance = train(c, seed=None, instance=d)[1]
            print("FINISHED EVALUATION:", d, config_name, i, parsed_args.loo_datasets, parsed_args.seed, parsed_args.algorithm, performance)
            performance["dataset"] = d
            performance["config"] = config_name
            performance["config_id"] = str(i)
            performance["loo_dataset"] = parsed_args.loo_datasets
            performance["seed"] = parsed_args.seed
            performance["algorithm"] = parsed_args.algorithm
            results.append(performance)
            
    df = pd.DataFrame(results)
    df.to_csv(f"smac3_final_performance/{parsed_args.algorithm}/loo_{parsed_args.loo_datasets}_seed_{parsed_args.seed}.csv", index=False)