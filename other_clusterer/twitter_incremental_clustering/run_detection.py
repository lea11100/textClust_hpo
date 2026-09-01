from twembeddings import build_matrix, ClusteringAlgo, cluster_event_match, load_dataset

import csv
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics.cluster import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder

from utils import METRICS_FILE

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument(
    "--model",
    nargs="+",
    required=True,
    help="One or several text embeddings",
)
parser.add_argument(
    "--dataset",
    required=True,
    help="Path to the dataset",
)

parser.add_argument(
    "--output-dir",
    required=False,
    default=".",
    help="Directory where clustering results will be written",
)

parser.add_argument("--lang", required=True, choices=["en", "fr"])

parser.add_argument(
    "--threshold",
    type=float,
)

parser.add_argument("--window", required=False, type=int)

parser.add_argument("--batch-size", required=False, type=int)

parser.add_argument(
    "--sub-model",
    required=False,
    type=str,
    default="",
    help="The name of HuggingFace sentence-BERT model",
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s : %(message)s", level=logging.INFO
)


NPY_COLUMNS = [
    "id",
    "text",
    "label",
    "words",
    "filtered_words",
    "entities",
    "user_id",
    "created_at",
    "url",
    "hashtags",
    "urls",
]


def find_date_created_at(created_at):
    created_at = str(created_at)
    if "+0000" in created_at:
        date = datetime.strptime(created_at, "%a %b %d %H:%M:%S +0000 %Y")
    else:
        date = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
    return date.strftime("%Y%m%d"), date.strftime("%H:%M:%S")


def normalize_embedding_name(name: str):
    return name.replace("/", "_").replace("-", "_")


def resolve_precomputed_embeddings_path(dataset_path: Path, model_name: str):
    normalized_model_name = normalize_embedding_name(model_name)
    dataset_name = dataset_path.stem

    candidate_roots = [
        dataset_path.parent / "embedded",
        dataset_path.parent.parent / "embedded",
    ]

    searched_paths = []
    for root in candidate_roots:
        embeddings_path = root / normalized_model_name / f"{dataset_name}.npy"
        if embeddings_path.exists():
            return embeddings_path
        searched_paths.append(str(embeddings_path))

    raise FileNotFoundError(
        f"Missing precomputed embeddings for model '{model_name}'. Searched: {searched_paths}"
    )

def prepare_dataframe(data: pd.DataFrame):
    for col in ["id", "text", "label", "created_at"]:
        assert col in data.columns, f"DataFrame must contain a '{col}' column"
        
    data["id"] = data["id"].astype(str)
    
    data["text"] = data["text"].fillna("").astype(str)
    le = LabelEncoder()
    data["label"] = np.array(le.fit_transform(data["label"]))
    
    data["created_at"] = data["created_at"].astype(str)
    data["created_at"] = pd.to_datetime(data["created_at"], errors="coerce", utc=False)
    if data["created_at"].isna().any():
        logging.warning(
            f"{data['created_at'].isna().sum()} 'created_at' values could not be parsed; those will be set to 1970-01-01 00:00:00"
        )
        data["created_at"] = data["created_at"].fillna(pd.Timestamp("1970-01-01 00:00:00"))
    
    return data.drop_duplicates("id", keep=False).sort_values("created_at").reset_index(drop=True)

def load_npy_dataset(dataset_path: Path):
    data = np.load(dataset_path, allow_pickle=True)
    if data.ndim != 2 or data.shape[1] < len(NPY_COLUMNS):
        raise ValueError(
            f"Expected a 2D numpy array with at least {len(NPY_COLUMNS)} columns, got {data.shape}"
        )

    columns = list(NPY_COLUMNS)
    if data.shape[1] > len(columns):
        columns.extend([f"extra_{idx}" for idx in range(data.shape[1] - len(columns))])

    frame = pd.DataFrame(data, columns=columns)
    return prepare_dataframe(frame)


def load_json_dataset(dataset_path: Path):
    frame = pd.read_json(dataset_path, lines=True)

    rename_map = {
        "tweetID": "id",
        "tweets": "text",
        "clusterNo": "label",
        "topic": "label",
    }
    frame = frame.rename(columns=rename_map)

    if "created_at" not in frame.columns:
        if "ts" in frame.columns:
            frame["created_at"] = frame["ts"]
        elif "timestamp" in frame.columns:
            frame["created_at"] = frame["ts"]
        else:
            raise ValueError(f"No 'created_at', 'ts', or 'timestamp' column found in the dataset {dataset_path}")

    return prepare_dataframe(frame)


def load_precomputed_embeddings(embeddings_path: Path, tweet_ids=None):
    archive = np.load(embeddings_path, allow_pickle=True).item()
    if "embeddings" not in archive.keys():
        raise KeyError(f"{embeddings_path} does not contain an 'embeddings' array")
    if "times" not in archive.keys():
        raise KeyError(f"{embeddings_path} does not contain a 'times' array")

    embeddings = archive["embeddings"]
    times = archive["times"]

    if tweet_ids is not None:
        if "tweet_ids" not in archive.keys():
            raise KeyError(f"{embeddings_path} does not contain a 'tweet_ids' array")

        archive_ids = archive["tweet_ids"]
        if set(tweet_ids) != set(archive_ids):
            raise ValueError(
                f"Embedding file {embeddings_path} has {len(archive_ids)} tweet ids, "
                f"but the dataset contains {len(tweet_ids)} rows"
            )

        archive_ids_array = np.asarray(archive_ids)
        tweet_ids_array = np.asarray(tweet_ids)
        if not np.array_equal(archive_ids_array, tweet_ids_array):
            positions = {tweet_id: idx for idx, tweet_id in enumerate(archive_ids_array)}
            try:
                order = [positions[tweet_id] for tweet_id in tweet_ids_array]
            except KeyError as exc:
                raise KeyError(
                    f"Tweet id {exc.args[0]} is missing in {embeddings_path}"
                ) from exc
            embeddings = embeddings[order]
            times = times[order]

    if embeddings.dtype == object:
        embeddings = np.vstack(embeddings)
    return np.asarray(embeddings), np.asarray(times)


def load_inputs(params):
    dataset_path = Path(params["dataset"])
    if dataset_path.suffix == ".npy":        
        data = load_npy_dataset(dataset_path)    
    elif dataset_path.suffix.lower() in {".json", ".jsonl"}:
        data = load_json_dataset(dataset_path)
    else:
        raise ValueError(f"Unsupported dataset file extension: {dataset_path.suffix}")
      
    embeddings_path = resolve_precomputed_embeddings_path(dataset_path, params["model"])
    vectors, times = load_precomputed_embeddings(embeddings_path, data["id"].to_numpy())
    
    
    seed = params.get("seed", None)
    max_samples = params.get("max_samples", None)
    if seed is not None:
        np.random.seed(seed)
        start = np.random.randint(0, len(data))
    else:
        start = 0
    if max_samples is not None:
        end = min(start + max_samples, len(data))
    else:
        end = len(data)
    data = data.iloc[start:end].reset_index(drop=True)
    vectors = vectors[start:end]
    times = times[start:end]
    return vectors, times, data

    


def run(args: dict):
    options_path = Path(__file__).resolve().with_name("options.yaml")
    with options_path.open("r") as f:
        options = yaml.safe_load(f)
    assert len(args["model"]) == 1, "Only one model can be specified at a time for incremental clustering"
    model = args["model"][0]
    start_time = datetime.now()
    # load standard parameters
    params = dict(options["standard"])
    if model in options:
        # change standard parameters for this specific model
        for opt in options[model]:
            params[opt] = options[model][opt]
    for arg in args:
        if args[arg] is not None:
            # params from command line overwrite options.yaml file
            params[arg] = args[arg]

    params.setdefault("remove_mentions", False)
    params["model"] = model

    # Create document embeddings
    X, times, data = load_inputs(params)

    # todo: improve window computation # LMS: I removed this because I want to optimize data independent
    # params["window"] = int(
    #     data.groupby("date").size().mean()
    #     * params["window"]
    #     / 24
    #     // params["batch_size"]
    #     * params["batch_size"]
    # )
    params["window"] = int(params["window"] * params["batch_size"])

    clustering = ClusteringAlgo(
        threshold=float(params["threshold"]),
        window_size=params["window"],
        batch_size=params["batch_size"],
    )
    if X is None:
        raise ValueError(
            f"No embeddings were loaded for dataset '{params['dataset']}' and model '{model}'"
        )
    clustering.add_vectors(X.copy())

    # Run clustering algorithm
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        y_pred = clustering.incremental_clustering()
    end_time = datetime.now()

    # Run evaluation
    #ami = adjusted_mutual_info_score(data.label, y_pred)
    #ari = adjusted_rand_score(data.label, y_pred)
    #p, r, f1 = cluster_event_match(data, y_pred)
    
    data["pred"] = y_pred
    # if params["save_results"] == True:
    # # Write detected labels to a csv file
    #     dataset_path = Path(params["dataset"])
    #     output_dir = Path(params["output_dir"])
    #     output_dir.mkdir(parents=True, exist_ok=True)
    #     filename = output_dir / f"{dataset_path.stem}_clustering_results.tsv"
    #     logging.info("Write predicted labels to {}".format(filename))
        
    #     data[["id", "label", "pred"]].to_csv(
    #         filename, index=False, sep="\t", quoting=csv.QUOTE_ALL
    #     )
    time_taken = (end_time - start_time).seconds
    embedding_time = times.sum()
    # Write evaluation metrics to a csv file
    params.update(
        {
            # "AMI": ami,
            # "ARI": ari,
            # "precision": p,
            # "recall": r,
            # "f1": f1,
            "seconds": (end_time - start_time).seconds,
            "time_taken_s": str(time_taken),
            "embedding_time_s": embedding_time,
            "total_time_s": time_taken + embedding_time,
        }
    )
    stats = pd.DataFrame(params, index=[0])
    stats = stats[
        [
            "dataset",
            "model",
            "sub_model",
            "lang",
            #"AMI",
            #"ARI",
            #"precision",
            #"recall",
            #"f1",
            "seconds",
            "threshold",
            "window",
            "batch_size",
            "remove_mentions",
            "hashtag_split",
            "embedding_time_s",
        ]
    ]
    #print(stats[["sub_model", "threshold", "AMI", "ARI", "f1"]].iloc[0])
    # if params["save_results"] == True:
    #     try:
    #         results = pd.read_csv(METRICS_FILE)
    #     except FileNotFoundError:
    #         results = pd.DataFrame()
    #     stats = pd.concat([results, stats], ignore_index=True)
    #     stats.to_csv(METRICS_FILE, index=False)
    #     logging.info("Saved results to {}".format(METRICS_FILE))
        
    return data.label, y_pred, stats


if __name__ == "__main__":
    args = vars(parser.parse_args())
    run(args)
