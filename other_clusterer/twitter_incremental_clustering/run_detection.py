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


def normalize_tweet_ids(values):
    series = pd.Series(values)
    numeric_values = pd.to_numeric(series, errors="coerce")
    if numeric_values.notna().all():
        return numeric_values
    return series.astype(str)


def resolve_precomputed_embeddings_path(dataset_path: Path, model_name: str):
    normalized_model_name = normalize_embedding_name(model_name)
    dataset_names = [dataset_path.stem]
    if dataset_path.stem:
        dataset_names.append(dataset_path.stem[:1].upper() + dataset_path.stem[1:])

    candidate_roots = [
        dataset_path.parent / "pre_embedded_data",
        dataset_path.parent.parent / "pre_embedded_data",
        Path(__file__).resolve().parent.parent / "textClust_stream_data" / "pre_embedded_data",
    ]

    for root in candidate_roots:
        for dataset_name in dataset_names:
            embeddings_path = root / dataset_name / f"{normalized_model_name}.npz"
            if embeddings_path.exists():
                return embeddings_path

    searched_paths = [
        str(root / dataset_name / f"{normalized_model_name}.npz")
        for root in candidate_roots
        for dataset_name in dataset_names
    ]
    raise FileNotFoundError(
        f"Missing precomputed embeddings for model '{model_name}'. Searched: {searched_paths}"
    )


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
    frame["id"] = normalize_tweet_ids(frame["id"])
    frame["text"] = frame["text"].fillna("").astype(str)
    le = LabelEncoder()
    frame["label"] = le.fit_transform(frame["label"])
    frame["created_at"] = frame["created_at"].astype(str)
    # Robustly parse timestamps; fall back to a default when missing/unparsable.
    times = pd.to_datetime(frame["created_at"], errors="coerce", utc=False)
    if times.isna().all():
        logging.warning(
            "No valid 'created_at' timestamps found; defaulting to 1970-01-01 00:00:00"
        )
        times = times.fillna(pd.Timestamp("1970-01-01 00:00:00"))
    elif times.isna().any():
        logging.warning(
            "Some 'created_at' values could not be parsed; those will be set to 1970-01-01 00:00:00"
        )
        times = times.fillna(pd.Timestamp("1970-01-01 00:00:00"))
    frame["date"] = times.dt.strftime("%Y%m%d")
    frame["time"] = times.dt.strftime("%H:%M:%S")
    return frame.drop_duplicates("id").sort_values("id").reset_index(drop=True)


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
            frame["created_at"] = pd.to_datetime(frame["ts"], unit="ms", errors="coerce")
        elif "timestamp" in frame.columns:
            timestamp_values = pd.to_numeric(frame["timestamp"], errors="coerce")
            if timestamp_values.notna().any():
                timestamp_unit = "ms" if timestamp_values.abs().max() > 1e11 else "s"
                frame["created_at"] = pd.to_datetime(
                    timestamp_values, unit=timestamp_unit, errors="coerce"
                )
            else:
                frame["created_at"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        else:
            frame["created_at"] = pd.NaT

    if "label" not in frame.columns:
        frame["label"] = np.nan

    frame["id"] = normalize_tweet_ids(frame["id"])
    frame["text"] = frame["text"].fillna("").astype(str)
    le = LabelEncoder()
    frame["label"] = le.fit_transform(frame["label"])
    frame["created_at"] = frame["created_at"].astype(str)

    times = pd.to_datetime(frame["created_at"], errors="coerce", utc=False)
    if times.isna().all():
        logging.warning(
            "No valid 'created_at' timestamps found; defaulting to 1970-01-01 00:00:00"
        )
        times = times.fillna(pd.Timestamp("1970-01-01 00:00:00"))
    elif times.isna().any():
        logging.warning(
            "Some 'created_at' values could not be parsed; those will be set to 1970-01-01 00:00:00"
        )
        times = times.fillna(pd.Timestamp("1970-01-01 00:00:00"))
    frame["date"] = times.dt.strftime("%Y%m%d")
    frame["time"] = times.dt.strftime("%H:%M:%S")
    return frame.drop_duplicates("id").sort_values("id").reset_index(drop=True)


def load_precomputed_embeddings(embeddings_path: Path, tweet_ids=None):
    archive = np.load(embeddings_path, allow_pickle=True)
    if "embeddings" not in archive.files:
        raise KeyError(f"{embeddings_path} does not contain an 'embeddings' array")

    embeddings = archive["embeddings"]
    if tweet_ids is not None:
        if "tweet_ids" not in archive.files:
            raise KeyError(f"{embeddings_path} does not contain a 'tweet_ids' array")

        archive_ids = normalize_tweet_ids(archive["tweet_ids"])
        tweet_ids = normalize_tweet_ids(tweet_ids)
        if np.any(~np.isin(tweet_ids, archive_ids)):
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

    if embeddings.dtype == object:
        embeddings = np.vstack(embeddings)
    return np.asarray(embeddings)


def load_inputs(params):
    dataset_path = Path(params["dataset"])
    is_json_dataset = dataset_path.suffix.lower() in {".json", ".jsonl"}
    if params["model"] == "sbert":
        if dataset_path.suffix == ".npy":
            data = load_npy_dataset(dataset_path)
        elif is_json_dataset:
            data = load_json_dataset(dataset_path)
        else:
            data = load_dataset(str(dataset_path), "vectors", False)
            data["id"] = pd.to_numeric(data["id"], errors="raise")

        embeddings_path = resolve_precomputed_embeddings_path(dataset_path, params["sub_model"])
        vectors = load_precomputed_embeddings(embeddings_path, data["id"].to_numpy())
        return vectors, data

    if dataset_path.suffix == ".npy":        
        data = load_npy_dataset(dataset_path)
        embeddings_path = resolve_precomputed_embeddings_path(dataset_path, params["model"])
        vectors = load_precomputed_embeddings(embeddings_path, data["id"].to_numpy())
        
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
        
        return vectors, data

    if is_json_dataset:
        data = load_json_dataset(dataset_path)
        embeddings_path = resolve_precomputed_embeddings_path(dataset_path, params["model"])
        vectors = load_precomputed_embeddings(embeddings_path, data["id"].to_numpy())
        return vectors, data

    # For non-.npy datasets, prefer precomputed embeddings when available to
    # avoid building features (which can fail when dataset paths are files).
    try:
        embeddings_path = resolve_precomputed_embeddings_path(dataset_path, params["model"])
    except FileNotFoundError:
        return build_matrix(**params)

    # Load dataset and align embeddings
    data = load_dataset(str(dataset_path), "vectors", False)
    data["id"] = pd.to_numeric(data["id"], errors="raise")
    vectors = load_precomputed_embeddings(embeddings_path, data["id"].to_numpy())
    return vectors, data


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
    X, data = load_inputs(params)

    # todo: improve window computation
    params["window"] = int(
        data.groupby("date").size().mean()
        * params["window"]
        / 24
        // params["batch_size"]
        * params["batch_size"]
    )

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
    ami = adjusted_mutual_info_score(data.label, y_pred)
    ari = adjusted_rand_score(data.label, y_pred)
    p, r, f1 = cluster_event_match(data, y_pred)
    
    data["pred"] = y_pred
    if params["save_results"] == True:
    # Write detected labels to a csv file
        dataset_path = Path(params["dataset"])
        output_dir = Path(params["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"{dataset_path.stem}_clustering_results.tsv"
        logging.info("Write predicted labels to {}".format(filename))
        
        data[["id", "label", "pred"]].to_csv(
            filename, index=False, sep="\t", quoting=csv.QUOTE_ALL
        )

    # Write evaluation metrics to a csv file
    params.update(
        {
            "AMI": ami,
            "ARI": ari,
            "precision": p,
            "recall": r,
            "f1": f1,
            "seconds": (end_time - start_time).seconds,
        }
    )
    stats = pd.DataFrame(params, index=[0])
    stats = stats[
        [
            "dataset",
            "model",
            "sub_model",
            "lang",
            "AMI",
            "ARI",
            "precision",
            "recall",
            "f1",
            "seconds",
            "threshold",
            "window",
            "batch_size",
            "remove_mentions",
            "hashtag_split",
        ]
    ]
    print(stats[["sub_model", "threshold", "AMI", "ARI", "f1"]].iloc[0])
    if params["save_results"] == True:
        try:
            results = pd.read_csv(METRICS_FILE)
        except FileNotFoundError:
            results = pd.DataFrame()
        stats = pd.concat([results, stats], ignore_index=True)
        stats.to_csv(METRICS_FILE, index=False)
        logging.info("Saved results to {}".format(METRICS_FILE))
        
    return data.label, y_pred


if __name__ == "__main__":
    args = vars(parser.parse_args())
    run(args)
