from typing import Optional
import pandas as pd
import os
import numpy as np
import json
import scipy
import seaborn as sns
import matplotlib.pyplot as plt

COLOR_PALETTE = sns.color_palette("Paired", n_colors=3)

COLORS = {
    "FSD": COLOR_PALETTE[2],
    "embClust": COLOR_PALETTE[1],
    "textClust": COLOR_PALETTE[0],
}

ALGORITHM_ORDER = list(COLORS.keys())

# Physical figure dimensions matching the LaTeX layout (two-column paper).
CM = 1 / 2.54
SINGLE_COLUMN_WIDTH = 8.5 * CM
DOUBLE_COLUMN_WIDTH = 17.5 * CM
PANEL_HEIGHT = 6 * CM


def set_publication_style() -> None:
    """Set rcParams so figures remain readable when included in LaTeX at 100% size."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 1.5,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 120,
        "savefig.dpi": 300,
    })


def facet_aspect(n_panels: int, height: float = PANEL_HEIGHT, width: float = DOUBLE_COLUMN_WIDTH) -> float:
    """Return the seaborn FacetGrid aspect so that n_panels side by side span the given width."""
    return (width / n_panels) / height

OLD_DATASETS = [ "News-T", "SO-T", "Trends-T", "Tweets-T" ]
NEW_DATASETS = [ "Arabic_7", "Arabic_Twitter", "CrisisLexT6", "CrisisLexT7", "CrisisLexT26", "CrisisMMD", "CrisisNLP", "Event2012", "Event2018",
                 "HumAID", "KBP", "MAVEN", "Mix_Data", "Semantic5-T", "Semantic10-T", "Semantic40-T"]
COMBINED_DATASETS = [ "NT", "NTS", "NTSO-T" ]
ALL_DATASETS = OLD_DATASETS + NEW_DATASETS + COMBINED_DATASETS
    
NUM_SEEDS = range(1, 11)
NUM_REPS = int(scipy.special.binom(10, 2))


def load_optimization_performance_old_datasets(file_path: str = "results/textClust_hpo_results/best_on_old_datasets") -> pd.DataFrame:
    dfs = []
    if not os.path.isdir(file_path):
        return pd.DataFrame()
    for file in os.listdir(f"{file_path}"):
        algorithm = ""
        if "embClust" in file:
            algorithm = "embClust"
        elif "textClust" in file:
            algorithm = "textClust"
        elif "inc_clust_ray" in file:
            algorithm = "FSD"
        df = pd.read_csv(os.path.join(f"{file_path}", file))
        df["algorithm"] = algorithm
        if "dataset_name" in df.columns:
            df = df.drop(columns=["dataset"], errors="ignore")
            df = df.rename(columns={"dataset_name": "dataset"})
        df["config_id"] = f"{algorithm}_" + df["config"].astype(str)
        df["config"] = df["config"].str.replace(r"\.json.*", ".json", regex=True)
        df["config_source"] = "full"
        df["seed"] = df["config"].str.extract(r"seed_(\d+)\.json").astype(int)
        df["is_test"] = True
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def load_optimization_performance(file_path: str = "results/textClust_hpo_results/smac3_final_performance") -> pd.DataFrame:
    dfs = []

    for file in os.listdir(f"{file_path}/embClust"):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(f"{file_path}/embClust", file))
            dfs.append(df)

    for file in os.listdir(f"{file_path}/textClust"):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(f"{file_path}/textClust", file))
            dfs.append(df)
        
    for file in os.listdir(f"{file_path}/inc_clust_ray"):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(f"{file_path}/inc_clust_ray", file))
            dfs.append(df)
            
    result = pd.concat(dfs, ignore_index=True)

    # Rename the dataset column to remove the file destination and suffix from the dataset names.
    result["dataset"] = (
        result["dataset"]
        .str.replace("data/textClust_stream_data/raw_data/", "", regex=False)
        .str.replace(".npy", "", regex=False)
        .str.replace(".json", "", regex=False)
        .str.replace("data/raw_data/", "")
    )

    # Parse and expand strings such as "['Arabic_Twitter', 'Semantic40-T']" inline.
    result[["loo_dataset_1", "loo_dataset_2"]] = (
        result["loo_datasets"]
        .apply(lambda value: value.replace("['", "").replace("']", "").split(" ") if value.startswith('[') else (None,None))
        .apply(pd.Series)
        )
    # Optional: mark rows whose evaluated dataset is one of the held-out datasets.
    result["is_test"] = result.apply(
        lambda row: row["dataset"] in {
            row["loo_dataset_1"],
            row["loo_dataset_2"],
        },
        axis=1,
    )

    assert len(result) != 0, "No rows were loaded from the CSV files. Please check the file paths and contents."
    assert result["is_test"].sum() != 0, "No rows were marked as test rows. Please check the dataset names and the parsing of the 'loo_datasets' column."

    # Rename inc_clust_ray to FSD 
    result["algorithm"] = result["algorithm"].replace({"inc_clust_ray": "FSD"})
    
    # Specify if configuration was optained during CV or full optimization
    result["config_source"] = "cv"
    result.loc[result["loo_datasets"] == "none", "config_source"] = "full"
    
    assert (result["config_source"]=="cv").any(), "No rows were marked as 'cv' in the 'config_source' column. Please check the 'loo_datasets' values."
    assert (result["config_source"]=="full").any(), "No rows were marked as 'full' in the 'config_source' column. Please check the 'loo_datasets' values."
    
    # Specify config_id
    result = result.rename(columns={"config_id": "config_index"})
    result["config_id"] = result.apply(
        lambda row: f"{row['algorithm']}_{row['config']}[{row['config_index']}]",
        axis=1
    )

    return result

def load_optimization_configs(file_path: str = "results/textClust_hpo_results/smac3_best_configs") -> pd.DataFrame:
    """
    Load the optimization configurations from the specified file path.

    Args:
        file_path (str): The path to the directory containing the optimization configuration files.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded optimization configurations.
    """
  
    def load_configurations(algorithm):
        config_dir = os.path.join(file_path, algorithm)
        results = []
        display_algorithm = "FSD" if algorithm == "inc_clust_ray" else algorithm

        for file in os.listdir(config_dir):
            if not file.endswith(".json"):
                continue

            config_path = os.path.join(config_dir, file)
            with open(config_path, "r") as config_file:
                config = pd.DataFrame(json.load(config_file))

            config["algorithm"] = display_algorithm
            config = config.reset_index(names=["config_index"])
            config["config"] = f"{file}"
            config["config_id"] = config.apply(
                lambda row: f"{row['algorithm']}_{row['config']}[{row['config_index']}]",
                axis=1
            )
            results.append(config)

        return pd.concat(results, ignore_index=True)

    result = pd.concat([
        load_configurations("embClust"),
        load_configurations("textClust"),
        load_configurations("inc_clust_ray")
    ], ignore_index=True)

    assert len(result) != 0, "No configurations were loaded from the JSON files. Please check the file paths and contents."

    return result


def load_default_performance_and_config(file_path_performance: str = "results/textClust_hpo_results/default_on_all_datasets", 
                                        file_path_config: str = "smac_hpo/defaults") -> pd.DataFrame:
    results = []
    for file in os.listdir(f"{file_path_config}"):
        df = pd.read_json(os.path.join(f"{file_path_config}", file), lines=False)
        algorithm = ""
        if "embClust" in file:
            algorithm = "embClust"
        elif "textClust" in file:
            algorithm = "textClust"
        elif "inc_clust_ray" in file:
            algorithm = "FSD"
        df["config_id"] = f"{algorithm}_{file}"
        df["algorithm"] = algorithm
        df["config"] = file
        performance = pd.read_csv(os.path.join(f"{file_path_performance}", f"smac_hpo_defaults_{file}_results.csv"))
        performance["config_id"] = f"{algorithm}_{file}"
        performance["algorithm"] = algorithm
        performance["config"] = file
        performance = performance.drop(columns=["dataset"])
        performance = performance.rename(columns={"dataset_name": "dataset"})
        df = df.merge(performance, on=["config_id", "algorithm", "config"], how="outer", indicator=True)
        assert df["_merge"].value_counts().get("both", 0) == len(performance), "Not all performance results have a matching configuration."
        df = df.drop(columns=["_merge"])
        df["is_test"] = True
        results.append(df)

    result = pd.concat(results, ignore_index=True)
    result["config_source"] = "default"
    return result



def build_results_csv(file_path_performance: str = "results/textClust_hpo_results/smac3_final_performance",
                      file_path_config: str = "results/textClust_hpo_results/smac3_best_configs",
                      file_path_performance_old: str = "results/textClust_hpo_results/best_on_old_datasets",
                      file_path_performance_default: str = "results/textClust_hpo_results/default_on_all_datasets",
                      file_path_config_default: str = "smac_hpo/defaults") -> pd.DataFrame:
    """
    Build a results CSV from the given path and store it in the specified location.

    Args:
        file_path_performance (str): The path to the directory containing the performance results.
        file_path_config (str): The path to the directory containing the configuration files.
        file_path_performance_old (str): The path to the directory containing the performance results on old datasets.
        file_path_performance_default (str): The path to the directory containing the default performance results.
        file_path_config_default (str): The path to the directory containing the default configuration files.

    Returns:
        pd.DataFrame: A DataFrame containing the combined results.
    """
    # Initialize an empty list to hold DataFrames
    
    df_performance = load_optimization_performance(file_path_performance)
    df_performance_on_old = load_optimization_performance_old_datasets(file_path_performance_old)
    performances = pd.concat([df_performance, df_performance_on_old], ignore_index=True)
    configs = load_optimization_configs(file_path_config)
    
    result = performances.merge(configs, how="outer", indicator=True)

    assert result["_merge"].value_counts().get("both", 0) == len(result), "Not all evaluation results have a matching configuration."

    result = result.drop(columns=["_merge"])
    
    default_performance_and_config = load_default_performance_and_config(file_path_performance_default, file_path_config_default)
    
    result = pd.concat([result, default_performance_and_config], ignore_index=True)
    
    assert result["config_id"].isna().sum() == 0, "Some rows have missing config_id values. Please check the merging of performance and configuration DataFrames."
    
    # Sort columns for better readability
    important_columns = ["config_source", "config_id", "algorithm", "config", "config_index", "dataset", "is_test"]
    column_order = important_columns + [col for col in result.columns if col not in important_columns]
    result = result[column_order]
    
    return result

def store_results_csv(df: pd.DataFrame, store_path: str = "results/textClust_hpo_results/full_merged_results.csv") -> None:
    """
    Store the given DataFrame as a CSV file at the specified path.

    Args:
        df (pd.DataFrame): The DataFrame to be stored.
        store_path (str): The path where the CSV file should be stored.
    """
    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    df.to_csv(store_path, index=False)

def load_results_csv(file_path: str = "results/textClust_hpo_results/full_merged_results.csv") -> pd.DataFrame:
    """
    Load a results CSV from the specified file path.

    Args:
        file_path (str): The path to the results CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded results.
    """
    return pd.read_csv(file_path)

def pareto_optimal_points(
    data: pd.DataFrame,
    objective_columns: list[str],
    maximize: bool | list[bool] = False,
) -> tuple[pd.Index, pd.DataFrame]:
    """Return the indices and objective values of Pareto-optimal rows.

    Parameters
    ----------
    data:
        DataFrame containing the objective columns.
    objective_columns:
        Column names used as objectives.
    maximize:
        Either one boolean applied to every objective or one boolean per
        objective. ``True`` means maximize that objective; ``False`` means
        minimize it. The default minimizes all objectives.

    Returns
    -------
    tuple[pd.Index, pd.DataFrame]
        The original DataFrame indices and a DataFrame containing the
        corresponding objective values.
    """
    if not objective_columns:
        raise ValueError("objective_columns must contain at least one column")

    missing_columns = [
        column for column in objective_columns if column not in data.columns
    ]
    if missing_columns:
        raise KeyError(f"Objective columns not found: {missing_columns}")

    if isinstance(maximize, bool):
        maximize = [maximize] * len(objective_columns)
    else:
        maximize = list(maximize)
        if len(maximize) != len(objective_columns):
            raise ValueError(
                "maximize must be a bool or have one value per objective column"
            )
        if not all(isinstance(value, bool) for value in maximize):
            raise TypeError("maximize must contain only booleans")

    objectives = data.loc[:, objective_columns]
    if objectives.isna().any().any():
        raise ValueError("Objective values must not contain missing values")

    # Convert all objectives to maximization objectives. A row dominates
    # another when it is at least as good in every objective and better in one.
    values = objectives.to_numpy(dtype=float)
    transformed = values.copy()
    for column_index, should_maximize in enumerate(maximize):
        if not should_maximize:
            transformed[:, column_index] *= -1

    dominates = (
        (transformed[:, None, :] >= transformed[None, :, :]).all(axis=2)
        & (transformed[:, None, :] > transformed[None, :, :]).any(axis=2)
    )
    pareto_mask = ~dominates.any(axis=0)

    pareto_indices = data.index[pareto_mask]
    pareto_objectives = objectives.loc[pareto_indices]
    return pareto_indices, pareto_objectives

def normalize(df: pd.DataFrame, columns: list[str], use_log: list[bool] | bool = False) -> pd.DataFrame:
    """Normalize the specified columns of a DataFrame to the range [0, 1].

    Parameters
    ----------
    df:
        DataFrame containing the columns to normalize.
    columns:
        List of column names to normalize.
    use_log:
        Either one boolean applied to every column or one boolean per
        column. ``True`` means apply log transformation before normalization.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with the specified columns normalized to the range [0, 1].    
    """
    if isinstance(use_log, bool):
        use_log = [use_log] * len(columns)
    else:
        use_log = list(use_log)
        if len(use_log) != len(columns):
            raise ValueError(
                "use_log must be a bool or have one value per column"
            )
        if not all(isinstance(value, bool) for value in use_log):
            raise TypeError("use_log must contain only booleans")

    df_normalized = df.copy()
    for column, log_transform in zip(columns, use_log):
        if log_transform:
            df_normalized[column] = np.log1p(df_normalized[column])
        min_value = 0.0
        max_value = df_normalized[column].max()
        df_normalized[column] = (df_normalized[column] - min_value) / (max_value - min_value)

    return df_normalized

if __name__ == "__main__":
    df = build_results_csv()
    store_results_csv(df)