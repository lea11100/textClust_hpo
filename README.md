# textClust_hpo

This repository contains the code for hyperparameter optimization (HPO) of
stream clustering algorithms for short-text event detection. It optimizes the
configurations of **textClust**, **embClust** and an incremental clustering
baseline (**FSD**, implemented with Ray) using
[SMAC3](https://github.com/automl/SMAC3) in a multi-objective setting, and
evaluates the resulting configurations on a collection of social media
datasets (leave-one-dataset-out and default configurations).

## Installation

### Prerequisites

- Python 3.11 or newer
- Git (the clustering libraries are included as git submodules)

### 1. Clone the repository with its submodules

```bash
git clone --recursive https://github.com/lea11100/textClust_hpo.git
cd textClust_hpo
```

If you already cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

The submodules are:

- `ext_lib/textClust` (textClust, git@github.com:lea11100/textClust.git)
- `ext_lib/mosmac` (SMAC3 fork, https://github.com/jeroenrook/SMAC3, branch `mosmac-merge`)

### 2. Install the dependencies

Run the installation script, which creates a virtual environment in
`.textClust_hpo_venv`, installs `requirements.txt`, installs the two local
packages in editable mode and tries to install `pygmo` (see [Pygmo Installation Guide](https://esa.github.io/pygmo2/install.html) for installing pygmo from source):

```bash
bash install.sh
source .textClust_hpo_venv/bin/activate
```

Alternatively, install the steps manually:

```bash
python -m venv .textClust_hpo_venv
source .textClust_hpo_venv/bin/activate
pip install -r requirements.txt
pip install -e ./ext_lib/textClust
pip install -e ./ext_lib/mosmac
pip install pygmo
```

On HPC systems that use Environment Modules, `install.sh` loads
`lang/Python/3.11.3-GCCcore-12.3.0` if the `module` command is available.

### Container (optional)

`environment.def` is an Apptainer/Singularity definition that builds a Python
3.11 image with all dependencies, the local packages, `pygmo` and the NLTK
`stopwords` corpus:

```bash
apptainer build python311.sif environment.def
```

The SLURM scripts in `smac_hpo/jobs/` expect this image as `python311.sif` in
the repository root.

### Data

The datasets are not included in this repository. They are available from the
University of Paderborn Sciebo share:

- Web: https://uni-paderborn.sciebo.de/s/LkxwYPFtiRkTBC6

Place them in `data/textClust_stream_data/` with the following layout:

```text
data/textClust_stream_data/
├── raw_data/              # <dataset>.npy or <dataset>.json
├── translated_no_emojis/  # <dataset>.npy
└── embedded/              # <model_name>/<dataset>.npy
```

`data/textClust_stream_data/check_consistency.py`
checks that the raw, translated and embedded datasets are consistent:

```bash
python data/textClust_stream_data/check_consistency.py --details
```

## Usage

Run the commands from the repository root.

### Optimize a configuration

`smac_hpo/optimize.py` runs SMAC3 for one algorithm and one random seed:

```bash
python smac_hpo/optimize.py --algorithm textClust --seed 1 --n_trials 250 --max_samples 30000
```

| Argument | Default | Description |
| --- | --- | --- |
| `--algorithm` | `embClust` | `embClust`, `textClust` or `inc_clust_ray` |
| `--seed` | `1` | Random seed for SMAC3 |
| `--loo_datasets` | `none` | Dataset(s) to leave out of optimization and use for evaluation |
| `--n_trials` | `250` | Number of optimization trials |
| `--max_samples` | `30000` | Maximum number of samples processed per run |

### Evaluate a configuration

`smac_hpo/eval_any_config.py` evaluates configurations from a JSON file (a list
of configuration objects) and writes the results to a CSV file:

```bash
python smac_hpo/eval_any_config.py path/to/configs.json textClust results/eval.csv --dataset Arabic_7 KBP
```

If `--dataset` is omitted, all datasets are evaluated.

### Run on a SLURM cluster

The scripts in `smac_hpo/jobs/` are SLURM job arrays that run the experiments
inside the Apptainer image:

- `run.sh`: optimization with leave-one-dataset-out splits
- `run_full.sh`: full optimization over the seeds and algorithms
- `eval_default_configurations.sh`: evaluation of the default configurations
- `eval_best_on_old_datasets.sh`: evaluation of the best configurations on the older datasets

Adjust the account, partition, mail address and paths in the scripts before use.

## Repository contents

| Path | Content |
| --- | --- |
| `smac_hpo/optimize.py` | SMAC3 multi-objective optimization for textClust, embClust and FSD |
| `smac_hpo/eval_any_config.py` | Evaluation of arbitrary configurations on the datasets |
| `smac_hpo/config_spaces/` | Configuration spaces (`.pcs`) for each algorithm |
| `smac_hpo/defaults/` | Default and river default configurations for each algorithm |
| `smac_hpo/jobs/` | SLURM job scripts for the experiments |
| `smac_hpo/evaluation/` | Notebooks and helper modules for the analysis (comparison, config analysis, generalization, statistics, trade-off configurations), `utils.py` with plotting and loading helpers, and `evolutionary_clustering.py` for joint decision and objective space clustering |
| `smac_hpo/eval.ipynb` | Notebook for evaluating results |
| `configs/` | `all_configs.csv` (all evaluated configurations) and `best_configs.csv` (best configurations found by using all datasets) |
| `results/textClust_hpo_results/` | Optimization outputs (SMAC3 runs and best configurations), evaluations on the old and all datasets, and final performance tables |
| `model_selection/` | Embedding model selection: MTEB leaderboard, results (`.jsonl`) and notebook |
| `other_clusterer/twitter_incremental_clustering/` | FSD-SBERT baseline for event detection on Twitter |
| `figures/` | Figures used in the paper (PDF) |
| `latex/` | LaTeX table of the best configurations |
| `ext_lib/textClust` | textClust submodule |
| `ext_lib/mosmac` | SMAC3 fork (MOSMAC) submodule |
| `environment.def` | Apptainer definition of the runtime environment |
| `install.sh` | Setup script for the virtual environment |
| `requirements.txt` | Pinned Python dependencies |

## License

See [LICENSE](LICENSE).
