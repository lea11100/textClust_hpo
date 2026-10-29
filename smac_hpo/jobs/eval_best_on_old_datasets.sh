#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sussekl@mail.uni-paderborn.de
#SBATCH --job-name=text_clust_hpo
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --account=hpc-prf-dedsm
#SBATCH --output=noctua_logs/%j_%a.out
#SBATCH --array=0-29

module load tools/Apptainer/1.1.9-GCCcore-11.3.0
DATA_DIR=$(readlink -f data/textClust_stream_data)

ALGORITHMS=(
    "embClust"
    "inc_clust_ray"
    "textClust"
)

SEEDS=( 1 2 3 4 5 6 7 8 9 10 )

DATASETS=(
    "NT"
    "NTS"
    "NTSO-T"
    "SO-T"
    "News-T"
    "Trends-T"
    "Tweets-T"
)

IDX_ALGORITHM=$((SLURM_ARRAY_TASK_ID / ${#SEEDS[@]}))
IDX_SEED=$((SLURM_ARRAY_TASK_ID % ${#SEEDS[@]}))

ALGORITHM=${ALGORITHMS[$IDX_ALGORITHM]}
SEED=${SEEDS[$IDX_SEED]}

echo "START runexperiment call at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Running with algorithm: $ALGORITHM"
echo "Running with datasets: ${DATASETS[@]}"
echo "Running with seed: $SEED"

file_name="smac3_best_configs/$ALGORITHM/best_config_loo_none_seed_$SEED.json"

apptainer exec \
    --no-home \
    --pwd /opt/project \
    --bind "$PWD:/opt/project" \
    --bind "$DATA_DIR:$DATA_DIR" \
    python311.sif \
    python -u smac_hpo/eval_any_config.py "$file_name" "$ALGORITHM" "best_on_old_datasets/${file_name//\//_}_results.csv" --dataset "${DATASETS[@]}"