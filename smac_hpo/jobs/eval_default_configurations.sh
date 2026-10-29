#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sussekl@mail.uni-paderborn.de
#SBATCH --job-name=text_clust_hpo
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --account=hpc-prf-dedsm
#SBATCH --output=noctua_logs/eval_default_configs_%J_%a.out
#SBATCH --array=0-6

module load tools/Apptainer/1.1.9-GCCcore-11.3.0

DEFAULT_FILES=(
    "smac_hpo/defaults/default_inc_clust_ray.json"
    "smac_hpo/defaults/river_default_embClust.json"
    "smac_hpo/defaults/default_embClust_2017.json"
    "smac_hpo/defaults/default_embClust_2022.json"
    "smac_hpo/defaults/river_default_textClust.json"
    "smac_hpo/defaults/default_textClust_2017.json"
    "smac_hpo/defaults/default_textClust_2022.json"
)

ALGORITHMS=(
    "inc_clust_ray"
    "embClust"
    "embClust"
    "embClust"
    "textClust"
    "textClust"
    "textClust"
)

ALGORITHM=${ALGORITHMS[$SLURM_ARRAY_TASK_ID]}
FILE=${DEFAULT_FILES[$SLURM_ARRAY_TASK_ID]}
DATA_DIR=$(readlink -f data/textClust_stream_data)

echo "START runexperiment call at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Running with default file: $FILE"
echo "Running with algorithm: $ALGORITHM"

apptainer exec \
    --no-home \
    --pwd /opt/project \
    --bind "$PWD:/opt/project" \
    --bind "$DATA_DIR:$DATA_DIR" \
    python311.sif \
    python -u smac_hpo/eval_any_config.py "$FILE" "$ALGORITHM" "default_on_all_datasets/${FILE//\//_}_results.csv"