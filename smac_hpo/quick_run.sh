#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sussekl@mail.uni-paderborn.de
#SBATCH --job-name=text_clust_hpo
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --account=hpc-prf-dedsm
#SBATCH --output=noctua_logs/%j_%a.out
#SBATCH --array=0-2


CONFIG_FILES=(
    "smac_hpo/default_embClust.json"
    "smac_hpo/default_inc_clust_ray.json"
    "smac_hpo/default_textClust.json"
)

ALGORITHM_PER_CONFIG=(
    "embClust"
    "inc_clust_ray"
    "textClust"
)

CONFIG_FILE=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}
ALGORITHM=${ALGORITHM_PER_CONFIG[$SLURM_ARRAY_TASK_ID]}

echo "START runexperiment call at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Running with config file: $CONFIG_FILE"

module load lang/Python/3.11.3-GCCcore-12.3.0
source /scratch/hpc-prf-dedsm/sussekl/textClust_hpo/.textClust_hpo_env/bin/activate

python -u smac_hpo/eval_any_config.py --config_path $CONFIG_FILE --algorithm $ALGORITHM --output_csv "any_config_evaluate/${CONFIG_FILE//\//_}_results.csv"