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

DATASETS=(
    "NT"
    "NTS"
    "NTSO-T"
    "SO-T"
    "News-T"
    "Trends-T"
    "Tweets-T"
)

CONFIG_FILES=(
    "smac3_best_configs/embClust/best_config_loo_none_seed_1.json"
    "smac3_best_configs/embClust/best_config_loo_none_seed_2.json"
    "smac3_best_configs/embClust/best_config_loo_none_seed_3.json"
    "smac3_best_configs/embClust/best_config_loo_none_seed_4.json"
    "smac3_best_configs/embClust/best_config_loo_none_seed_5.json"
    "smac3_best_configs/embClust/best_config_loo_none_seed_6.json"
    "smac3_best_configs/embClust/best_config_loo_none_seed_7.json"
    "smac3_best_configs/embClust/best_config_loo_none_seed_8.json"
    "smac3_best_configs/embClust/best_config_loo_none_seed_9.json"
    "smac3_best_configs/embClust/best_config_loo_none_seed_10.json"
    "smac3_best_configs/inc_clust_ray/best_config_loo_none_seed_1.json"
    "smac3_best_configs/inc_clust_ray/best_config_loo_none_seed_2.json"
    "smac3_best_configs/inc_clust_ray/best_config_loo_none_seed_3.json"
    "smac3_best_configs/inc_clust_ray/best_config_loo_none_seed_4.json"
    "smac3_best_configs/inc_clust_ray/best_config_loo_none_seed_5.json"
    "smac3_best_configs/inc_clust_ray/best_config_loo_none_seed_6.json"
    "smac3_best_configs/inc_clust_ray/best_config_loo_none_seed_7.json"
    "smac3_best_configs/inc_clust_ray/best_config_loo_none_seed_8.json"
    "smac3_best_configs/inc_clust_ray/best_config_loo_none_seed_9.json"
    "smac3_best_configs/inc_clust_ray/best_config_loo_none_seed_10.json"
    "smac3_best_configs/textClust/best_config_loo_none_seed_1.json"
    "smac3_best_configs/textClust/best_config_loo_none_seed_2.json"
    "smac3_best_configs/textClust/best_config_loo_none_seed_3.json"
    "smac3_best_configs/textClust/best_config_loo_none_seed_4.json"
    "smac3_best_configs/textClust/best_config_loo_none_seed_5.json"
    "smac3_best_configs/textClust/best_config_loo_none_seed_6.json"
    "smac3_best_configs/textClust/best_config_loo_none_seed_7.json"
    "smac3_best_configs/textClust/best_config_loo_none_seed_8.json"
    "smac3_best_configs/textClust/best_config_loo_none_seed_9.json"
    "smac3_best_configs/textClust/best_config_loo_none_seed_10.json"
)

ALGORITHM_PER_CONFIG=(
    "embClust"
    "embClust"
    "embClust"
    "embClust"
    "embClust"
    "embClust"
    "embClust"
    "embClust"
    "embClust"
    "embClust"  
    "inc_clust_ray"
    "inc_clust_ray"
    "inc_clust_ray"
    "inc_clust_ray"
    "inc_clust_ray"
    "inc_clust_ray"
    "inc_clust_ray"
    "inc_clust_ray"
    "inc_clust_ray"
    "inc_clust_ray"
    "textClust"
    "textClust"
    "textClust"
    "textClust"
    "textClust"
    "textClust"
    "textClust"
    "textClust"
    "textClust"
    "textClust"
)

CONFIG_FILE=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}
ALGORITHM=${ALGORITHM_PER_CONFIG[$SLURM_ARRAY_TASK_ID]}

echo "START runexperiment call at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Running with config file: $CONFIG_FILE"
echo "Running with algorithm: $ALGORITHM"

module load lang/Python/3.11.3-GCCcore-12.3.0
source /scratch/hpc-prf-dedsm/sussekl/textClust_hpo/.textClust_hpo_env/bin/activate

python -u smac_hpo/eval_any_config.py $CONFIG_FILE $ALGORITHM "any_config_evaluate/${CONFIG_FILE//\//_}_results.csv" --dataset "${DATASETS[@]}"