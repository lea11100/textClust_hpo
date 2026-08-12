#!/bin/bash
#SBATCH --time=7-12
#SBATCH --mem=4G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sussekl@mail.uni-paderborn.de
#SBATCH --job-name=text_clust_hpo_eval
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --account=hpc-prf-dedsm
#SBATCH --output=output/smac/%a.out
#SBATCH --array=0-200

if [[ $SLURM_ARRAY_TASK_ID -eq 200 ]]; then
    echo "Evaluating default configuration"
    python experiments/smac/eval_best_config.py --use-default-config --algorithm textClust --output-csv smac3_results_eval/textClust/default_performances.json
    python experiments/smac/eval_best_config.py --use-default-config --algorithm embClust --output-csv smac3_results_eval/embClust/default_performances.json
    echo "Finished evaluating default configuration"
    exit 0
fi

DATASETS=( "Semantic10-T" "Semantic40-T" "Semantic5-T" "NT" "NTS" "NTSO" "SO-T" "News-T" "Trends-T" "Tweets-T" )
SEEDS=( 1 2 3 4 5 6 7 8 9 10 )
ALGORITHMS=( "textClust" "embClust" )

N_DATASETS=${#DATASETS[@]}
N_SEEDS=${#SEEDS[@]}
N_ALGORITHMS=${#ALGORITHMS[@]}

IDX_DATASET=$((SLURM_ARRAY_TASK_ID / (N_SEEDS * N_ALGORITHMS)))
IDX_SEED=$(( (SLURM_ARRAY_TASK_ID / N_ALGORITHMS) % N_SEEDS ))
IDX_ALGORITHM=$((SLURM_ARRAY_TASK_ID % N_ALGORITHMS))

DATASET=${DATASETS[$IDX_DATASET]}
SEED=${SEEDS[$IDX_SEED]}
ALGORITHM=${ALGORITHMS[$IDX_ALGORITHM]}

echo "START runexperiment call at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Running with dataset: $DATASET, seed: $SEED, algorithm: $ALGORITHM"

module load lang/Python/3.11.3-GCCcore-12.3.0
source /scratch/hpc-prf-dedsm/sussekl/textClust/.textClustEnv/bin/activate

CONFIG_FILE="best_config_loo_${DATASET}_seed_${SEED}.json"

python -u experiments/smac/eval_best_config.py $CONFIG_FILE --algorithm $ALGORITHM --output-csv smac3_results_eval/${ALGORITHM}/${CONFIG_FILE}