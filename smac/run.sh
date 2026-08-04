#!/bin/bash
#SBATCH --time=7-12
#SBATCH --mem=4G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sussekl@mail.uni-paderborn.de
#SBATCH --job-name=text_clust_hpo
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --account=hpc-prf-dedsm
#SBATCH --output=output/smac/%a.out
#SBATCH --array=0-159

DATASETS=( "Arabic_Twitter Semantic40-T" "KBP Arabic_7"
           "Mix_Data CrisisLexT6" "Semantic10-T Event2018"
           "MAVEN HumAID" "Semantic5-T CrisisLexT7"
           "CrisisNLP CrisisMMD" "Event2012 CrisisLexT26" )
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

python experiments/smac/optimize.py --seed $SEED --algorithm $ALGORITHM --loo_datasets $DATASETS