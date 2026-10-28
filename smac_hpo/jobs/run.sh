#!/bin/bash
#SBATCH --time=12-12
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sussekl@mail.uni-paderborn.de
#SBATCH --job-name=text_clust_hpo
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --account=hpc-prf-dedsm
#SBATCH --output=smac3_logs/%a.out
#SBATCH --array=0-240

module load tools/Apptainer/1.1.9-GCCcore-11.3.0

DATASETS=( "Arabic_Twitter Semantic40-T" "KBP Arabic_7"
           "Mix_Data CrisisLexT6" "Semantic10-T Event2018"
           "MAVEN HumAID" "Semantic5-T CrisisLexT7"
           "CrisisNLP CrisisMMD" "Event2012 CrisisLexT26" )
SEEDS=( 1 2 3 4 5 6 7 8 9 10 )
ALGORITHMS=( "textClust" "embClust" "inc_clust_ray" )

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

apptainer exec \
    --no-home \
    --pwd /opt/project \
    --bind "$PWD:/opt/project" \
    python311.sif \
    python -u smac_hpo/optimize.py \
    --seed "$SEED" \
    --algorithm "$ALGORITHM" \
    --loo_datasets "$DATASET"