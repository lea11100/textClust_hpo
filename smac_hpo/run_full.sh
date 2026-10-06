#!/bin/bash
#SBATCH --time=2-12
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sussekl@mail.uni-paderborn.de
#SBATCH --job-name=text_clust_hpo
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --account=hpc-prf-dedsm
#SBATCH --output=smac3_logs/%a.out
#SBATCH --array=0-29

SEEDS=( 1 2 3 4 5 6 7 8 9 10 )
ALGORITHMS=( "textClust" "embClust" "inc_clust_ray" )

N_SEEDS=${#SEEDS[@]}
N_ALGORITHMS=${#ALGORITHMS[@]}

echo "N_SEEDS: $N_SEEDS, N_ALGORITHMS: $N_ALGORITHMS"

IDX_SEED=$(( (SLURM_ARRAY_TASK_ID / N_ALGORITHMS) % N_SEEDS ))
IDX_ALGORITHM=$((SLURM_ARRAY_TASK_ID % N_ALGORITHMS))

SEED=${SEEDS[$IDX_SEED]}
ALGORITHM=${ALGORITHMS[$IDX_ALGORITHM]}

echo "START runexperiment call at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Running with seed: $SEED, algorithm: $ALGORITHM"

module load lang/Python/3.11.3-GCCcore-12.3.0
source /scratch/hpc-prf-dedsm/sussekl/textClust_hpo/.textClust_hpo_env/bin/activate

python -u smac_hpo/optimize.py --seed $SEED --algorithm $ALGORITHM