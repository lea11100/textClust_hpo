#!/bin/bash
#SBATCH --time=3-12
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sussekl@mail.uni-paderborn.de
#SBATCH --job-name=text_clust_hpo
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --account=hpc-prf-dedsm
#SBATCH --output=noctua_logs/run_full_%J_%a.out
#SBATCH --array=0-29

module load tools/Apptainer/1.1.9-GCCcore-11.3.0
DATA_DIR=$(readlink -f data/textClust_stream_data)

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

apptainer exec \
	--no-home \
	--pwd /opt/project \
	--bind "$PWD:/opt/project" \
	--bind "$DATA_DIR:$DATA_DIR" \
	python311.sif \
	python -u smac_hpo/optimize.py \
	--seed "$SEED" \
	--algorithm "$ALGORITHM"