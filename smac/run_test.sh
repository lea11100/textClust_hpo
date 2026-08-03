#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sussekl@mail.uni-paderborn.de
#SBATCH --job-name=text_clust_hpo
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --account=hpc-prf-dedsm

echo "START runexperiment call at $(date '+%Y-%m-%d %H:%M:%S')"

module load lang/Python/3.11.3-GCCcore-12.3.0
source /scratch/hpc-prf-dedsm/sussekl/textClust/.textClustEnv/bin/activate

echo "loaded modules and activated virtual environment at $(date '+%Y-%m-%d %H:%M:%S')"

python experiments/smac/optimize.py --loo_dataset News-T --seed 2

echo "FINISHED runexperiment call at $(date '+%Y-%m-%d %H:%M:%S')"  