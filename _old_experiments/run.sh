#!/bin/bash
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user sussekl@mail.uni-paderborn.de
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -J "text_clust"
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -A hpc-prf-dedsm
#SBATCH --array=0-9

module load lang/Python/3.10.4-GCCcore-11.3.0
module load system/CUDA/12.6.0
source /scratch/hpc-prf-dedsm/sussekl/textClust/.venv/bin/activate

#pip install .

datasets=(
	"examples/data/News-T.json"
	"examples/data/NT.json"
	"examples/data/NTS.json"
	"examples/data/NTSO-T.json"
	"examples/data/Semantic-T_peaked.json"
	"examples/data/Semantic-T.json"
	"examples/data/SO-T.json"
	"examples/data/Trends-T_modified.json"
	"examples/data/Trends-T.json"
	"examples/data/Tweets-T.json"
)

mkdir -p experiments/results

dataset_name=$(basename "${datasets[$SLURM_ARRAY_TASK_ID]}" .json)
srun --ntasks=1 --gres=gpu:a100:1 python experiments/experiment.py --datasets "${datasets[$SLURM_ARRAY_TASK_ID]}" --output "experiments/results/${dataset_name}_comparison_results.csv" --max-samples 2000 --device "cuda"