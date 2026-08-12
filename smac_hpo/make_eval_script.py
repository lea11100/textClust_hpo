import json
import os

script = """
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
#SBATCH --array=0-num_experiments

experiment=(experiment_list)

echo "START runexperiment call at $(date '+%Y-%m-%d %H:%M:%S')"

module load lang/Python/3.11.3-GCCcore-12.3.0
source /scratch/hpc-prf-dedsm/sussekl/textClust/.textClustEnv/bin/activate

COMMANDS_PER_TASK=1
TOTAL_COMMANDS=${#experiment[@]}
task_start=$(( SLURM_ARRAY_TASK_ID * COMMANDS_PER_TASK ))
task_end=$(( task_start + COMMANDS_PER_TASK - 1 ))
for (( cmd_index=task_start; cmd_index<=task_end && cmd_index<TOTAL_COMMANDS; cmd_index++ )); do
    python experiments/smac/eval_best_config.py ${experiment[$cmd_index]}
done
"""

experiment_list = []

for filename in sorted(os.listdir("smac3_results")):
    if filename.endswith(".json"):
        experiment_list.append(f"\"{filename} --output-csv smac3_results_eval/{filename}\"")

experiment_list.append("--use-default-config")  # Add the default configuration as the last experiment

experiment_string = " \\\n".join(experiment_list)

script = script.replace("experiment_list", experiment_string)
script = script.replace("num_experiments", str(len(experiment_list) -1 ))
print(script)