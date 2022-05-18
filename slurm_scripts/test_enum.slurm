#!/bin/bash
#SBATCH --job-name=test_enum
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=1         # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --time 00:02:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfsdswork/projects/rech/six/uue59kq/logs/enum/%x-%j.out           # output file name
#SBATCH --array=0-12
#SBATCH --account=six@ajs

# ===== GET DATASET ======

DATASET_ID=$SLURM_ARRAY_TASK_ID
LIST_DATASET=("lm_indic-bn_oscar.jsonl" "lm_pt_oscar.jsonl" "lm_indic-ur_oscar.jsonl" "lm_zhs_oscar.jsonl" "lm_indic-hi_oscar.jsonl" "lm_ar_oscar.jsonl" "lm_ca_oscar.jsonl" "lm_es_oscar.jsonl" "lm_en_oscar.jsonl" "lm_id_oscar.jsonl" "lm_fr_oscar.jsonl" "lm_vi_oscar.jsonl" "lm_eu_oscar.jsonl")
DATASET_NAME=${LIST_DATASET[$SLURM_ARRAY_TASK_ID]}
echo $DATASET_NAME
