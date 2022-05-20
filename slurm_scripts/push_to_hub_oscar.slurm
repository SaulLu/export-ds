#!/bin/bash
#SBATCH --job-name=push_to_hub_oscar
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1    # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=4         # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=compil
#SBATCH --time 04:00:00        # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfsdswork/projects/rech/six/uue59kq/logs/push_to_hub_oscar/%x-%j.out     # output file name
#SBATCH --array=0
#SBATCH --account=ajs@cpu

conda activate lucile
# ===== GET DATASET ======

DATASET_ID=$SLURM_ARRAY_TASK_ID
LIST_DATASET=(
    "lm_ar_oscar"
    "lm_eu_oscar"
    "lm_indic-hi_oscar"
    "lm_zhs_oscar"
    "lm_ca_oscar"
    "lm_fr_oscar"
    "lm_indic-ur_oscar"
    "lm_en_oscar"
    "lm_id_oscar"
    "lm_pt_oscar"
    "lm_es_oscar"
    "lm_indic-bn_oscar"
    "lm_vi_oscar"
)
DATASET_NAME=${LIST_DATASET[$SLURM_ARRAY_TASK_ID]}
echo "DATASET_NAME "$DATASET_NAME
echo "SLURM_ARRAY_TASK_ID "$SLURM_ARRAY_TASK_ID

# ===== PUSH TO HUB ======
WORKING_DIR=/gpfswork/rech/six/uue59kq/repos/export-ds
pushd $WORKING_DIR

python python_scripts/push_to_hub.py \
    --dataset-name $DATASET_NAME \
    --path-prefix /gpfswork/rech/six/commun/bigscience-training/oscar_pii_no_id_no_num_jsonl/