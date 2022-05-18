import argparse
from pathlib import Path
import logging
import datasets
from datasets import load_from_disk
from huggingface_hub import HfApi

datasets.utils.logging.set_verbosity_info()
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description="Push to hub")
    parser.add_argument("--dataset-name", type=str)
    parser.add_argument(
        "--path-prefix",
        type=Path,
        default="/gpfswork/rech/six/commun/bigscience-training/clean_v2/bigscience-catalogue-lm-data",
    )
    parser.add_argument("--hub_repo_prefix", type=Path, default="bigscience-catalogue-lm-data")
    args = parser.parse_args()
    return args


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    args = get_args()
    logger.info(f"** The job is runned with the following arguments: **\n{args}\n **** ")

    hf_api = HfApi()
    list_datasets = hf_api.list_datasets(author="bigscience-catalogue-lm-data", use_auth_token=True)

    dset_id = f"{args.hub_repo_prefix}/cleaned_{args.dataset_name}"

    # check if ds already pushed
    check = [ds_info.id for ds_info in list_datasets if ds_info.id == dset_id]
    if len(check) > 0:
        logging.info(f"The dataset {dset_id} has already been pushed to the hub. Doing nothing.")
        return

    # dset = load_from_disk(args.path_prefix / args.dataset_name / "final")
    # logging.info("Datset loaded ", dset)
    # dset.push_to_hub(dset_id, private=True)
    # logging.info("Finish successfully")


if __name__ == "__main__":
    main()
