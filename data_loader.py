from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader


DATASET_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/"
    "thakur/BEIR/datasets/scifact.zip"
)
DATA_DIR = "data"


def load_scifact_dataset():
    print("Loading SciFact dataset...")
    data_path = beir_util.download_and_unzip(
        DATASET_URL, DATA_DIR
    )

    corpus, queries, qrels = GenericDataLoader(
        data_folder=data_path
    ).load(split="test")

    return corpus, queries, qrels