from typing import cast

from datasets import Dataset, concatenate_datasets, load_dataset


def get_datasets():
    ds_arabic: Dataset = load_dataset("mssqpi/Arabic-OCR-Dataset", split="train")
    ds_arabic_split = ds_arabic.train_test_split(test_size=0.2, seed=42)
    ds_arabic_train = ds_arabic_split["train"]
    ds_arabic_test = ds_arabic_split["test"]

    parsynth_ds_train = load_dataset("hezarai/parsynth-ocr-200k", split="train")
    parsynth_ds_test = load_dataset("hezarai/parsynth-ocr-200k", split="test")

    persion_ocr_ds_train = load_dataset("ordaktaktak/Persian-OCR-230k", streaming=True, split="train")
    persion_ocr_ds_test = load_dataset("ordaktaktak/Persian-OCR-230k", streaming=True, split="test")

    nastaliq_ds: Dataset = load_dataset("PuristanLabs1/urdu-ocr-1M", "nastaliq")
    nastaliq_ds_split = nastaliq_ds.train_test_split(test_size=0.2, seed=42)
    nastaliq_ds_train = nastaliq_ds_split["train"]
    nastaliq_ds_test = nastaliq_ds_split["test"]

    naskh_ds = load_dataset("PuristanLabs1/urdu-ocr-1M", "naskh")
    naskh_ds_split = naskh_ds.train_test_split(test_size=0.2, seed=42)
    naskh_ds_train = naskh_ds_split["train"]
    naskh_ds_test = naskh_ds_split["test"]

    urdu_news_ds_train = load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="train")

    urdu_news_ds_test = load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="test")
    urdu_news_ds_val = load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="validation")

    combine_urdu_news_ds_test = concatenate_datasets(
        [
            cast(Dataset, urdu_news_ds_test),
            cast(Dataset, urdu_news_ds_val)
        ],
        axis=1
    )
