import os
from typing import cast

from datasets import Dataset, IterableDataset, interleave_datasets, load_dataset
from datasets import Image as HFImage

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_BASE_DIR = os.path.join(SCRIPT_DIR, "Persian-OCR-230k")


def prepare_dataset(ds: Dataset, select_cols=True) -> IterableDataset:
    if select_cols:
        ds = ds.select_columns(["image", "text"])

    ds = ds.cast_column("image", HFImage(decode=True))

    iterable_ds = ds.to_iterable_dataset()
    return iterable_ds

def fix_persian_image_path(example):
    """Prepends absolute base directory path to relative image filename."""
    if isinstance(example["fname"], str):
        example["fname"] = os.path.join(IMAGE_BASE_DIR, example["fname"])
    return example

def get_datasets():
    # --- Arabic ---
    ds_arabic = prepare_dataset(
        cast(Dataset, load_dataset("mssqpi/Arabic-OCR-Dataset", split="train"))
    )
    print("finished loading Arabic data")

    # --- Farsi ---
    parsynth_train = prepare_dataset(
        cast(Dataset, load_dataset("hezarai/parsynth-ocr-200k", split="train")).rename_column(
            "image_path", "image"
        )
    )
    parsynth_test = prepare_dataset(
        cast(Dataset, load_dataset("hezarai/parsynth-ocr-200k", split="test")).rename_column(
            "image_path", "image"
        )
    )
    print("finished loading Farsi data")

    # --- Persian ---
    persian_dict = load_dataset("ordaktaktak/Persian-OCR-230k")

    persian_train_raw = cast(Dataset, persian_dict["train"]).map(fix_persian_image_path)
    persian_test_raw = cast(Dataset, persian_dict["test"]).map(fix_persian_image_path)

    persian_train = prepare_dataset(
        persian_train_raw.rename_column("fname", "image")
    )
    persian_test = prepare_dataset(
        persian_test_raw.rename_column("fname", "image")
    )
    print("finished loading Persian data")

    # --- Urdu ---
    nastaliq = prepare_dataset(
        cast(Dataset, load_dataset("PuristanLabs1/urdu-ocr-1M", "nastaliq", split="train"))
    )
    naskh = prepare_dataset(
        cast(Dataset, load_dataset("PuristanLabs1/urdu-ocr-1M", "naskh", split="train"))
    )
    urdu_news = prepare_dataset(
        cast(Dataset, load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="train"))
    )
    urdu_news_test = prepare_dataset(
        cast(Dataset, load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="test"))
    )
    urdu_news_val = prepare_dataset(
        cast(Dataset, load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="validation"))
    )
    print("finished loading Urdu data")

    # --- Kannada ---
    kannada_train = prepare_dataset(
        cast(Dataset, load_dataset("darknight054/indic-mozhi-ocr", "kannada", split="train"))
    )
    kannada_val = prepare_dataset(
        cast(
            Dataset, load_dataset("darknight054/indic-mozhi-ocr", "kannada", split="validation")
        )
    )
    kannada_test = prepare_dataset(
        cast(Dataset, load_dataset("darknight054/indic-mozhi-ocr", "kannada", split="test"))
    )
    kannada_df_test = interleave_datasets([kannada_val, kannada_test])
    print("finished loading Kannada data")

    test_dataset = interleave_datasets(
        [
            ds_arabic.take(600),
            nastaliq.take(600),
            naskh.take(600),
            urdu_news_test,
            parsynth_test,
            persian_test,
            urdu_news_val,
            kannada_df_test,
        ],
        seed=42,
    )

    train_dataset = interleave_datasets(
        [
            ds_arabic.skip(600),
            nastaliq.skip(600),
            naskh.skip(600),
            urdu_news,
            parsynth_train,
            persian_train,
            kannada_train,
        ],
        seed=42,
        stopping_strategy="all_exhausted",
    )

    return {"train": train_dataset, "test": test_dataset}
