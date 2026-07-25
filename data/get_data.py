import os
from typing import cast

from datasets import (
    Image,
    IterableDataset,
    concatenate_datasets,
    interleave_datasets,
    load_dataset,
)

# Determine the absolute directory where get_data.py is located (/projects/.../urdu_manuscript/data)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Build the absolute path to the images directory regardless of where run.py is executed from
IMAGE_BASE_DIR = os.path.join(SCRIPT_DIR, "Persian-OCR-230k")

def resolve_path(example):
    """Prepends the correct absolute base directory to the image path string or dictionary."""
    # Handle case where 'image' is a dictionary from Image(decode=False)
    if isinstance(example["image"], dict) and "path" in example["image"]:
        path = example["image"]["path"]
        if path and not path.startswith(IMAGE_BASE_DIR):
            example["image"]["path"] = os.path.join(IMAGE_BASE_DIR, path)

    # Handle case where 'image' is still a raw string
    elif isinstance(example["image"], str):
        if not example["image"].startswith(IMAGE_BASE_DIR):
            example["image"] = os.path.join(IMAGE_BASE_DIR, example["image"])

    return example

def force_image_schema(ds):
    """Ensures the dataset has an 'image' feature of type Image without premature decoding."""
    return ds.cast_column("image", Image(decode=False))

def get_datasets():
    # --- Arabic ---
    ds_arabic: IterableDataset = load_dataset(
        "mssqpi/Arabic-OCR-Dataset", split="train", streaming=False, keep_in_memory=False
    ).to_iterable_dataset()

    # --- Farsi ---
    parsynth_train_load = load_dataset("hezarai/parsynth-ocr-200k", split="train", streaming=False, keep_in_memory=False)
    parsynth_train_raw = parsynth_train_load.rename_column("image_path", "image")
    parsynth_train: IterableDataset = force_image_schema(parsynth_train_raw.select_columns(["image", "text"])).to_iterable_dataset()

    parsynth_test_load = load_dataset("hezarai/parsynth-ocr-200k", split="test", streaming=False, keep_in_memory=False)
    parsynth_test_raw = parsynth_test_load.rename_column("image_path", "image")
    parsynth_test: IterableDataset = force_image_schema(parsynth_test_raw.select_columns(["image", "text"])).to_iterable_dataset()

    # --- Persian ---
    persian_ocr_dict = load_dataset("ordaktaktak/Persian-OCR-230k", streaming=False)

    persian_ocr_train_raw = persian_ocr_dict["train"].rename_column("fname", "image")
    persian_ocr_train_mapped = persian_ocr_train_raw.map(resolve_path, load_from_cache_file=False)
    persian_ocr_train: IterableDataset = force_image_schema(persian_ocr_train_mapped.select_columns(["image", "text"])).to_iterable_dataset()

    persian_ocr_test_raw = persian_ocr_dict["test"].rename_column("fname", "image")
    persian_ocr_test_mapped = persian_ocr_test_raw.map(resolve_path, load_from_cache_file=False)
    persian_ocr_test: IterableDataset = force_image_schema(persian_ocr_test_mapped.select_columns(["image", "text"])).to_iterable_dataset()

    # --- Urdu ---
    nastaliq: IterableDataset = load_dataset(
        "PuristanLabs1/urdu-ocr-1M",
        "nastaliq",
        split="train",
        streaming=False,
        keep_in_memory=False
    ).select_columns(["image", "text"]).to_iterable_dataset()

    naskh: IterableDataset = load_dataset(
        "PuristanLabs1/urdu-ocr-1M",
        "naskh",
        split="train",
        streaming=False,
        keep_in_memory=False
    ).select_columns(["image", "text"]).to_iterable_dataset()

    urdu_news: IterableDataset = load_dataset(
        "oddadmix/qari-0.2.2-news-dataset-large",
        split="train",
        streaming=False,
        keep_in_memory=False
    ).select_columns(["image", "text"]).to_iterable_dataset()

    urdu_news_test: IterableDataset = load_dataset(
        "oddadmix/qari-0.2.2-news-dataset-large",
        split="test",
        streaming=False,
        keep_in_memory=False
    ).select_columns(["image", "text"]).to_iterable_dataset()

    urdu_news_val: IterableDataset = load_dataset(
        "oddadmix/qari-0.2.2-news-dataset-large",
        split="validation",
        streaming=False,
        keep_in_memory=False
    ).select_columns(["image", "text"]).to_iterable_dataset()

    # --- Kannada ---
    kannada_df_train: IterableDataset = load_dataset(
        "darknight054/indic-mozhi-ocr",
        "kannada",
        split='train',
        streaming=False,
        keep_in_memory=False
    ).select_columns(["image", "text"]).to_iterable_dataset()

    val: IterableDataset = load_dataset(
        "darknight054/indic-mozhi-ocr",
        "kannada",
        split='validation',
        streaming=False,
        keep_in_memory=False
    ).select_columns(["image", "text"]).to_iterable_dataset()

    test: IterableDataset = load_dataset(
        "darknight054/indic-mozhi-ocr",
        "kannada",
        split='test',
        streaming=False,
        keep_in_memory=False
    ).select_columns(["image", "text"]).to_iterable_dataset()

    kannada_df_test = interleave_datasets([val, test])

    # --- Slicing & Interleaving ---
    test_dataset = interleave_datasets(
        [
            force_image_schema(ds_arabic.take(600)),
            force_image_schema(nastaliq.take(600)),
            force_image_schema(naskh.take(600)),
            force_image_schema(urdu_news_test),
            parsynth_test,
            persian_ocr_test,
            force_image_schema(urdu_news_val),
            force_image_schema(kannada_df_test)
        ],
        seed=42
    )

    train_dataset = interleave_datasets(
        [
            force_image_schema(ds_arabic.skip(600)),
            force_image_schema(nastaliq.skip(600)),
            force_image_schema(naskh.skip(600)),
            force_image_schema(urdu_news),
            parsynth_train,
            persian_ocr_train,
            force_image_schema(kannada_df_train)
        ],
        seed=42,
        stopping_strategy="all_exhausted"
    )

    return {
        "train": train_dataset,
        "test": test_dataset
    }
