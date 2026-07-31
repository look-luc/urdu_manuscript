import os
from typing import cast

from datasets import (
    Image,
    IterableDataset,
    interleave_datasets,
    load_dataset,
)

# Determine the absolute directory where get_data.py is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Build the absolute path to the images directory
IMAGE_BASE_DIR = os.path.join(SCRIPT_DIR, "Persian-OCR-230k")


def resolve_path(example):
    image_val = example.get("image")

    if isinstance(image_val, dict) and "path" in image_val:
        path = image_val["path"]
        if path and not path.startswith(("http://", "https://")) and not path.startswith(IMAGE_BASE_DIR):
            example["image"]["path"] = os.path.join(IMAGE_BASE_DIR, path)

    elif isinstance(image_val, str):
        if not image_val.startswith(("http://", "https://")) and not image_val.startswith(IMAGE_BASE_DIR):
            example["image"] = os.path.join(IMAGE_BASE_DIR, image_val)

    return example


def is_valid_example(example):
    """Filters out empty text, null images, and non-existent local image paths."""
    img = example.get("image")
    txt = example.get("text")

    if img is None or txt is None:
        return False

    # Verify text is not blank filler
    if isinstance(txt, str) and not txt.strip():
        return False

    # Check local path validity if string/dict path is present
    if isinstance(img, dict) and "path" in img and img["path"]:
        path_str = str(img["path"])
        if not path_str.startswith(("http://", "https://")) and not os.path.exists(path_str):
            return False
    elif isinstance(img, str) and not img.startswith(("http://", "https://")):
        if not os.path.exists(img):
            return False

    return True


def prepare_dataset(ds, select_cols=True) -> IterableDataset:
    """Enforces column selection, schema casting, and validity filtering before stream conversion."""
    if select_cols:
        ds = ds.select_columns(["image", "text"])
    ds = ds.cast_column("image", Image(decode=False))
    ds = ds.filter(is_valid_example)
    return ds.to_iterable_dataset()


def get_datasets():
    # --- Arabic ---
    ds_arabic_raw = load_dataset(
        "mssqpi/Arabic-OCR-Dataset", split="train", streaming=False, keep_in_memory=False
    )
    ds_arabic = prepare_dataset(ds_arabic_raw)

    # --- Farsi ---
    parsynth_train_raw = load_dataset("hezarai/parsynth-ocr-200k", split="train", streaming=False, keep_in_memory=False).rename_column("image_path", "image")
    parsynth_train = prepare_dataset(parsynth_train_raw)

    parsynth_test_raw = load_dataset("hezarai/parsynth-ocr-200k", split="test", streaming=False, keep_in_memory=False).rename_column("image_path", "image")
    parsynth_test = prepare_dataset(parsynth_test_raw)

    # --- Persian ---
    persian_ocr_dict = load_dataset("ordaktaktak/Persian-OCR-230k", streaming=False)

    persian_ocr_train_raw = persian_ocr_dict["train"].rename_column("fname", "image").map(resolve_path, load_from_cache_file=False)
    persian_ocr_train = prepare_dataset(persian_ocr_train_raw)

    persian_ocr_test_raw = persian_ocr_dict["test"].rename_column("fname", "image").map(resolve_path, load_from_cache_file=False)
    persian_ocr_test = prepare_dataset(persian_ocr_test_raw)

    # --- Urdu ---
    nastaliq_raw = load_dataset("PuristanLabs1/urdu-ocr-1M", "nastaliq", split="train", streaming=False, keep_in_memory=False)
    nastaliq = prepare_dataset(nastaliq_raw)

    naskh_raw = load_dataset("PuristanLabs1/urdu-ocr-1M", "naskh", split="train", streaming=False, keep_in_memory=False)
    naskh = prepare_dataset(naskh_raw)

    urdu_news_raw = load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="train", streaming=False, keep_in_memory=False)
    urdu_news = prepare_dataset(urdu_news_raw)

    urdu_news_test_raw = load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="test", streaming=False, keep_in_memory=False)
    urdu_news_test = prepare_dataset(urdu_news_test_raw)

    urdu_news_val_raw = load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="validation", streaming=False, keep_in_memory=False)
    urdu_news_val = prepare_dataset(urdu_news_val_raw)

    # --- Kannada ---
    kannada_df_train_raw = load_dataset("darknight054/indic-mozhi-ocr", "kannada", split="train", streaming=False, keep_in_memory=False)
    kannada_df_train = prepare_dataset(kannada_df_train_raw)

    val_raw = load_dataset("darknight054/indic-mozhi-ocr", "kannada", split="validation", streaming=False, keep_in_memory=False)
    val = prepare_dataset(val_raw)

    test_raw = load_dataset("darknight054/indic-mozhi-ocr", "kannada", split="test", streaming=False, keep_in_memory=False)
    test = prepare_dataset(test_raw)

    kannada_df_test = interleave_datasets([val, test])

    # --- Slicing & Interleaving ---
    test_dataset = interleave_datasets(
        [
            ds_arabic.take(600),
            nastaliq.take(600),
            naskh.take(600),
            urdu_news_test,
            parsynth_test,
            persian_ocr_test,
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
            persian_ocr_train,
            kannada_df_train,
        ],
        seed=42,
        stopping_strategy="all_exhausted",
    )

    return {
        "train": train_dataset,
        "test": test_dataset,
    }
