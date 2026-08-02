import os
from typing import cast

from datasets import IterableDataset, interleave_datasets, load_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_BASE_DIR = os.path.join(SCRIPT_DIR, "Persian-OCR-230k")


def standardize_stream(ds: IterableDataset, img_col: str, txt_col: str) -> IterableDataset:
    """Utility to rename columns and select only 'image' and 'text'."""
    cols = ds.column_names or []

    if img_col in cols and img_col != "image":
        ds = ds.rename_column(img_col, "image")
    if txt_col in cols and txt_col != "text":
        ds = ds.rename_column(txt_col, "text")

    return ds.select_columns(["image", "text"])


def get_datasets(buffer_size: int = 100):
    """Loads dataset streams lazily with a configurable shuffle buffer size."""
    print(f"Loading datasets in streaming mode (buffer_size={buffer_size})...")

    # --- 1. Arabic ---
    sard_raw = cast(
        IterableDataset,
        load_dataset("riotu-lab/SARD", split="Traditional_Arabic", streaming=True),
    )
    ds_arabic = standardize_stream(sard_raw, img_col="image", txt_col="label")

    # --- 2. Farsi / Persian ---
    parsynth_train_raw = cast(
        IterableDataset,
        load_dataset("hezarai/parsynth-ocr-200k", split="train", streaming=True),
    )
    parsynth_train = standardize_stream(parsynth_train_raw, img_col="image_path", txt_col="text")

    parsynth_test_raw = cast(
        IterableDataset,
        load_dataset("hezarai/parsynth-ocr-200k", split="test", streaming=True),
    )
    parsynth_test = standardize_stream(parsynth_test_raw, img_col="image_path", txt_col="text")

    persian_train_raw = cast(
        IterableDataset,
        load_dataset("ordaktaktak/Persian-OCR-230k", split="train", streaming=True),
    )
    persian_train = standardize_stream(persian_train_raw, img_col="image", txt_col="fname")

    persian_test_raw = cast(
        IterableDataset,
        load_dataset("ordaktaktak/Persian-OCR-230k", split="test", streaming=True),
    )
    persian_test = standardize_stream(persian_test_raw, img_col="image", txt_col="fname")

    persian_train_combined = interleave_datasets(
        [parsynth_train, persian_train],
        probabilities=[0.5, 0.5],
        seed=42,
    )

    # --- 3. Urdu ---
    nastaliq_raw = cast(
        IterableDataset,
        load_dataset("PuristanLabs1/urdu-ocr-1M", "nastaliq", split="train", streaming=True),
    )
    nastaliq = standardize_stream(nastaliq_raw, img_col="image", txt_col="text")

    naskh_raw = cast(
        IterableDataset,
        load_dataset("PuristanLabs1/urdu-ocr-1M", "naskh", split="train", streaming=True),
    )
    naskh = standardize_stream(naskh_raw, img_col="image", txt_col="text")

    urdu_news_raw = cast(
        IterableDataset,
        load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="train", streaming=True),
    )
    urdu_news = standardize_stream(urdu_news_raw, img_col="image", txt_col="text")

    urdu_news_test_raw = cast(
        IterableDataset,
        load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="test", streaming=True),
    )
    urdu_news_test = standardize_stream(urdu_news_test_raw, img_col="image", txt_col="text")

    urdu_news_val_raw = cast(
        IterableDataset,
        load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="validation", streaming=True),
    )
    urdu_news_val = standardize_stream(urdu_news_val_raw, img_col="image", txt_col="text")

    # --- Interleaving Standardized Streams ---
    test_sources = [
        ds_arabic.take(500),
        nastaliq.take(1000),
        naskh.take(400),
        urdu_news_test.take(300),
        parsynth_test.take(400),
        persian_test.take(400),
        urdu_news_val.take(300),
    ]

    test_dataset = interleave_datasets(test_sources, seed=42)

    train_sources = [
        nastaliq.skip(1000),
        ds_arabic.skip(500),
        naskh.skip(400),
        persian_train_combined,
        urdu_news,
    ]

    train_probabilities = [0.55, 0.20, 0.12, 0.08, 0.05]

    train_dataset = interleave_datasets(
        datasets=train_sources,
        probabilities=train_probabilities,
        stopping_strategy="all_exhausted",
        seed=42,
    ).shuffle(seed=42, buffer_size=buffer_size)

    return {"train": train_dataset, "test": test_dataset}
