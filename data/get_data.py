import os
from typing import cast

import torchvision.transforms.functional as F
from datasets import (
    IterableDataset,
    interleave_datasets,
    load_dataset,
)
from torchvision.io import ImageReadMode, read_image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_BASE_DIR = os.path.join(SCRIPT_DIR, "Persian-OCR-230k")


def get_datasets(buffer_size: int = 10000):
    print("Loading datasets in streaming mode...")

    # --- 1. Arabic ---
    sard_raw = cast(
        IterableDataset,
        load_dataset("riotu-lab/SARD", split="Traditional_Arabic", streaming=True),
    )
    cols = sard_raw.column_names or []

    if "text" not in cols:
        if "label" in cols:
            sard_raw = sard_raw.rename_column("label", "text")
        elif "transcription" in cols:
            sard_raw = sard_raw.rename_column("transcription", "text")

    if "image" not in cols and "img" in cols:
        sard_raw = sard_raw.rename_column("img", "image")

    ds_arabic = sard_raw.select_columns(["image", "text"])
    print("Finished loading Arabic dataset.")

    # --- 2. Farsi / Persian ---
    parsynth_train_raw = cast(
        IterableDataset,
        load_dataset("hezarai/parsynth-ocr-200k", split="train", streaming=True),
    ).rename_column("image_path", "image")

    parsynth_test_raw = cast(
        IterableDataset,
        load_dataset("hezarai/parsynth-ocr-200k", split="test", streaming=True),
    )

    parsynth_test = parsynth_test_raw

    persian_train_raw = cast(
        IterableDataset,
        load_dataset("ordaktaktak/Persian-OCR-230k", split="train", streaming=True),
    ).rename_column("fname", "text")

    persian_train = persian_train_raw

    persian_test_raw = cast(
        IterableDataset,
        load_dataset("ordaktaktak/Persian-OCR-230k", split="test", streaming=True),
    ).rename_column("fname", "text")

    persian_test = persian_test_raw

    persian_train_combined = interleave_datasets(
        [parsynth_train_raw, persian_train],
        probabilities=[0.5, 0.5],
        seed=42,
    )
    print("Finished loading Farsi/Persian datasets.")

    # --- 3. Urdu ---
    nastaliq = cast(
            IterableDataset,
            load_dataset("PuristanLabs1/urdu-ocr-1M", "nastaliq", split="train", streaming=True),
        )

    naskh = cast(
        IterableDataset,
        load_dataset("PuristanLabs1/urdu-ocr-1M", "naskh", split="train", streaming=True),
    )

    urdu_news = cast(
        IterableDataset,
        load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="train", streaming=True),
    )

    urdu_news_test = cast(
        IterableDataset,
        load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="test", streaming=True),
    )

    urdu_news_val = cast(
        IterableDataset,
        load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="validation", streaming=True),
    )
    print("Finished loading Urdu datasets.")

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
