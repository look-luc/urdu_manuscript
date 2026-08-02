import os
from typing import cast

from datasets import IterableDataset, interleave_datasets, load_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_BASE_DIR = os.path.join(SCRIPT_DIR, "Persian-OCR-230k")


def fix_persian_image_path(example):
    """Prepends absolute base directory path to relative image filename if string."""
    if isinstance(example.get("fname"), str):
        example["fname"] = os.path.join(IMAGE_BASE_DIR, example["fname"])
    return example


def get_datasets():
    # --- 1. Arabic (Streamed) ---
    sard_raw = cast(
        IterableDataset,
        load_dataset("riotu-lab/SARD", split="train", streaming=True)[
            "Traditional_Arabic"
        ],
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
    print("finished loading Arabic streaming stream")

    # --- 2. Farsi / Persian (Streamed) ---
    parsynth_train = (
        cast(
            IterableDataset,
            load_dataset(
                "hezarai/parsynth-ocr-200k", split="train", streaming=True
            ),
        )
        .rename_column("image_path", "image")
        .select_columns(["image", "text"])
    )

    parsynth_test = (
        cast(
            IterableDataset,
            load_dataset(
                "hezarai/parsynth-ocr-200k", split="test", streaming=True
            ),
        )
        .rename_column("image_path", "image")
        .select_columns(["image", "text"])
    )

    persian_train = (
        cast(
            IterableDataset,
            load_dataset(
                "ordaktaktak/Persian-OCR-230k", split="train", streaming=True
            ),
        )
        .map(fix_persian_image_path)
        .rename_column("fname", "image")
        .select_columns(["image", "text"])
    )

    persian_test = (
        cast(
            IterableDataset,
            load_dataset(
                "ordaktaktak/Persian-OCR-230k", split="test", streaming=True
            ),
        )
        .map(fix_persian_image_path)
        .rename_column("fname", "image")
        .select_columns(["image", "text"])
    )

    persian_train_combined = interleave_datasets(
        [parsynth_train, persian_train],
        probabilities=[0.5, 0.5],
        seed=42,
    )
    print("finished loading Farsi/Persian streaming stream")

    # --- 3. Urdu (Streamed) ---
    nastaliq = cast(
        IterableDataset,
        load_dataset(
            "PuristanLabs1/urdu-ocr-1M",
            "nastaliq",
            split="train",
            streaming=True,
        ),
    ).select_columns(["image", "text"])

    naskh = cast(
        IterableDataset,
        load_dataset(
            "PuristanLabs1/urdu-ocr-1M",
            "naskh",
            split="train",
            streaming=True,
        ),
    ).select_columns(["image", "text"])

    urdu_news = cast(
        IterableDataset,
        load_dataset(
            "oddadmix/qari-0.2.2-news-dataset-large",
            split="train",
            streaming=True,
        ),
    ).select_columns(["image", "text"])

    urdu_news_test = cast(
        IterableDataset,
        load_dataset(
            "oddadmix/qari-0.2.2-news-dataset-large",
            split="test",
            streaming=True,
        ),
    ).select_columns(["image", "text"])

    urdu_news_val = cast(
        IterableDataset,
        load_dataset(
            "oddadmix/qari-0.2.2-news-dataset-large",
            split="validation",
            streaming=True,
        ),
    ).select_columns(["image", "text"])

    print("finished loading Urdu streaming stream")

    # --- 4. Test Dataset Assembly ---
    test_dataset = interleave_datasets(
        [
            ds_arabic.take(600),
            nastaliq.take(800),
            naskh.take(800),
            urdu_news_test,
            parsynth_test,
            persian_test,
            urdu_news_val,
        ],
        seed=42,
    )

    # --- 5. Train Dataset Assembly ---
    train_sources = [
        nastaliq.skip(800),
        naskh.skip(800),
        ds_arabic.skip(600),
        persian_train_combined,
        urdu_news,
    ]

    train_probabilities = [0.50, 0.20, 0.15, 0.10, 0.05]

    train_dataset = interleave_datasets(
        datasets=train_sources,
        probabilities=train_probabilities,
        stopping_strategy="all_exhausted",
        seed=42,
    ).shuffle(buffer_size=10000, seed=42)

    return {"train": train_dataset, "test": test_dataset}
