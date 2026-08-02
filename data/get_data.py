import os
from typing import cast

from datasets import Image as HFImage
from datasets import IterableDataset, interleave_datasets, load_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_BASE_DIR = os.path.join(SCRIPT_DIR, "Persian-OCR-230k")


def fix_persian_image_path(example):
    """Safely resolves relative Persian image paths and checks for existence on disk."""
    if not isinstance(example, dict):
        return {"fname": None}

    fname = example.get("fname")
    if isinstance(fname, str):
        full_path = os.path.join(IMAGE_BASE_DIR, fname)
        if os.path.exists(full_path):
            example["fname"] = full_path
        else:
            example["fname"] = None
    return example


def is_valid_example(example):
    """Filters out corrupt, missing, or un-decoded image and text samples."""
    if not isinstance(example, dict):
        return False
    if example.get("image") is None or example.get("text") is None:
        return False
    return True


def get_datasets():
    # --- 1. Arabic (Streamed) ---
    sard_raw = cast(
        IterableDataset,
        load_dataset("riotu-lab/SARD", streaming=True)[
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

    ds_arabic = (
        sard_raw.select_columns(["image", "text"])
        .cast_column("image", HFImage(decode=True))
        .filter(is_valid_example)
    )
    print("finished loading Arabic streaming stream")

    # --- 2. Farsi / Persian (Streamed & Filtered) ---
    parsynth_train = (
        cast(
            IterableDataset,
            load_dataset(
                "hezarai/parsynth-ocr-200k", split="train", streaming=True
            ),
        )
        .rename_column("image_path", "image")
        .select_columns(["image", "text"])
        .cast_column("image", HFImage(decode=True))
        .filter(is_valid_example)
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
        .cast_column("image", HFImage(decode=True))
        .filter(is_valid_example)
    )

    persian_train = (
        cast(
            IterableDataset,
            load_dataset(
                "ordaktaktak/Persian-OCR-230k", split="train", streaming=True
            ),
        )
        .map(fix_persian_image_path)
        .filter(lambda x: isinstance(x, dict) and x.get("fname") is not None)
        .rename_column("fname", "image")
        .select_columns(["image", "text"])
        .cast_column("image", HFImage(decode=True))
        .filter(is_valid_example)
    )

    persian_test = (
        cast(
            IterableDataset,
            load_dataset(
                "ordaktaktak/Persian-OCR-230k", split="test", streaming=True
            ),
        )
        .map(fix_persian_image_path)
        .filter(lambda x: isinstance(x, dict) and x.get("fname") is not None)
        .rename_column("fname", "image")
        .select_columns(["image", "text"])
        .cast_column("image", HFImage(decode=True))
        .filter(is_valid_example)
    )

    persian_train_combined = interleave_datasets(
        [parsynth_train, persian_train],
        probabilities=[0.5, 0.5],
        seed=42,
    )
    print("finished loading Farsi/Persian streaming stream")

    # --- 3. Urdu (Streamed & Filtered) ---
    nastaliq = (
        cast(
            IterableDataset,
            load_dataset(
                "PuristanLabs1/urdu-ocr-1M",
                "nastaliq",
                split="train",
                streaming=True,
            ),
        )
        .select_columns(["image", "text"])
        .cast_column("image", HFImage(decode=True))
        .filter(is_valid_example)
    )

    naskh = (
        cast(
            IterableDataset,
            load_dataset(
                "PuristanLabs1/urdu-ocr-1M",
                "naskh",
                split="train",
                streaming=True,
            ),
        )
        .select_columns(["image", "text"])
        .cast_column("image", HFImage(decode=True))
        .filter(is_valid_example)
    )

    urdu_news = (
        cast(
            IterableDataset,
            load_dataset(
                "oddadmix/qari-0.2.2-news-dataset-large",
                split="train",
                streaming=True,
            ),
        )
        .select_columns(["image", "text"])
        .cast_column("image", HFImage(decode=True))
        .filter(is_valid_example)
    )

    urdu_news_test = (
        cast(
            IterableDataset,
            load_dataset(
                "oddadmix/qari-0.2.2-news-dataset-large",
                split="test",
                streaming=True,
            ),
        )
        .select_columns(["image", "text"])
        .cast_column("image", HFImage(decode=True))
        .filter(is_valid_example)
    )

    urdu_news_val = (
        cast(
            IterableDataset,
            load_dataset(
                "oddadmix/qari-0.2.2-news-dataset-large",
                split="validation",
                streaming=True,
            ),
        )
        .select_columns(["image", "text"])
        .cast_column("image", HFImage(decode=True))
        .filter(is_valid_example)
    )

    print("finished loading Urdu streaming stream")

    # --- 4. Test Dataset Assembly ---
    test_dataset = interleave_datasets(
        [
            ds_arabic,
            nastaliq,
            naskh,
            urdu_news_test,
            parsynth_test,
            persian_test,
            urdu_news_val,
        ],
        seed=42,
    )

    # --- 5. Train Dataset Assembly ---
    train_sources = [
        nastaliq,
        naskh,
        ds_arabic,
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
