import os
from typing import cast

from datasets import (
    IterableDataset,
    interleave_datasets,
    load_dataset,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_BASE_DIR = os.path.join(SCRIPT_DIR, "Persian-OCR-230k")


def format_dataset_stream(
    ds: IterableDataset, default_img_dir: str = None
) -> IterableDataset:
    """Standardizes any streaming dataset to strictly yield keys: ['image', 'text']."""
    cols = ds.column_names or []

    if "text" not in cols:
        if "transcription" in cols:
            ds = ds.rename_column("transcription", "text")
        elif "label" in cols:
            ds = ds.rename_column("label", "text")

    cols = ds.column_names or []
    if "image" not in cols:
        if "image_path" in cols:
            ds = ds.rename_column("image_path", "image")
        elif "img" in cols:
            ds = ds.rename_column("img", "image")
        elif "fname" in cols and default_img_dir:
            ds = ds.map(
                lambda x: {
                    "image": os.path.join(default_img_dir, x["fname"]),
                    "text": x.get("text", ""),
                }
            )
        elif "fname" in cols:
            ds = ds.rename_column("fname", "image")

    return ds.select_columns(["image", "text"])


def get_datasets(buffer_size: int = 10000):
    print("Loading datasets in streaming mode...")

    # --- 1. Arabic ---
    sard_raw = cast(
        IterableDataset,
        load_dataset(
            "riotu-lab/SARD", split="Traditional_Arabic", streaming=True
        ),
    )
    ds_arabic = format_dataset_stream(sard_raw)
    print("Finished loading Arabic dataset.")

    # --- 2. Farsi / Persian ---
    parsynth_train_raw = cast(
        IterableDataset,
        load_dataset(
            "hezarai/parsynth-ocr-200k", split="train", streaming=True
        ),
    )
    parsynth_train = format_dataset_stream(parsynth_train_raw)

    parsynth_test_raw = cast(
        IterableDataset,
        load_dataset("hezarai/parsynth-ocr-200k", split="test", streaming=True),
    )
    parsynth_test = format_dataset_stream(parsynth_test_raw)

    persian_train_raw = cast(
        IterableDataset,
        load_dataset(
            "ordaktaktak/Persian-OCR-230k", split="train", streaming=True
        ),
    )
    persian_train = format_dataset_stream(
        persian_train_raw, default_img_dir=IMAGE_BASE_DIR
    )

    persian_test_raw = cast(
        IterableDataset,
        load_dataset(
            "ordaktaktak/Persian-OCR-230k", split="test", streaming=True
        ),
    )
    persian_test = format_dataset_stream(
        persian_test_raw, default_img_dir=IMAGE_BASE_DIR
    )

    persian_train_combined = interleave_datasets(
        [parsynth_train, persian_train],
        probabilities=[0.5, 0.5],
        seed=42,
    )
    print("Finished loading Farsi/Persian datasets.")

    # --- 3. Urdu ---
    nastaliq_raw = cast(
        IterableDataset,
        load_dataset(
            "PuristanLabs1/urdu-ocr-1M",
            "nastaliq",
            split="train",
            streaming=True,
        ),
    )
    nastaliq = format_dataset_stream(nastaliq_raw)

    naskh_raw = cast(
        IterableDataset,
        load_dataset(
            "PuristanLabs1/urdu-ocr-1M",
            "naskh",
            split="train",
            streaming=True,
        ),
    )
    naskh = format_dataset_stream(naskh_raw)

    urdu_news_raw = cast(
        IterableDataset,
        load_dataset(
            "oddadmix/qari-0.2.2-news-dataset-large",
            split="train",
            streaming=True,
        ),
    )
    urdu_news = format_dataset_stream(urdu_news_raw)

    urdu_news_test_raw = cast(
        IterableDataset,
        load_dataset(
            "oddadmix/qari-0.2.2-news-dataset-large",
            split="test",
            streaming=True,
        ),
    )
    urdu_news_test = format_dataset_stream(urdu_news_test_raw)

    urdu_news_val_raw = cast(
        IterableDataset,
        load_dataset(
            "oddadmix/qari-0.2.2-news-dataset-large",
            split="validation",
            streaming=True,
        ),
    )
    urdu_news_val = format_dataset_stream(urdu_news_val_raw)
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
