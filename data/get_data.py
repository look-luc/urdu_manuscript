import os
from typing import cast

import torchvision.transforms.functional as F
from datasets import (
    Features,
    IterableDataset,
    Value,
    interleave_datasets,
    load_dataset,
)
from datasets import (
    Image as HFImage,
)
from torchvision.io import ImageReadMode, read_image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_BASE_DIR = os.path.join(SCRIPT_DIR, "Persian-OCR-230k")


def fix_persian_image_path(example):
    if not isinstance(example, dict):
        return {"fname": None, "text": None}

    fname = example.get("fname")
    if isinstance(fname, str):
        full_path = os.path.join(IMAGE_BASE_DIR, fname)
        if os.path.exists(full_path):
            try:
                example["fname"] = F.to_pil_image(
                    read_image(full_path, mode=ImageReadMode.RGB)
                )
                return example
            except Exception:
                pass

    example["fname"] = None
    return example


def fix_parsynth_image_path(example):
    if not isinstance(example, dict):
        return {"image_path": None, "text": None}

    img_path = example.get("image_path")
    if isinstance(img_path, str):
        full_path = os.path.join(SCRIPT_DIR, img_path)
        if os.path.exists(full_path):
            try:
                example["image_path"] = F.to_pil_image(
                    read_image(full_path, mode=ImageReadMode.RGB)
                )
                return example
            except Exception:
                pass

    example["image_path"] = None
    return example


def is_valid_example(example):
    """Filters out corrupt, missing, or empty image and text samples."""
    if not isinstance(example, dict):
        return False

    img = example.get("image")
    txt = example.get("text")

    if img is None or txt is None:
        return False
    if isinstance(txt, str) and not txt.strip():
        return False

    return True


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

    ds_arabic = sard_raw.select_columns(["image", "text"]).filter(is_valid_example)
    print("Finished loading Arabic dataset.")

    # --- 2. Farsi / Persian ---
    parsynth_features = Features({
        "image_path": HFImage(),
        "text": Value("string"),
    })

    parsynth_train_raw = cast(
        IterableDataset,
        load_dataset("hezarai/parsynth-ocr-200k", split="train", streaming=True),
    )
    parsynth_train_raw.info.features = parsynth_features

    parsynth_train = (
        parsynth_train_raw
        .map(fix_parsynth_image_path)
        .rename_column("image_path", "image")
        .select_columns(["image", "text"])
        .filter(is_valid_example)
    )

    parsynth_test_raw = cast(
        IterableDataset,
        load_dataset("hezarai/parsynth-ocr-200k", split="test", streaming=True),
    )
    parsynth_test_raw.info.features = parsynth_features

    parsynth_test = (
        parsynth_test_raw
        .map(fix_parsynth_image_path)
        .rename_column("image_path", "image")
        .select_columns(["image", "text"])
        .filter(is_valid_example)
    )

    persian_features = Features({
        "fname": HFImage(),
        "text": Value("string"),
    })

    persian_train_raw = cast(
        IterableDataset,
        load_dataset("ordaktaktak/Persian-OCR-230k", split="train", streaming=True),
    )
    persian_train_raw.info.features = persian_features

    persian_train = (
        persian_train_raw
        .map(fix_persian_image_path)
        .rename_column("fname", "image")
        .select_columns(["image", "text"])
        .filter(is_valid_example)
    )

    persian_test_raw = cast(
        IterableDataset,
        load_dataset("ordaktaktak/Persian-OCR-230k", split="test", streaming=True),
    )
    persian_test_raw.info.features = persian_features

    persian_test = (
        persian_test_raw
        .map(fix_persian_image_path)
        .rename_column("fname", "image")
        .select_columns(["image", "text"])
        .filter(is_valid_example)
    )

    persian_train_combined = interleave_datasets(
        [parsynth_train, persian_train],
        probabilities=[0.5, 0.5],
        seed=42,
    )
    print("Finished loading Farsi/Persian datasets.")

    # --- 3. Urdu ---
    nastaliq = (
        cast(
            IterableDataset,
            load_dataset("PuristanLabs1/urdu-ocr-1M", "nastaliq", split="train", streaming=True),
        )
        .select_columns(["image", "text"])
        .filter(is_valid_example)
    )

    naskh = (
        cast(
            IterableDataset,
            load_dataset("PuristanLabs1/urdu-ocr-1M", "naskh", split="train", streaming=True),
        )
        .select_columns(["image", "text"])
        .filter(is_valid_example)
    )

    urdu_news = (
        cast(
            IterableDataset,
            load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="train", streaming=True),
        )
        .select_columns(["image", "text"])
        .filter(is_valid_example)
    )

    urdu_news_test = (
        cast(
            IterableDataset,
            load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="test", streaming=True),
        )
        .select_columns(["image", "text"])
        .filter(is_valid_example)
    )

    urdu_news_val = (
        cast(
            IterableDataset,
            load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="validation", streaming=True),
        )
        .select_columns(["image", "text"])
        .filter(is_valid_example)
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
