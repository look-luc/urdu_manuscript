import os
from typing import cast

from datasets import Dataset, interleave_datasets, load_dataset
from datasets import Image as HFImage

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_BASE_DIR = os.path.join(SCRIPT_DIR, "Persian-OCR-230k")


def fix_persian_image_path(example):
    """Safely resolves relative Persian image paths and checks for existence on disk."""
    fname = example.get("fname")
    if isinstance(fname, str):
        full_path = os.path.join(IMAGE_BASE_DIR, fname)
        if os.path.exists(full_path):
            example["fname"] = full_path
        else:
            example["fname"] = None
    else:
        example["fname"] = None
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


def get_datasets(num_proc: int = 4):
    print("Loading datasets in non-streaming mode (downloading/caching locally)...")

    # --- 1. Arabic ---
    sard_raw = cast(Dataset, load_dataset("riotu-lab/SARD", split="Traditional_Arabic"))
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
        .filter(is_valid_example, num_proc=num_proc)
    )
    print("Finished loading Arabic dataset.")

    # --- 2. Farsi / Persian ---
    parsynth_train = (
        cast(Dataset, load_dataset("hezarai/parsynth-ocr-200k", split="train"))
        .rename_column("image_path", "image")
        .select_columns(["image", "text"])
        .filter(is_valid_example, num_proc=num_proc)
    )

    parsynth_test = (
        cast(Dataset, load_dataset("hezarai/parsynth-ocr-200k", split="test"))
        .rename_column("image_path", "image")
        .select_columns(["image", "text"])
        .filter(is_valid_example, num_proc=num_proc)
    )

    persian_train = (
        cast(Dataset, load_dataset("ordaktaktak/Persian-OCR-230k", split="train"))
        .map(fix_persian_image_path, num_proc=num_proc)
        .filter(lambda x: x.get("fname") is not None, num_proc=num_proc)
        .rename_column("fname", "image")
        .cast_column("image", HFImage())
        .select_columns(["image", "text"])
        .filter(is_valid_example, num_proc=num_proc)
    )

    persian_test = (
        cast(Dataset, load_dataset("ordaktaktak/Persian-OCR-230k", split="test"))
        .map(fix_persian_image_path, num_proc=num_proc)
        .filter(lambda x: x.get("fname") is not None, num_proc=num_proc)
        .rename_column("fname", "image")
        .cast_column("image", HFImage())
        .select_columns(["image", "text"])
        .filter(is_valid_example, num_proc=num_proc)
    )

    persian_train_combined = interleave_datasets(
        [parsynth_train, persian_train],
        probabilities=[0.5, 0.5],
        seed=42,
    )
    print("Finished loading Farsi/Persian datasets.")

    # --- 3. Urdu ---
    nastaliq = (
        cast(Dataset, load_dataset("PuristanLabs1/urdu-ocr-1M", "nastaliq", split="train"))
        .select_columns(["image", "text"])
        .filter(is_valid_example, num_proc=num_proc)
    )

    naskh = (
        cast(Dataset, load_dataset("PuristanLabs1/urdu-ocr-1M", "naskh", split="train"))
        .select_columns(["image", "text"])
        .filter(is_valid_example, num_proc=num_proc)
    )

    urdu_news = (
        cast(Dataset, load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="train"))
        .select_columns(["image", "text"])
        .filter(is_valid_example, num_proc=num_proc)
    )

    urdu_news_test = (
        cast(Dataset, load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="test"))
        .select_columns(["image", "text"])
        .filter(is_valid_example, num_proc=num_proc)
    )

    urdu_news_val = (
        cast(Dataset, load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="validation"))
        .select_columns(["image", "text"])
        .filter(is_valid_example, num_proc=num_proc)
    )
    print("Finished loading Urdu datasets.")

    # --- 4. Test Dataset Assembly ---
    test_sources = [
        ds_arabic.select(range(min(600, len(ds_arabic)))),
        nastaliq.select(range(min(800, len(nastaliq)))),
        naskh.select(range(min(800, len(naskh)))),
        urdu_news_test,
        parsynth_test,
        persian_test,
        urdu_news_val,
    ]

    test_dataset = interleave_datasets(test_sources, seed=42)

    # --- 5. Train Dataset Assembly ---
    train_sources = [
        nastaliq.select(range(min(800, len(nastaliq)), len(nastaliq))),
        naskh.select(range(min(800, len(naskh)), len(naskh))),
        ds_arabic.select(range(min(600, len(ds_arabic)), len(ds_arabic))),
        persian_train_combined,
        urdu_news,
    ]

    train_probabilities = [0.50, 0.20, 0.15, 0.10, 0.05]

    train_dataset = interleave_datasets(
        datasets=train_sources,
        probabilities=train_probabilities,
        stopping_strategy="all_exhausted",
        seed=42,
    ).shuffle(seed=42)

    return {"train": train_dataset, "test": test_dataset}
