from typing import cast

from datasets import (
    Dataset,
    Image,
    concatenate_datasets,
    interleave_datasets,
    load_dataset,
)


def force_schema_and_select(ds: Dataset) -> Dataset:
    """Ensures consistent column names, casts image types, and filters to ['image', 'text']."""
    cols = ds.column_names or []

    # 1. Standardize Image Column
    if "image_path" in cols:
        ds = ds.rename_column("image_path", "image")
    elif "img" in cols:
        ds = ds.rename_column("img", "image")

    # 2. Standardize Text Column
    if "transcription" in cols:
        ds = ds.rename_column("transcription", "text")
    elif "label" in cols:
        ds = ds.rename_column("label", "text")

    # 3. Cast image feature type
    ds = ds.cast_column("image", Image())

    # 4. Filter strictly to target schema
    return ds.select_columns(["image", "text"])


def get_datasets():
    print("Loading datasets...")

    # --- 1. Arabic ---
    # Loaded without streaming to match other map-style datasets
    sard_raw = cast(Dataset, load_dataset("riotu-lab/SARD", split="Traditional_Arabic"))
    sard_ds = force_schema_and_select(sard_raw)
    sard_split = sard_ds.train_test_split(test_size=0.2, seed=42)
    arabic_train, arabic_test = sard_split["train"], sard_split["test"]

    # --- 2. Farsi/Persian ---
    parsynth_train_raw = cast(Dataset, load_dataset("hezarai/parsynth-ocr-200k", split="train"))
    parsynth_test_raw = cast(Dataset, load_dataset("hezarai/parsynth-ocr-200k", split="test"))

    farsi_train = force_schema_and_select(parsynth_train_raw)
    farsi_test = force_schema_and_select(parsynth_test_raw)

    # --- 3. Urdu ---
    nastaliq_ds = load_dataset("PuristanLabs1/urdu-ocr-1M", "nastaliq")
    naskh_ds = load_dataset("PuristanLabs1/urdu-ocr-1M", "naskh")
    urdu_news_ds = load_dataset("oddadmix/qari-0.2.2-news-dataset-large")

    urdu_train = concatenate_datasets([
        force_schema_and_select(nastaliq_ds["train"]),
        force_schema_and_select(naskh_ds["train"]),
        force_schema_and_select(urdu_news_ds["train"]),
    ])

    urdu_test = concatenate_datasets([
        force_schema_and_select(nastaliq_ds["test"]),
        force_schema_and_select(naskh_ds["test"]),
        force_schema_and_select(urdu_news_ds["test"]),
    ])

    # --- 4. Interleave with Probabilities ---
    train_sources = [arabic_train, farsi_train, urdu_train]
    test_sources = [arabic_test, farsi_test, urdu_test]

    # Define sampling probabilities across [arabic, farsi, urdu]
    train_probabilities = [0.20, 0.30, 0.50]
    test_probabilities = [0.20, 0.30, 0.50]

    train_dataset = interleave_datasets(
        datasets=train_sources,
        probabilities=train_probabilities,
        seed=42,
    ).shuffle(seed=42)

    test_dataset = interleave_datasets(
        datasets=test_sources,
        probabilities=test_probabilities,
        seed=42,
    )

    return {"train": train_dataset, "test": test_dataset}
