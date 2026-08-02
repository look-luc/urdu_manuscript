from typing import cast

from datasets import IterableDataset, interleave_datasets, load_dataset


def standardize_stream(
    ds: IterableDataset, img_col: str = "image", txt_col: str = "text"
) -> IterableDataset:
    """Safely renames columns and filters the stream down to ['image', 'text']."""
    cols = list(ds.features.keys()) if ds.features is not None else (ds.column_names or [])

    if img_col in cols and img_col != "image":
        ds = ds.rename_column(img_col, "image")
    if txt_col in cols and txt_col != "text":
        ds = ds.rename_column(txt_col, "text")

    return ds.select_columns(["image", "text"])


def get_datasets(buffer_size: int = 100):
    print(f"Loading datasets in streaming mode (buffer_size={buffer_size})...")

    # --- 1. Arabic ---
    sard_raw = cast(
        IterableDataset,
        load_dataset("riotu-lab/SARD", split="Traditional_Arabic", streaming=True),
    )
    ds_arabic = standardize_stream(sard_raw, img_col="image", txt_col="label")

    # Use take/skip instead of train_test_split
    ds_arabic_test = ds_arabic.take(1000)
    ds_arabic_train = ds_arabic.skip(1000)

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

    persian_pixel = cast(IterableDataset, load_dataset(
        "Omarrran/Persian_Pixel",
        name="full",
        split="train",
        streaming=True,
    ))

    persian_pixel = standardize_stream(persian_pixel, img_col="image", txt_col="text")

    # --- 3. Urdu Datasets ---
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

    urdu_news_train_raw = cast(
        IterableDataset,
        load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="train", streaming=True),
    )
    urdu_news_train = standardize_stream(urdu_news_train_raw, img_col="image", txt_col="text")

    urdu_news_test_raw = cast(
        IterableDataset,
        load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="test", streaming=True),
    )
    urdu_news_test = standardize_stream(urdu_news_test_raw, img_col="image", txt_col="text")

    # Interleave Urdu training streams (replaces concatenate_datasets)
    urdu_ds_train = interleave_datasets(
        [nastaliq.skip(1000), naskh.skip(1000), urdu_news_train],
        seed=42,
    )

    # Interleave Urdu test streams
    urdu_ds_test = interleave_datasets(
        [nastaliq.take(1000), naskh.take(1000), urdu_news_test],
        seed=42,
    )

    # --- Combine All Sources ---
    test_sources = [
        ds_arabic_test,
        parsynth_test,
        persian_pixel.take(100000),
        urdu_ds_test,
    ]
    test_dataset = interleave_datasets(test_sources, seed=42)

    train_sources = [
        ds_arabic_train,
        parsynth_train,
        persian_pixel.skip(100000),
        urdu_ds_train,
    ]
    train_probabilities = [0.15, 0.175, 0.175, 0.50]

    train_dataset = interleave_datasets(
        datasets=train_sources,
        probabilities=train_probabilities,
        stopping_strategy="all_exhausted",
        seed=42,
    ).shuffle(seed=42, buffer_size=buffer_size)

    return {"train": train_dataset, "test": test_dataset}
