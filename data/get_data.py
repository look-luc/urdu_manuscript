from typing import cast

from datasets import (
    Features,
    Image,
    IterableDataset,
    Value,
    interleave_datasets,
    load_dataset,
)


def standardize_stream(
    ds: IterableDataset, img_col: str = "image", txt_col: str = "text"
) -> IterableDataset:
    candidate_img_cols = [img_col, "image", "image_path", "img", "file_name"]
    candidate_txt_cols = [txt_col, "text", "label", "caption", "transcription"]

    def transform_fn(example):
        # Resolve image key dynamically
        actual_img_col = next((k for k in candidate_img_cols if k in example), None)
        actual_txt_col = next((k for k in candidate_txt_cols if k in example), None)

        if actual_img_col is None:
            raise KeyError(f"None of {candidate_img_cols} found in sample keys: {list(example.keys())}")
        if actual_txt_col is None:
            raise KeyError(f"None of {candidate_txt_cols} found in sample keys: {list(example.keys())}")

        return {
            "image": example[actual_img_col],
            "text": str(example[actual_txt_col]),
        }

    ds = ds.map(transform_fn)
    ds = ds.select_columns(["image", "text"])

    target_features = Features({"image": Image(), "text": Value("string")})
    return ds.cast(target_features)


def get_datasets(buffer_size: int = 100):
    print(f"Loading datasets in streaming mode (buffer_size={buffer_size})...")

    # --- 1. Arabic ---
    sard_raw = cast(
        IterableDataset,
        load_dataset("riotu-lab/SARD", split="Traditional_Arabic", streaming=True),
    )
    ds_arabic = standardize_stream(sard_raw, img_col="image", txt_col="label")

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

    persian_pixel_raw = cast(
        IterableDataset,
        load_dataset("Omarrran/Persian_Pixel", name="full", split="train", streaming=True),
    )

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

    # Interleave Urdu training streams
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
