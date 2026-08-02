import base64
import io
from typing import cast

from datasets import (
    Features,
    Image,
    IterableDataset,
    Value,
    interleave_datasets,
    load_dataset,
)


def get_datasets(buffer_size: int = 100):
    print(f"Loading datasets in streaming mode (buffer_size={buffer_size})...")

    arabic_train = cast(
        IterableDataset,
        load_dataset("MohamedRashad/arabic-img2md", split="train", streaming=True),
    ).rename_column("markdown", "text").select_columns(["image", "text"])
    arabic_test = cast(
        IterableDataset,
        load_dataset("MohamedRashad/arabic-img2md", split="test", streaming=True),
    ).rename_column("markdown", "text").select_columns(["image", "text"])

    parsynth_train_raw = cast(
        IterableDataset,
        load_dataset("hezarai/parsynth-ocr-200k", split="train", streaming=True),
    ).rename_column("image_path", "image").select_columns(["image", "text"])

    parsynth_test_raw = cast(
        IterableDataset,
        load_dataset("hezarai/parsynth-ocr-200k", split="test", streaming=True),
    ).rename_column("image_path", "image").select_columns(["image", "text"])

    persian_pixel_raw = cast(
        IterableDataset,
        load_dataset("Omarrran/Persian_Pixel", name="full", split="train", streaming=True),
    ).select_columns(["image", "text"])

    nastaliq_raw = cast(
        IterableDataset,
        load_dataset("PuristanLabs1/urdu-ocr-1M", "nastaliq", split="train", streaming=True),
    ).select_columns(["image", "text"])

    naskh_raw = cast(
        IterableDataset,
        load_dataset("PuristanLabs1/urdu-ocr-1M", "naskh", split="train", streaming=True),
    ).select_columns(["image", "text"])

    urdu_news_train_raw = cast(
        IterableDataset,
        load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="train", streaming=True),
    ).select_columns(["image", "text"])

    urdu_news_test_raw = cast(
        IterableDataset,
        load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="test", streaming=True),
    ).select_columns(["image", "text"])

    # Interleave Urdu sub-streams
    urdu_ds_train = interleave_datasets(
        [nastaliq_raw.skip(1000), naskh_raw.skip(1000), urdu_news_train_raw],
        seed=42,
    )
    urdu_ds_test = interleave_datasets(
        [nastaliq_raw.take(1000), naskh_raw.take(1000), urdu_news_test_raw],
        seed=42,
    )

    # Combine all test sources
    test_sources = [
        arabic_test,
        parsynth_test,
        persian_pixel.take(100000),
        urdu_ds_test,
    ]
    test_dataset = interleave_datasets(test_sources, seed=42)

    train_sources = [
        arabic_train,
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
