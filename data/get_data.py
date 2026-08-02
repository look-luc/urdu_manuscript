import os
from typing import Tuple, Union

from datasets import (
    IterableDataset,
    IterableDatasetDict,
    interleave_datasets,
    load_dataset,
)
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_BASE_DIR = os.path.join(SCRIPT_DIR, "Persian-OCR-230k", "Images")
if not os.path.exists(IMAGE_BASE_DIR):
    IMAGE_BASE_DIR = os.path.join(SCRIPT_DIR, "Persian-OCR-230k")


def transform_example(example: dict, default_img_dir: str = None) -> dict:
    """Dynamically converts image paths/objects to PIL Images and standardizes text fields."""
    image = None

    # 1. Resolve Image
    if "image" in example and example["image"] is not None:
        image = example["image"]
    elif "image_path" in example and example["image_path"] is not None:
        image = Image.open(example["image_path"]).convert("RGB")
    elif "img" in example and example["img"] is not None:
        image = example["img"]
    elif "fname" in example and example["fname"] is not None:
        img_path = (
            os.path.join(default_img_dir, example["fname"])
            if default_img_dir
            else example["fname"]
        )
        image = Image.open(img_path).convert("RGB")

    # If image is a string path that wasn't decoded yet
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    # 2. Resolve Text Column
    text = (
        example.get("text")
        or example.get("transcription")
        or example.get("label")
        or ""
    )

    return {"image": image, "text": text}


def get_streaming_split_pair(
    ds_obj: Union[IterableDatasetDict, IterableDataset],
    test_count: int = 1000,
) -> Tuple[IterableDataset, IterableDataset]:
    """Extracts train and test streams lazily using take/skip if test split does not exist."""
    if isinstance(ds_obj, IterableDatasetDict) or hasattr(ds_obj, "keys"):
        if "test" in ds_obj:
            return ds_obj["train"], ds_obj["test"]
        elif "validation" in ds_obj:
            return ds_obj["train"], ds_obj["validation"]
        elif "val" in ds_obj:
            return ds_obj["train"], ds_obj["val"]
        else:
            train_stream = ds_obj["train"]
            return train_stream.skip(test_count), train_stream.take(test_count)
    else:
        return ds_obj.skip(test_count), ds_obj.take(test_count)


def get_datasets():
    print("Loading streaming datasets...")

    # --- 1. Arabic ---
    sard_raw = load_dataset(
        "riotu-lab/SARD", split="Traditional_Arabic", streaming=True
    )
    arabic_train = sard_raw.skip(1000).map(transform_example)
    arabic_test = sard_raw.take(1000).map(transform_example)

    # --- 2. Farsi / Persian ---
    parsynth_raw = load_dataset("hezarai/parsynth-ocr-200k", streaming=True)
    parsynth_tr_stream, parsynth_te_stream = get_streaming_split_pair(
        parsynth_raw, test_count=1000
    )
    parsynth_train = parsynth_tr_stream.map(transform_example)
    parsynth_test = parsynth_te_stream.map(transform_example)

    persian_230k_raw = load_dataset("ordaktaktak/Persian-OCR-230k", streaming=True)
    persian_tr_stream, persian_te_stream = get_streaming_split_pair(
        persian_230k_raw, test_count=1000
    )
    persian_train = persian_tr_stream.map(
        lambda x: transform_example(x, default_img_dir=IMAGE_BASE_DIR)
    )
    persian_test = persian_te_stream.map(
        lambda x: transform_example(x, default_img_dir=IMAGE_BASE_DIR)
    )

    # Combine Farsi sub-streams lazily
    farsi_train = interleave_datasets([parsynth_train, persian_train])
    farsi_test = interleave_datasets([parsynth_test, persian_test])

    # --- 3. Urdu ---
    nastaliq_ds = load_dataset(
        "PuristanLabs1/urdu-ocr-1M", "nastaliq", streaming=True
    )
    naskh_ds = load_dataset("PuristanLabs1/urdu-ocr-1M", "naskh", streaming=True)
    urdu_news_ds = load_dataset(
        "oddadmix/qari-0.2.2-news-dataset-large", streaming=True
    )

    nas_tr, nas_te = get_streaming_split_pair(nastaliq_ds, test_count=1000)
    naskh_tr, naskh_te = get_streaming_split_pair(naskh_ds, test_count=1000)
    news_tr, news_te = get_streaming_split_pair(urdu_news_ds, test_count=1000)

    urdu_train = interleave_datasets([
        nas_tr.map(transform_example),
        naskh_tr.map(transform_example),
        news_tr.map(transform_example),
    ])

    urdu_test = interleave_datasets([
        nas_te.map(transform_example),
        naskh_te.map(transform_example),
        news_te.map(transform_example),
    ])

    train_sources = [arabic_train, farsi_train, urdu_train]
    test_sources = [arabic_test, farsi_test, urdu_test]

    train_probabilities = [0.20, 0.30, 0.50]
    test_probabilities = [0.20, 0.30, 0.50]

    train_dataset = interleave_datasets(
        datasets=train_sources,
        probabilities=train_probabilities,
        seed=42,
    ).shuffle(seed=42, buffer_size=10000)

    test_dataset = interleave_datasets(
        datasets=test_sources,
        probabilities=test_probabilities,
        seed=42,
    )

    return {"train": train_dataset, "test": test_dataset}
