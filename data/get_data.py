import os
from typing import Tuple, Union

import numpy as np
import torch
import torchvision.io as tv_io
from datasets import (
    IterableDataset,
    IterableDatasetDict,
    interleave_datasets,
    load_dataset,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_BASE_DIR = os.path.join(SCRIPT_DIR, "Persian-OCR-230k", "Images")
if not os.path.exists(IMAGE_BASE_DIR):
    IMAGE_BASE_DIR = os.path.join(SCRIPT_DIR, "Persian-OCR-230k")


def load_as_tensor(
    raw_img: Union[str, torch.Tensor, dict, None], default_dir: str = ""
) -> Union[torch.Tensor, None]:
    """Safely converts any image representation into a 3-channel RGB PyTorch Tensor [3, H, W]."""
    if raw_img is None:
        return None

    if isinstance(raw_img, torch.Tensor):
        if raw_img.ndim == 3 and raw_img.shape[0] != 3 and raw_img.shape[2] == 3:
            return raw_img.permute(2, 0, 1).contiguous()
        return raw_img

    if isinstance(raw_img, str):
        path = (
            os.path.join(default_dir, raw_img)
            if default_dir and not os.path.isabs(raw_img)
            else raw_img
        )
        if os.path.exists(path):
            file_bytes = tv_io.read_file(path)
            return tv_io.decode_image(file_bytes, mode=tv_io.ImageReadMode.RGB)
        return None

    if isinstance(raw_img, dict) and raw_img.get("bytes") is not None:
        byte_tensor = torch.frombuffer(raw_img["bytes"], dtype=torch.uint8)
        return tv_io.decode_image(byte_tensor, mode=tv_io.ImageReadMode.RGB)

    if hasattr(raw_img, "convert"):
        np_arr = np.array(raw_img.convert("RGB"))
        return torch.from_numpy(np_arr).permute(2, 0, 1).contiguous()

    return None


def transform_example(example: dict, default_img_dir: str = "") -> dict:
    """Dynamically converts image paths/objects to PyTorch Tensors and standardizes text fields."""
    raw_img = (
        example.get("image")
        if example.get("image") is not None
        else example.get("image_path")
        if example.get("image_path") is not None
        else example.get("img")
        if example.get("img") is not None
        else example.get("fname")
    )

    image_tensor = load_as_tensor(raw_img, default_dir=default_img_dir)

    text = (
        example.get("text")
        or example.get("transcription")
        or example.get("label")
        or ""
    )

    return {"image": image_tensor, "text": text}


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
