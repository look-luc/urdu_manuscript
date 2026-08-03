import base64
from typing import cast

import datasets
import numpy as np
import torch
import torchvision.io as tv_io
from datasets import IterableDataset, interleave_datasets, load_dataset


def to_raw_bytes(example):
    """Normalizes any image source to raw 1D binary bytes for PyArrow streaming compatibility without PIL."""
    img_data = example.get("image") or example.get("image_path") or example.get("image_base64")
    txt_data = example.get("text") or example.get("markdown") or example.get("chunk")

    raw_bytes = None

    if isinstance(img_data, str) and (img_data.startswith("data:image") or "image_base64" in example):
        if "," in img_data:
            img_data = img_data.split(",", 1)[1]
        raw_bytes = base64.b64decode(img_data)

    elif isinstance(img_data, dict) and "bytes" in img_data and img_data["bytes"]:
        raw_bytes = img_data["bytes"]

    elif isinstance(img_data, str):
        with open(img_data, "rb") as f:
            raw_bytes = f.read()

    elif isinstance(img_data, (torch.Tensor, np.ndarray)):
        img_tensor = torch.from_numpy(img_data) if isinstance(img_data, np.ndarray) else img_data
        if img_tensor.ndim == 3 and img_tensor.shape[-1] in (3, 4):
            if img_tensor.shape[-1] == 4:  # Strip alpha channel if RGBA
                img_tensor = img_tensor[:, :, :3]
            img_tensor = img_tensor.permute(2, 0, 1)

        if img_tensor.dtype != torch.uint8:
            img_tensor = img_tensor.to(torch.uint8)

        # Encode tensor directly to PNG binary bytes using torchvision
        raw_bytes = bytes(tv_io.encode_png(img_tensor.cpu()).numpy())

    if raw_bytes is None:
        raise ValueError(f"Unable to extract raw image bytes from payload type: {type(img_data)}")

    return {"image_bytes": raw_bytes, "text": str(txt_data)}


def _prepare_stream(dataset_name: str, split: str, name: str = "") -> IterableDataset:
    """Loads a dataset stream and casts image features to raw byte vectors."""
    kwargs = {"split": split, "streaming": True}
    if name:
        kwargs["name"] = name

    ds = load_dataset(dataset_name, **kwargs)

    # Disable automatic PIL object generation upstream
    if "image" in ds.features:
        ds = ds.cast_column("image", datasets.Image(decode=False))

    return cast(
        IterableDataset,
        ds.map(to_raw_bytes).select_columns(["image_bytes", "text"])
    )


def get_datasets(buffer_size: int = 100):
    print(f"Loading datasets with pure torchvision byte pipeline (buffer_size={buffer_size})...")

    arabic_train = _prepare_stream("MohamedRashad/arabic-img2md", split="train")
    arabic_test = _prepare_stream("MohamedRashad/arabic-img2md", split="test")

    parsynth_train = _prepare_stream("hezarai/parsynth-ocr-200k", split="train")
    parsynth_test = _prepare_stream("hezarai/parsynth-ocr-200k", split="test")

    persian_pixel = _prepare_stream("Omarrran/Persian_Pixel", name="full", split="train")

    nastaliq_raw = _prepare_stream("PuristanLabs1/urdu-ocr-1M", name="nastaliq", split="train")
    naskh_raw = _prepare_stream("PuristanLabs1/urdu-ocr-1M", name="naskh", split="train")

    urdu_news_train = _prepare_stream("oddadmix/qari-0.2.2-news-dataset-large", split="train")
    urdu_news_test = _prepare_stream("oddadmix/qari-0.2.2-news-dataset-large", split="test")

    urdu_ds_train = interleave_datasets(
        [nastaliq_raw.skip(1000), naskh_raw.skip(1000), urdu_news_train],
        seed=42,
    )
    urdu_ds_test = interleave_datasets(
        [nastaliq_raw.take(1000), naskh_raw.take(1000), urdu_news_test],
        seed=42,
    )

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
    train_dataset = interleave_datasets(
        datasets=train_sources,
        probabilities=[0.15, 0.175, 0.175, 0.50],
        stopping_strategy="all_exhausted",
        seed=42,
    ).shuffle(seed=42, buffer_size=buffer_size)

    return {"train": train_dataset, "test": test_dataset}
