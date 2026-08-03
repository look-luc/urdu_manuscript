import base64
from typing import cast

import torch
import torchvision.io as tv_io
from datasets import IterableDataset, interleave_datasets, load_dataset


def to_torchvision_rgb(example):
    """Decodes image sources (base64, raw bytes, file paths) to RGB torch.Tensor using torchvision.io."""
    img_data = example.get("image") or example.get("image_path") or example.get("image_base64")
    txt_data = example.get("text") or example.get("markdown") or example.get("chunk")

    if isinstance(img_data, str) and (img_data.startswith("data:image") or "image_base64" in example):
        if "," in img_data:
            img_data = img_data.split(",", 1)[1]
        raw_bytes = base64.b64decode(img_data)
        byte_tensor = torch.frombuffer(raw_bytes, dtype=torch.uint8)
        img_tensor = tv_io.decode_image(byte_tensor, mode=tv_io.ImageReadMode.RGB)

    elif isinstance(img_data, dict) and "bytes" in img_data and img_data["bytes"]:
        byte_tensor = torch.frombuffer(img_data["bytes"], dtype=torch.uint8)
        img_tensor = tv_io.decode_image(byte_tensor, mode=tv_io.ImageReadMode.RGB)

    elif isinstance(img_data, str):
        img_tensor = tv_io.read_image(img_data, mode=tv_io.ImageReadMode.RGB)

    elif isinstance(img_data, torch.Tensor):
        img_tensor = img_data

    elif hasattr(img_data, "__array__"):
        import numpy as np
        arr = np.asarray(img_data)
        img_tensor = torch.from_numpy(arr)
        if img_tensor.ndim == 3 and img_tensor.shape[-1] == 3:
            img_tensor = img_tensor.permute(2, 0, 1)

    else:
        raise ValueError(f"Unsupported image payload type: {type(img_data)}")

    img_tensor = img_tensor.permute(1, 2, 0)

    return {"image": img_tensor, "text": str(txt_data)}


def get_datasets(buffer_size: int = 100):
    print(f"Loading datasets with torchvision.io pipeline (buffer_size={buffer_size})...")

    arabic_train = cast(
        IterableDataset,
        load_dataset("MohamedRashad/arabic-img2md", split="train", streaming=True),
    ).map(to_torchvision_rgb).select_columns(["image", "text"])

    arabic_test = cast(
        IterableDataset,
        load_dataset("MohamedRashad/arabic-img2md", split="test", streaming=True),
    ).map(to_torchvision_rgb).select_columns(["image", "text"])

    parsynth_train = cast(
        IterableDataset,
        load_dataset("hezarai/parsynth-ocr-200k", split="train", streaming=True),
    ).map(to_torchvision_rgb).select_columns(["image", "text"])

    parsynth_test = cast(
        IterableDataset,
        load_dataset("hezarai/parsynth-ocr-200k", split="test", streaming=True),
    ).map(to_torchvision_rgb).select_columns(["image", "text"])

    persian_pixel = cast(
        IterableDataset,
        load_dataset("Omarrran/Persian_Pixel", name="full", split="train", streaming=True),
    ).map(to_torchvision_rgb).select_columns(["image", "text"])

    nastaliq_raw = cast(
        IterableDataset,
        load_dataset("PuristanLabs1/urdu-ocr-1M", "nastaliq", split="train", streaming=True),
    ).map(to_torchvision_rgb).select_columns(["image", "text"])

    naskh_raw = cast(
        IterableDataset,
        load_dataset("PuristanLabs1/urdu-ocr-1M", "naskh", split="train", streaming=True),
    ).map(to_torchvision_rgb).select_columns(["image", "text"])

    urdu_news_train = cast(
        IterableDataset,
        load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="train", streaming=True),
    ).map(to_torchvision_rgb).select_columns(["image", "text"])

    urdu_news_test = cast(
        IterableDataset,
        load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="test", streaming=True),
    ).map(to_torchvision_rgb).select_columns(["image", "text"])

    # Interleave sub-streams cleanly
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
