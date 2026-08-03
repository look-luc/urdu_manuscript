import base64
from typing import cast

import datasets
import numpy as np
import torch
import torchvision.io as tv_io
from datasets import IterableDataset, interleave_datasets, load_dataset


def to_torchvision_rgb(example):
    img_data = example.get("image") or example.get("image_path") or example.get("image_base64")
    txt_data = example.get("text") or example.get("markdown") or example.get("chunk")

    if isinstance(img_data, str) and (img_data.startswith("data:image") or "image_base64" in example):
        if "," in img_data:
            img_data = img_data.split(",", 1)[1]
        raw_bytes = base64.b64decode(img_data)
        byte_tensor = torch.frombuffer(bytearray(raw_bytes), dtype=torch.uint8)
        img_tensor = tv_io.decode_image(byte_tensor, mode=tv_io.ImageReadMode.RGB)

    elif isinstance(img_data, dict) and "bytes" in img_data and img_data["bytes"]:
        byte_tensor = torch.frombuffer(bytearray(img_data["bytes"]), dtype=torch.uint8)
        img_tensor = tv_io.decode_image(byte_tensor, mode=tv_io.ImageReadMode.RGB)

    elif isinstance(img_data, str):
        img_tensor = tv_io.read_image(img_data, mode=tv_io.ImageReadMode.RGB)

    elif isinstance(img_data, torch.Tensor):
        img_tensor = img_data

    else:
        try:
            arr = np.asarray(img_data)
            img_tensor = torch.from_numpy(arr)
            if img_tensor.ndim == 3 and img_tensor.shape[-1] in (3, 4):
                if img_tensor.shape[-1] == 4:  # Strip alpha channel if RGBA
                    img_tensor = img_tensor[:, :, :3]
                img_tensor = img_tensor.permute(2, 0, 1)
        except Exception:
            raise ValueError(f"Unsupported image payload type: {type(img_data)}")

    if img_tensor.ndim == 3 and img_tensor.shape[0] in (1, 3):
        img_tensor = img_tensor.permute(1, 2, 0)

    img_numpy = img_tensor.cpu().numpy() if isinstance(img_tensor, torch.Tensor) else np.asarray(img_tensor)

    return {"image": img_numpy, "text": str(txt_data)}


def _prepare_stream(dataset_name: str, split: str, name: str = "") -> IterableDataset:
    """Loads a dataset stream and disables automatic PIL image decoding."""
    kwargs = {"split": split, "streaming": True}
    if name:
        kwargs["name"] = name

    ds = load_dataset(dataset_name, **kwargs)

    # Disables automatic PIL object generation upstream
    if "image" in ds.features:
        ds = ds.cast_column("image", datasets.Image(decode=False))

    return cast(
        IterableDataset,
        ds.map(to_torchvision_rgb).select_columns(["image", "text"])
    )


def get_datasets(buffer_size: int = 100):
    print(f"Loading datasets with torchvision.io pipeline (buffer_size={buffer_size})...")

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
