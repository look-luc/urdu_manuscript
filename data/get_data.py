import os
from typing import cast

import torch
import torchvision.io as torchvision_io
from datasets import Dataset, IterableDataset, interleave_datasets, load_dataset
from datasets import Image as HFImage
from torchvision.io import ImageReadMode

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_BASE_DIR = os.path.join(SCRIPT_DIR, "Persian-OCR-230k")


def resolve_path(example):
    image_val = example.get("image")
    if isinstance(image_val, dict) and "path" in image_val:
        path = image_val["path"]
        if path and not path.startswith(("http://", "https://")) and not path.startswith(IMAGE_BASE_DIR):
            example["image"]["path"] = os.path.join(IMAGE_BASE_DIR, path)
    elif isinstance(image_val, str):
        if not image_val.startswith(("http://", "https://")) and not image_val.startswith(IMAGE_BASE_DIR):
            example["image"] = os.path.join(IMAGE_BASE_DIR, image_val)
    return example


def _get_image_dimensions(raw_img) -> tuple[int, int] | None:
    """Extracts (width, height) using torchvision instead of PIL."""
    try:
        if isinstance(raw_img, dict):
            if "bytes" in raw_img and raw_img["bytes"]:
                byte_tensor = torch.frombuffer(
                    bytearray(raw_img["bytes"]), dtype=torch.uint8
                )
                tensor_img = torchvision_io.decode_image(byte_tensor, mode=ImageReadMode.RGB)
                return tensor_img.shape[2], tensor_img.shape[1]  # (W, H)
            elif "path" in raw_img and raw_img["path"]:
                path_str = str(raw_img["path"])
                if os.path.exists(path_str):
                    tensor_img = torchvision_io.read_image(path_str, mode=ImageReadMode.RGB)
                    return tensor_img.shape[2], tensor_img.shape[1]  # (W, H)

        elif isinstance(raw_img, str):
            if os.path.exists(raw_img):
                tensor_img = torchvision_io.read_image(raw_img, mode=ImageReadMode.RGB)
                return tensor_img.shape[2], tensor_img.shape[1]  # (W, H)

    except Exception:
        pass
    return None


def is_valid_example(example, min_pixels: int = 3136) -> bool:
    img = example.get("image")
    txt = example.get("text")
    if img is None or txt is None:
        return False
    if isinstance(txt, str) and not txt.strip():
        return False

    dims = _get_image_dimensions(img)
    if dims is not None:
        w, h = dims
        if w == 0 or h == 0:
            return False

        aspect_ratio = w / h
        max_safe_ratio = min_pixels / (28 * 28)
        min_safe_ratio = (28 * 28) / min_pixels

        if aspect_ratio > (max_safe_ratio * 10) or aspect_ratio < (min_safe_ratio / 10):
            return False

    return True


def prepare_dataset(ds: Dataset, select_cols=True) -> IterableDataset:
    if select_cols:
        ds = ds.select_columns(["image", "text"])

    # Enforce decode=False so HF datasets outputs {'bytes': ..., 'path': ...} dicts instead of PIL objects
    ds = ds.cast_column("image", HFImage(decode=False))

    iterable_ds = ds.to_iterable_dataset()
    return iterable_ds.filter(is_valid_example)


def get_datasets():
    # --- Arabic ---
    ds_arabic = prepare_dataset(cast(Dataset, load_dataset("mssqpi/Arabic-OCR-Dataset", split="train")))

    # --- Farsi ---
    parsynth_train = prepare_dataset(
        cast(Dataset, load_dataset("hezarai/parsynth-ocr-200k", split="train")).rename_column("image_path", "image")
    )
    parsynth_test = prepare_dataset(
        cast(Dataset, load_dataset("hezarai/parsynth-ocr-200k", split="test")).rename_column("image_path", "image")
    )

    # --- Persian ---
    persian_dict = load_dataset("ordaktaktak/Persian-OCR-230k")
    persian_train = prepare_dataset(cast(Dataset, persian_dict["train"]).rename_column("fname", "image").map(resolve_path))
    persian_test = prepare_dataset(cast(Dataset, persian_dict["test"]).rename_column("fname", "image").map(resolve_path))

    # --- Urdu ---
    nastaliq = prepare_dataset(cast(Dataset, load_dataset("PuristanLabs1/urdu-ocr-1M", "nastaliq", split="train")))
    naskh = prepare_dataset(cast(Dataset, load_dataset("PuristanLabs1/urdu-ocr-1M", "naskh", split="train")))
    urdu_news = prepare_dataset(cast(Dataset, load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="train")))
    urdu_news_test = prepare_dataset(cast(Dataset, load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="test")))
    urdu_news_val = prepare_dataset(cast(Dataset, load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="validation")))

    # --- Kannada ---
    kannada_train = prepare_dataset(cast(Dataset, load_dataset("darknight054/indic-mozhi-ocr", "kannada", split="train")))
    kannada_val = prepare_dataset(cast(Dataset, load_dataset("darknight054/indic-mozhi-ocr", "kannada", split="validation")))
    kannada_test = prepare_dataset(cast(Dataset, load_dataset("darknight054/indic-mozhi-ocr", "kannada", split="test")))
    kannada_df_test = interleave_datasets([kannada_val, kannada_test])

    test_dataset = interleave_datasets(
        [
            ds_arabic.take(600),
            nastaliq.take(600),
            naskh.take(600),
            urdu_news_test,
            parsynth_test,
            persian_test,
            urdu_news_val,
            kannada_df_test,
        ],
        seed=42,
    )

    train_dataset = interleave_datasets(
        [
            ds_arabic.skip(600),
            nastaliq.skip(600),
            naskh.skip(600),
            urdu_news,
            parsynth_train,
            persian_train,
            kannada_train,
        ],
        seed=42,
        stopping_strategy="all_exhausted",
    )

    return {"train": train_dataset, "test": test_dataset}
