import base64
from typing import cast

import datasets
from datasets import IterableDataset, interleave_datasets, load_dataset


def to_raw_bytes(example):
    """Normalizes image data to raw binary bytes with strict schema guarantees without PIL."""
    img_data = example.get("image") or example.get("image_path") or example.get("image_base64")
    txt_data = example.get("text") or example.get("markdown") or example.get("chunk")

    raw_bytes = None

    if isinstance(img_data, dict):
        if img_data.get("bytes"):
            raw_bytes = img_data["bytes"]
        elif img_data.get("path"):
            with open(img_data["path"], "rb") as f:
                raw_bytes = f.read()

    elif isinstance(img_data, str):
        if img_data.startswith("data:image") or "image_base64" in example:
            if "," in img_data:
                img_data = img_data.split(",", 1)[1]
            raw_bytes = base64.b64decode(img_data)
        else:
            with open(img_data, "rb") as f:
                raw_bytes = f.read()

    elif isinstance(img_data, bytes):
        raw_bytes = img_data

    if not isinstance(raw_bytes, bytes):
        raise TypeError(
            f"Unable to extract binary bytes for image_bytes column. Got type: {type(img_data)}"
        )

    return {"image_bytes": raw_bytes, "text": str(txt_data or "")}


def _prepare_stream(dataset_name: str, split: str, name: str = "") -> IterableDataset:
    kwargs = {"split": split, "streaming": True}
    if name:
        kwargs["name"] = name

    ds = load_dataset(dataset_name, **kwargs)

    if "image" in ds.features:
        ds = ds.cast_column("image", datasets.Image(decode=False))

    return cast(
        IterableDataset,
        ds.map(to_raw_bytes).select_columns(["image_bytes", "text"])
    )


def get_datasets(buffer_size: int = 100):
    print(f"Loading dataset streams with binary byte alignment (buffer_size={buffer_size})...")

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
