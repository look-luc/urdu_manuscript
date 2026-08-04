import io
import os
import urllib.request
from typing import cast

from datasets import Dataset, interleave_datasets, load_dataset

CACHE_DIR = os.getenv("HF_HOME", f"/scratch/alpine/{os.getenv('USER', '')}/.cache/huggingface")
NUM_PROC = os.cpu_count() or 1


def _fetch_bytes(path_or_url: str) -> bytes:
    if path_or_url.startswith(("http://", "https://")):
        req = urllib.request.Request(
            path_or_url, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req) as response:
            return response.read()
    else:
        with open(path_or_url, "rb") as f:
            return f.read()


def _process_single_image(feature) -> bytes:
    if isinstance(feature, bytes):
        return feature
    elif isinstance(feature, str):
        return _fetch_bytes(feature)
    elif isinstance(feature, dict):
        if feature.get("bytes") is not None:
            return feature["bytes"]
        elif feature.get("path") is not None:
            return _fetch_bytes(feature["path"])
        else:
            raise ValueError("Dict feature missing both 'bytes' and 'path'")
    elif hasattr(feature, "save"):
        buf = io.BytesIO()
        feature.convert("RGB").save(buf, format="JPEG")
        return buf.getvalue()
    else:
        raise ValueError(f"Unsupported image format: {type(feature)}")


def _all_same_type(batch: dict) -> dict:
    if "image" not in batch:
        raise ValueError("image is not a column of data")

    batch["image"] = [_process_single_image(img) for img in batch["image"]]
    return batch


def get_datasets(buffer_size: int = 1000):
    print("loading arabic")
    arabic_train = cast(
        Dataset,
        load_dataset("MohamedRashad/arabic-img2md", split="train", cache_dir=CACHE_DIR),
    ).rename_column("markdown", "text").select_columns(["image", "text"]).map(_all_same_type, batched=True, num_proc=NUM_PROC)

    arabic_test = cast(
        Dataset,
        load_dataset("MohamedRashad/arabic-img2md", split="test", cache_dir=CACHE_DIR),
    ).rename_column("markdown", "text").select_columns(["image", "text"]).map(_all_same_type, batched=True, num_proc=NUM_PROC)
    print("finish loading arabic")

    print("loading persian")
    parsynth_train = cast(
        Dataset,
        load_dataset("hezarai/parsynth-ocr-200k", split="train", cache_dir=CACHE_DIR),
    ).rename_column("image_path", "image").select_columns(["image", "text"]).map(_all_same_type, batched=True, num_proc=NUM_PROC)

    parsynth_test = cast(
        Dataset,
        load_dataset("hezarai/parsynth-ocr-200k", split="test", cache_dir=CACHE_DIR),
    ).rename_column("image_path", "image").select_columns(["image", "text"]).map(_all_same_type, batched=True, num_proc=NUM_PROC)

    persian_pixel = cast(
        Dataset,
        load_dataset("Omarrran/Persian_Pixel", name="full", split="train", cache_dir=CACHE_DIR),
    ).select_columns(["image", "text"]).map(_all_same_type, batched=True, num_proc=NUM_PROC)
    print("finish loading persian")

    print("loading urdu")
    nastaliq_raw_train = cast(
        Dataset,
        load_dataset("PuristanLabs1/urdu-ocr-1M", name="nastaliq", split="train", cache_dir=CACHE_DIR),
    ).select_columns(["image", "text"]).map(_all_same_type, batched=True, num_proc=NUM_PROC)

    nastaliq_raw_val = cast(
        Dataset,
        load_dataset("PuristanLabs1/urdu-ocr-1M", name="nastaliq", split="val", cache_dir=CACHE_DIR),
    ).select_columns(["image", "text"]).map(_all_same_type, batched=True, num_proc=NUM_PROC)

    naskh_raw_train = cast(
        Dataset,
        load_dataset("PuristanLabs1/urdu-ocr-1M", name="naskh", split="train", cache_dir=CACHE_DIR),
    ).select_columns(["image", "text"]).map(_all_same_type, batched=True, num_proc=NUM_PROC)

    naskh_raw_test = cast(
        Dataset,
        load_dataset("PuristanLabs1/urdu-ocr-1M", name="naskh", split="val", cache_dir=CACHE_DIR),
    ).select_columns(["image", "text"]).map(_all_same_type, batched=True, num_proc=NUM_PROC)

    urdu_news_train = cast(
        Dataset,
        load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="train", cache_dir=CACHE_DIR),
    ).select_columns(["image", "text"]).map(_all_same_type, batched=True, num_proc=NUM_PROC)

    urdu_news_test = cast(
        Dataset,
        load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="test", cache_dir=CACHE_DIR),
    ).select_columns(["image", "text"]).map(_all_same_type, batched=True, num_proc=NUM_PROC)

    urdu_news_val = cast(
        Dataset,
        load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="validation", cache_dir=CACHE_DIR),
    ).select_columns(["image", "text"]).map(_all_same_type, batched=True, num_proc=NUM_PROC)
    print("finish loading urdu")

    urdu_historical_train = interleave_datasets(
        [nastaliq_raw_train, naskh_raw_train],
        seed=42,
    )
    urdu_ds_test = interleave_datasets(
        [nastaliq_raw_val, naskh_raw_test],
        seed=42,
    )

    test_sources = [
        arabic_test,
        parsynth_test,
        persian_pixel.take(100000),
        urdu_ds_test,
        urdu_news_test,
        urdu_news_val,
    ]
    test_dataset = cast(Dataset, interleave_datasets(test_sources, seed=42))

    # Separate train sources to allow explicit sampling control
    train_sources = [
        arabic_train,
        parsynth_train,
        persian_pixel.skip(100000),
        urdu_historical_train,
        urdu_news_train,
    ]

    train_dataset = cast(
        Dataset,
        interleave_datasets(
            datasets=train_sources,
            probabilities=[0.20, 0.15, 0.15, 0.40, 0.10],
            stopping_strategy="all_exhausted",
            seed=42,
        ).shuffle(
            seed=42,
            buffer_size=buffer_size,
        ),
    )

    return {"train": train_dataset, "test": test_dataset}
