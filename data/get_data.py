import io
import os
import urllib.request
from typing import cast

from datasets import Dataset, interleave_datasets, load_dataset, load_from_disk

SCRATCH_BASE = f"/scratch/alpine/{os.getenv('USER', '')}"
CACHE_DIR = os.getenv("HF_HOME", f"{SCRATCH_BASE}/.cache/huggingface")
PROCESSED_DIR = os.getenv("PROCESSED_DATA_DIR", f"{SCRATCH_BASE}/processed_datasets")

SLURM_CPUS = os.getenv("SLURM_CPUS_PER_TASK")
SYSTEM_CPUS = int(SLURM_CPUS) if SLURM_CPUS else (os.cpu_count() or 1)
NUM_PROC = min(SYSTEM_CPUS, 16)
MAP_BATCH_SIZE = 256


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
    train_cache_path = os.path.join(PROCESSED_DIR, "train")
    test_cache_path = os.path.join(PROCESSED_DIR, "test")

    if os.path.exists(train_cache_path) and os.path.exists(test_cache_path):
        print(f"Loading pre-processed datasets directly from disk cache: {PROCESSED_DIR}")
        train_dataset = load_from_disk(train_cache_path)
        test_dataset = load_from_disk(test_cache_path)
        return {"train": train_dataset, "test": test_dataset}

    print(f"No disk cache found at '{PROCESSED_DIR}'. Running mapping pipeline...")
    map_config = {
        "function": _all_same_type,
        "batched": True,
        "batch_size": MAP_BATCH_SIZE,
        "writer_batch_size": MAP_BATCH_SIZE,
        "num_proc": NUM_PROC,
    }

    print("Loading arabic datasets...")
    arabic_raw_train = cast(
        Dataset,
        load_dataset("MohamedRashad/arabic-img2md", split="train", cache_dir=CACHE_DIR),
    )
    arabic_train = (
        arabic_raw_train.select(range(min(15000, len(arabic_raw_train))))
        .rename_column("markdown", "text")
        .select_columns(["image", "text"])
        .map(**map_config)
    )

    arabic_raw_test = cast(
        Dataset,
        load_dataset("MohamedRashad/arabic-img2md", split="test", cache_dir=CACHE_DIR),
    )
    arabic_test = (
        arabic_raw_test.select(range(min(3000, len(arabic_raw_test))))
        .rename_column("markdown", "text")
        .select_columns(["image", "text"])
        .map(**map_config)
    )

    print("Loading persian datasets...")
    parsynth_train = cast(
        Dataset,
        load_dataset("hezarai/parsynth-ocr-200k", split="train", cache_dir=CACHE_DIR),
    ).rename_column("image_path", "image").select_columns(["image", "text"]).map(**map_config)

    parsynth_test = cast(
        Dataset,
        load_dataset("hezarai/parsynth-ocr-200k", split="test", cache_dir=CACHE_DIR),
    ).rename_column("image_path", "image").select_columns(["image", "text"]).map(**map_config)

    persian_raw = cast(
        Dataset,
        load_dataset("Omarrran/Persian_Pixel", name="full", split="train", cache_dir=CACHE_DIR),
    ).select_columns(["image", "text"])

    persian_pixel_test = persian_raw.select(range(10000)).map(**map_config)
    persian_pixel_train = persian_raw.select(range(10000, 35000)).map(**map_config)

    print("Loading urdu datasets...")
    nastaliq_raw_train = cast(
        Dataset,
        load_dataset("PuristanLabs1/urdu-ocr-1M", name="nastaliq", split="train", cache_dir=CACHE_DIR),
    ).select_columns(["image", "text"]).select(range(150000)).map(**map_config)

    nastaliq_raw_val = cast(
        Dataset,
        load_dataset("PuristanLabs1/urdu-ocr-1M", name="nastaliq", split="val", cache_dir=CACHE_DIR),
    ).select_columns(["image", "text"]).map(**map_config)

    test_sources = [
        arabic_test,
        parsynth_test,
        persian_pixel_test,
        nastaliq_raw_val,
    ]
    test_dataset = cast(Dataset, interleave_datasets(test_sources, seed=42))

    train_sources = [
        arabic_train,
        parsynth_train,
        persian_pixel_train,
        nastaliq_raw_train,
    ]

    train_dataset = cast(
        Dataset,
        interleave_datasets(
            datasets=train_sources,
            probabilities=[0.15, 0.20, 0.15, 0.50],
            stopping_strategy="all_exhausted",
            seed=42,
        ).shuffle(
            seed=42,
        ),
    )

    print(f"Saving processed datasets to disk cache: {PROCESSED_DIR}")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_dataset.save_to_disk(train_cache_path)
    test_dataset.save_to_disk(test_cache_path)

    return {"train": train_dataset, "test": test_dataset}
