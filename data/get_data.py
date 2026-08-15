import io
import os
import urllib.request
from typing import cast

import kagglehub
import pandas as pd
from datasets import Dataset, Image, interleave_datasets, load_dataset, load_from_disk

SCRATCH_BASE = f"/scratch/alpine/{os.getenv('USER', '')}"
CACHE_DIR = os.getenv("HF_HOME", f"{SCRATCH_BASE}/.cache/huggingface")
PROCESSED_DIR = os.getenv("PROCESSED_DATA_DIR", f"{SCRATCH_BASE}/processed_datasets")

KAGGLE_CACHE_DIR = os.path.join(SCRATCH_BASE, "kaggle_cache")
os.environ["KAGGLEHUB_CACHE"] = KAGGLE_CACHE_DIR

SLURM_CPUS = os.getenv("SLURM_CPUS_PER_TASK")
SYSTEM_CPUS = int(SLURM_CPUS) if SLURM_CPUS else (os.cpu_count() or 1)
NUM_PROC = int(os.getenv("SLURM_CPUS_PER_TASK", "8"))
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
        arabic_raw_train.select(range(min(2000, len(arabic_raw_train))))
        .rename_column("markdown", "text")
        .select_columns(["image", "text"])
        .map(**map_config)
        .cast_column("image", Image())
    )

    arabic_raw_test = cast(
        Dataset,
        load_dataset("MohamedRashad/arabic-img2md", split="test", cache_dir=CACHE_DIR),
    )
    arabic_test = (
        arabic_raw_test.select(range(min(250, len(arabic_raw_test))))
        .rename_column("markdown", "text")
        .select_columns(["image", "text"])
        .map(**map_config)
        .cast_column("image", Image())
    )

    print("Loading urdu nastaliq datasets...")
    nastaliq_raw_train = (
        cast(
            Dataset,
            load_dataset("PuristanLabs1/urdu-ocr-1M", name="nastaliq", split="train", cache_dir=CACHE_DIR),
        )
        .select_columns(["image", "text"])
        .select(range(5000))
        .map(**map_config)
        .cast_column("image", Image())
    )

    nastaliq_raw_val = (
        cast(
            Dataset,
            load_dataset("PuristanLabs1/urdu-ocr-1M", name="nastaliq", split="val", cache_dir=CACHE_DIR),
        )
        .select_columns(["image", "text"])
        .select(range(1000))
        .map(**map_config)
        .cast_column("image", Image())
    )

    print("Loading Persian Pixel dataset...")
    full_persian = cast(
        Dataset,
        load_dataset("Omarrran/Persian_Pixel", "full", split="train", cache_dir=CACHE_DIR),
    )
    persian_split = full_persian.select_columns(["image", "text"]).train_test_split(test_size=0.1, seed=42)
    persian_train = persian_split["train"].map(**map_config).cast_column("image", Image())
    persian_test = persian_split["test"].map(**map_config).cast_column("image", Image())

    print("Loading urdoocr dataset...")
    urdu_dir = kagglehub.dataset_download("i191796majid/urdoocr")
    file_path = os.path.join(urdu_dir, "main.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Expected metadata file not found at: {file_path}")

    df = pd.read_csv(file_path)
    img_col, text_col = "file_name", "text"

    file_map = {}
    for root, _, files in os.walk(urdu_dir):
        for file in files:
            full_p = os.path.join(root, file)
            file_map[file] = full_p
            rel_p = os.path.relpath(full_p, urdu_dir).lstrip("/\\")
            file_map[rel_p] = full_p

    def resolve_path(p):
        clean_p = str(p).strip().lstrip("/\\")
        base_name = os.path.basename(clean_p)
        return file_map.get(clean_p) or file_map.get(base_name)

    df["image"] = df[img_col].apply(resolve_path)
    df["text"] = df[text_col].astype(str)

    missing_count = df["image"].isna().sum()
    if missing_count > 0:
        print(f"Warning: Dropping {missing_count} rows with missing image files.")
        df = df.dropna(subset=["image"])

    urdu_raw = Dataset.from_pandas(df[["image", "text"]])
    urdu_split = urdu_raw.select_columns(["image", "text"]).train_test_split(test_size=0.1, seed=42)
    urdu_train = urdu_split["train"].map(**map_config).cast_column("image", Image())
    urdu_test = urdu_split["test"].map(**map_config).cast_column("image", Image())

    train_sources = [arabic_train, nastaliq_raw_train, persian_train, urdu_train]
    test_sources = [arabic_test, nastaliq_raw_val, persian_test, urdu_test]

    print("Interleaving datasets...")
    test_dataset = cast(Dataset, interleave_datasets(test_sources, seed=42))

    train_dataset = cast(
        Dataset,
        interleave_datasets(
            datasets=train_sources,
            probabilities=[0.3, 0.1, 0.3, 0.3],
            stopping_strategy="all_exhausted",
            seed=42,
        ).shuffle(seed=42),
    )

    print(f"Saving processed datasets to disk cache: {PROCESSED_DIR}")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_dataset.save_to_disk(train_cache_path)
    test_dataset.save_to_disk(test_cache_path)

    return {"train": train_dataset, "test": test_dataset}
