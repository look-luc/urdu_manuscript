from datasets import Image, IterableDataset, interleave_datasets, load_dataset


def get_split_or_empty(ds, split_name):
    targets = [split_name, "val", "validation", "test"]
    for target in targets:
        if target in ds:
            return ds[target]
    print(f"Warning: No valid split found. Available: {list(ds.keys())}")
    return None

def force_image_schema(ds):
    if ds.features["image"].dtype == "image":
        return ds
    return ds.cast_column("image", Image())

from datasets import Image, IterableDataset, interleave_datasets, load_dataset


def get_split_or_empty(ds, split_name):
    targets = [split_name, "val", "validation", "test"]
    for target in targets:
        if target in ds:
            return ds[target]
    print(f"Warning: No valid split found. Available: {list(ds.keys())}")
    return None

def force_image_schema(ds):
    if ds.features["image"].dtype == "image":
        return ds
    return ds.cast_column("image", Image())

def get_datasets():
    # --- Arabic ---
    ds_arabic: IterableDataset = load_dataset("mssqpi/Arabic-OCR-Dataset", split="train", streaming=True) # type: ignore

    # Farsi
    parsynth_train_raw = load_dataset("hezarai/parsynth-ocr-200k", split="train", streaming=True).rename_column("image_path", "image")
    parsynth_train: IterableDataset = force_image_schema(parsynth_train_raw) # type: ignore

    parsynth_test_raw = load_dataset("hezarai/parsynth-ocr-200k", split="test", streaming=True).rename_column("image_path", "image")
    parsynth_test: IterableDataset = force_image_schema(parsynth_test_raw) # type: ignore

    persian_ocr_train_raw = load_dataset("ordaktaktak/Persian-OCR-230k", split="train", streaming=False).rename_column("fname", "image")
    persian_ocr_train: IterableDataset = force_image_schema(persian_ocr_train_raw).to_iterable_dataset()

    persian_ocr_test_raw = load_dataset("ordaktaktak/Persian-OCR-230k", split="test", streaming=False).rename_column("fname", "image")
    persian_ocr_test: IterableDataset = force_image_schema(persian_ocr_test_raw).to_iterable_dataset()

    # Urdu
    nastaliq: IterableDataset = load_dataset("PuristanLabs1/urdu-ocr-1M", "nastaliq", split="train", streaming=True) # type: ignore
    naskh: IterableDataset = load_dataset("PuristanLabs1/urdu-ocr-1M", "naskh", split="train", streaming=True) # type: ignore
    urdu_news: IterableDataset = load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="train", streaming=True) # type: ignore

    urdu_news_test: IterableDataset = load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="test", streaming=True) # type: ignore
    urdu_news_val: IterableDataset = load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="validation", streaming=True) # type: ignore

    test_dataset = interleave_datasets(
        [
            ds_arabic.take(500),
            nastaliq.take(500),
            naskh.take(500),
            urdu_news_test,
            parsynth_test,  # Now safe to interleave
            persian_ocr_test,
            urdu_news_val
        ],
        seed=42
    )

    train_dataset = interleave_datasets(
        [
            ds_arabic.skip(500),
            nastaliq.skip(500),
            naskh.skip(500),
            urdu_news,
            parsynth_train,  # Now safe to interleave
            persian_ocr_train
        ],
        seed=42,
        stopping_strategy="all_exhausted"
    )

    return {
        "train": train_dataset,
        "test": test_dataset
    }
