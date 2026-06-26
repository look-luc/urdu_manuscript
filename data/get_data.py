from typing import cast

from datasets import Dataset, Image, concatenate_datasets, load_dataset


def get_split_or_empty(ds, split_name):
    targets = [split_name, "val", "validation", "test"]
    for target in targets:
        if target in ds:
            return ds[target]
    print(f"Warning: No valid split found. Available: {list(ds.keys())}")
    return None

def force_image_schema(ds: Dataset) -> Dataset:
    if ds.features["image"].dtype == "image":
        return ds
    return ds.cast_column("image", Image())

def get_datasets():
    # --- Arabic ---
    ds_arabic = load_dataset("mssqpi/Arabic-OCR-Dataset")["train"]
    ds_arabic = force_image_schema(ds_arabic)
    ds_arabic_split = ds_arabic.train_test_split(test_size=0.2, seed=42)
    ds_arabic_train, ds_arabic_test = ds_arabic_split["train"], ds_arabic_split["test"]

    # --- Farsi/Persian ---
    parsynth_ds_train = load_dataset("hezarai/parsynth-ocr-200k", split="train")
    parsynth_ds_test = load_dataset("hezarai/parsynth-ocr-200k", split="test")

    parsynth_ds_train = parsynth_ds_train.rename_column("image_path", "image")
    parsynth_ds_test = parsynth_ds_test.rename_column("image_path", "image")

    parsynth_ds_train = force_image_schema(parsynth_ds_train)
    parsynth_ds_test = force_image_schema(parsynth_ds_test)

    # --- Urdu Datasets ---
    nastaliq_ds = load_dataset("PuristanLabs1/urdu-ocr-1M", "nastaliq", keep_in_memory=False)
    naskh_ds = load_dataset("PuristanLabs1/urdu-ocr-1M", "naskh")
    urdu_news_ds = load_dataset("oddadmix/qari-0.2.2-news-dataset-large")

    def process_urdu(ds_split):
        return force_image_schema(ds_split)

    # 1. Collect Training Splits
    urdu_train_splits = [
        process_urdu(nastaliq_ds["train"]),
        process_urdu(naskh_ds["train"]),
        process_urdu(urdu_news_ds["train"])
    ]
    urdu_ds_train = concatenate_datasets(urdu_train_splits)

    urdu_test_splits = []
    for ds in [nastaliq_ds, naskh_ds, urdu_news_ds]:
        split = get_split_or_empty(ds, "test") or get_split_or_empty(ds, "validation")
        if split:
            urdu_test_splits.append(process_urdu(split))

    if urdu_test_splits:
        urdu_ds_test = concatenate_datasets(urdu_test_splits)
    else:
        raise ValueError("No test or validation splits found in Urdu datasets.")

    return {
        "train": {
            "arabic": ds_arabic_train,
            "farsi": parsynth_ds_train,
            "urdu": urdu_ds_train
        },
        "test": {
            "arabic": ds_arabic_test,
            "farsi": parsynth_ds_test,
            "urdu": urdu_ds_test
        }
    }
