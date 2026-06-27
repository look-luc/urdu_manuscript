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
    parsynth_train_load = load_dataset("hezarai/parsynth-ocr-200k", split="train", streaming=True)
    parsynth_train_raw = parsynth_train_load.rename_column("image_path", "image") # type: ignore
    parsynth_train: IterableDataset = force_image_schema(parsynth_train_raw)

    parsynth_test_load = load_dataset("hezarai/parsynth-ocr-200k", split="test", streaming=True)
    parsynth_test_raw = parsynth_test_load.rename_column("image_path", "image") # type: ignore
    parsynth_test: IterableDataset = force_image_schema(parsynth_test_raw)

    persian_ocr_dict = load_dataset("ordaktaktak/Persian-OCR-230k", streaming=False)

    persian_ocr_train_raw = persian_ocr_dict["train"].rename_column("fname", "image")
    persian_ocr_train = force_image_schema(persian_ocr_train_raw).to_iterable_dataset()

    persian_ocr_test_raw = persian_ocr_dict["test"].rename_column("fname", "image")
    persian_ocr_test = force_image_schema(persian_ocr_test_raw).to_iterable_dataset()

    # Urdu
    nastaliq: IterableDataset = load_dataset("PuristanLabs1/urdu-ocr-1M", "nastaliq", split="train", streaming=True) # type: ignore
    naskh: IterableDataset = load_dataset("PuristanLabs1/urdu-ocr-1M", "naskh", split="train", streaming=True) # type: ignore
    urdu_news: IterableDataset = load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="train", streaming=True) # type: ignore

    urdu_news_test: IterableDataset = load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="test", streaming=True) # type: ignore
    urdu_news_val: IterableDataset = load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="validation", streaming=True) # type: ignore

    test_dataset = interleave_datasets(
        [
            force_image_schema(ds_arabic.take(500)), # type: ignore
            force_image_schema(nastaliq.take(500)), # type: ignore
            force_image_schema(naskh.take(500)), # type: ignore
            force_image_schema(urdu_news_test), # type: ignore
            parsynth_test,
            persian_ocr_test,
            force_image_schema(urdu_news_val) # type: ignore
        ],
        seed=42
    )

    train_dataset = interleave_datasets(
        [
            force_image_schema(ds_arabic.skip(500)), # type: ignore
            force_image_schema(nastaliq.skip(500)), # type: ignore
            force_image_schema(naskh.skip(500)), # type: ignore
            force_image_schema(urdu_news), # type: ignore
            parsynth_train,
            persian_ocr_train
        ],
        seed=42,
        stopping_strategy="all_exhausted"
    )

    return {
        "train": train_dataset,
        "test": test_dataset
    }
