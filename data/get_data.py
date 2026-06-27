from datasets import Image, interleave_datasets, load_dataset


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
    ds_arabic = load_dataset("mssqpi/Arabic-OCR-Dataset", split="train", streaming=True)

    # Farsi
    parsynth_train = load_dataset("hezarai/parsynth-ocr-200k", split="train", streaming=True).rename_column("image_path", "image")
    parsynth_test = load_dataset("hezarai/parsynth-ocr-200k", split="test", streaming=True).rename_column("image_path", "image")

    persian_ocr_train = load_dataset("ordaktaktak/Persian-OCR-230k", split="train", streaming=True).rename_column("fname", "image")
    persian_ocr_test = load_dataset("ordaktaktak/Persian-OCR-230k", split="test", streaming=True).rename_column("fname", "image")

    # Urdu
    nastaliq = load_dataset("PuristanLabs1/urdu-ocr-1M", "nastaliq", split="train", streaming=True)
    naskh = load_dataset("PuristanLabs1/urdu-ocr-1M", "naskh", split="train", streaming=True)
    urdu_news = load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="train", streaming=True)

    urdu_news_test = load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="test", streaming=True)
    urdu_news_val = load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="validation", streaming=True)

    test_dataset = interleave_datasets(
        [
            force_image_schema(ds_arabic.take(500)),
            force_image_schema(nastaliq.take(500)),
            force_image_schema(naskh.take(500)),
            force_image_schema(urdu_news_test.take(500)),
            force_image_schema(parsynth_test.take(500)),
            force_image_schema(persian_ocr_test.take(500)),
            force_image_schema(urdu_news_val)
        ],
        probabilities=[0.2, 0.2, 0.2, 0.2, 0.2],
        seed=42
    )

    train_dataset = interleave_datasets(
        [
            force_image_schema(ds_arabic.skip(500)),
            force_image_schema(nastaliq.skip(500)),
            force_image_schema(naskh.skip(500)),
            force_image_schema(urdu_news),
            force_image_schema(parsynth_train),
            force_image_schema(persian_ocr_train)
        ],
        probabilities=[0.2, 0.2, 0.2, 0.2, 0.2],
        seed=42,
        stopping_strategy="all_exhausted"
    )

    return {
        "train": train_dataset,
        "test": test_dataset
    }
