from typing import cast

from datasets import Dataset, Image, concatenate_datasets, load_dataset


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
    nastaliq_ds = load_dataset("oddadmix/qari-0.2.2-nastaliq-dataset-large")
    naskh_ds = load_dataset("oddadmix/qari-0.2.2-naskh-dataset-large")
    urdu_news_ds = load_dataset("oddadmix/qari-0.2.2-news-dataset-large")

    def process_urdu(ds_split):
        return force_image_schema(ds_split)

    urdu_ds_train = concatenate_datasets([
        process_urdu(nastaliq_ds["train"]),
        process_urdu(naskh_ds["train"]),
        process_urdu(urdu_news_ds["train"])
    ])

    urdu_ds_test = concatenate_datasets([
        process_urdu(nastaliq_ds["test"]),
        process_urdu(naskh_ds["test"]),
        process_urdu(urdu_news_ds["test"]),
        process_urdu(urdu_news_ds["validation"])
    ])

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
