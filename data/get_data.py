from typing import cast

from datasets import Dataset, Image, concatenate_datasets, load_dataset


def get_datasets():
    # Arabic
    ds_arabic = load_dataset("mssqpi/Arabic-OCR-Dataset")["train"]
    ds_arabic_split = ds_arabic.train_test_split(test_size=0.2, seed=42)
    ds_arabic_split = ds_arabic_split.cast_column("image", Image())
    ds_arabic_train = ds_arabic_split["train"]
    ds_arabic_test = ds_arabic_split["test"]

    # Farsi/Persian
    parsynth_ds_train = load_dataset("hezarai/parsynth-ocr-200k", split="train")
    parsynth_ds_test = load_dataset("hezarai/parsynth-ocr-200k", split="test")

    parsynth_ds_train = parsynth_ds_train.rename_column("image_path", "image")
    parsynth_ds_train = parsynth_ds_train.cast_column("image", Image())
    parsynth_ds_test = parsynth_ds_test.rename_column("image_path", "image")
    parsynth_ds_test = parsynth_ds_test.cast_column("image", Image())


    persion_ocr_ds_train = load_dataset("ordaktaktak/Persian-OCR-230k", split="train")
    persion_ocr_ds_test = load_dataset("ordaktaktak/Persian-OCR-230k", split="test")

    persion_ocr_ds_train = persion_ocr_ds_train.rename_column("fname", "image")
    persion_ocr_ds_train = persion_ocr_ds_train.cast_column("image", Image())
    persion_ocr_ds_test = persion_ocr_ds_test.rename_column("fname", "image")
    persion_ocr_ds_test = persion_ocr_ds_test.cast_column("image", Image())

    persian_ds_train = concatenate_datasets(
        [parsynth_ds_train,
            persion_ocr_ds_train]
    )

    persian_ds_test = concatenate_datasets(
       [ parsynth_ds_test,
           persion_ocr_ds_test]
    )

    # Urdu
    nastaliq_ds = load_dataset("PuristanLabs1/urdu-ocr-1M", "nastaliq")["train"]
    nastaliq_ds_split = nastaliq_ds.train_test_split(test_size=0.2, seed=42)
    nastaliq_ds_split = nastaliq_ds_split.cast_column("image", Image())
    nastaliq_ds_train = nastaliq_ds_split["train"]
    nastaliq_ds_test = nastaliq_ds_split["test"]

    naskh_ds = load_dataset("PuristanLabs1/urdu-ocr-1M", "naskh")["train"]
    naskh_ds_split = naskh_ds.train_test_split(test_size=0.2, seed=42)
    naskh_ds_split = naskh_ds_split.cast_column("image", Image())
    naskh_ds_train = naskh_ds_split["train"]
    naskh_ds_test = naskh_ds_split["test"]

    urdu_news_ds_train = load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="train")
    urdu_news_ds_train = urdu_news_ds_train.cast_column("image", Image())

    urdu_news_ds_test = load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="test")
    urdu_news_ds_val = load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="validation")

    combine_urdu_news_ds_test = concatenate_datasets(
        [
            cast(Dataset, urdu_news_ds_test),
            cast(Dataset, urdu_news_ds_val)
        ],
        axis=0
    )

    urdu_ds_train = concatenate_datasets(
        [nastaliq_ds_train,
        naskh_ds_train,
        urdu_news_ds_train],
        axis=0
    )

    urdu_ds_test = concatenate_datasets(
        [
            nastaliq_ds_test,
            naskh_ds_test,
            combine_urdu_news_ds_test
        ],
        axis=0
    )

    return {
        "train": {
            "arabic": ds_arabic_train,
            "farsi": persian_ds_train,
            "urdu": urdu_ds_train
        },
        "test": {
            "arabic": ds_arabic_test,
            "farsi": persian_ds_test,
            "urdu": urdu_ds_test
        }
    }
