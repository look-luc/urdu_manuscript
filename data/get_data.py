from typing import cast

import torchvision.io as tv_io
from datasets import IterableDataset, interleave_datasets, load_dataset


def _all_same_type(data:IterableDataset):
    pass

def get_datasets(buffer_size:int=1000):
    print("loading arabic")
    arabic_train = cast(
        IterableDataset,
        load_dataset("MohamedRashad/arabic-img2md", split="train")
    ).rename_column("mardown", "text").select_columns(["image", "text"])
    arabic_test = cast(
        IterableDataset,
        load_dataset("MohamedRashad/arabic-img2md", split="test")
    ).rename_column("mardown", "text").select_columns(["image", "text"])
    print("finish loading arabic")

    print("loading persian")
    parsynth_train = cast(
        IterableDataset,
        load_dataset("hezarai/parsynth-ocr-200k", split="train")
    ).rename_column("image_path", "image").select_columns(["image", "text"])
    parsynth_test = cast(
        IterableDataset,
        load_dataset("hezarai/parsynth-ocr-200k", split="test")
    ).rename_column("image_path", "image").select_columns(["image", "text"])

    persian_pixel = cast(
        IterableDataset,
        load_dataset("Omarrran/Persian_Pixel", name="full", split="train")
    ).select_columns(["image", "text"])
    print("finish loading persian")

    print("loading urdu")
    nastaliq_raw_train = cast(
        IterableDataset,
        load_dataset("PuristanLabs1/urdu-ocr-1M", name="nastaliq", split="train")
    ).select_columns(["image", "text"])
    nastaliq_raw_val = cast(
        IterableDataset,
        load_dataset("PuristanLabs1/urdu-ocr-1M", name="nastaliq", split="val")
    ).select_columns(["image", "text"])

    naskh_raw_train = cast(
        IterableDataset,
        load_dataset("PuristanLabs1/urdu-ocr-1M", name="naskh", split="train")
    ).select_columns(["image", "text"])
    naskh_raw_test = cast(
        IterableDataset,
        load_dataset("PuristanLabs1/urdu-ocr-1M", name="naskh", split="test")
    ).select_columns(["image", "text"])

    urdu_news_train = cast(
        IterableDataset,
        load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="train")
    ).select_columns(["image", "text"])
    urdu_news_test = cast(
        IterableDataset,
        load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="test")
    ).select_columns(["image", "text"])
    urdu_news_val = cast(
        IterableDataset,
        load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="val")
    ).select_columns(["image", "text"])
    print("finish loading urdu")

    urdu_ds_train = interleave_datasets(
        [nastaliq_raw_train, naskh_raw_train, urdu_news_train],
        seed=42,
    )
    urdu_ds_test = interleave_datasets(
        [nastaliq_raw_val, naskh_raw_test, urdu_news_test, urdu_news_val],
        seed=42,
    )

    test_sources = [
        arabic_test,
        parsynth_test,
        persian_pixel.take(100000),
        urdu_ds_test,
    ]
    test_dataset = cast(IterableDataset, interleave_datasets(test_sources, seed=42))

    train_sources = [
        arabic_train,
        parsynth_train,
        persian_pixel.skip(100000),
        urdu_ds_train,
    ]
    train_dataset = cast(
        IterableDataset,
        interleave_datasets(
            datasets=train_sources,
            probabilities=[0.15, 0.175, 0.175, 0.50],
            stopping_strategy="all_exhausted",
            seed=42,
        ).shuffle(
            seed=42,
            buffer_size=buffer_size
        )
    )

    return {"train": train_dataset, "test": test_dataset}
