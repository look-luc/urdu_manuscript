from typing import cast

import torch
import torchvision.io as tv_io
from datasets import IterableDataset, interleave_datasets, load_dataset
from torchvision.transforms.functional import to_pil_image


def _all_same_type(example:dict):
    if "image" in example.keys():
        feature = example.get("image")
    else:
        raise ValueError("image is not a column of data")

    if isinstance(feature, str):
        example["image"] =  to_pil_image(tv_io.read_image(feature, mode=tv_io.ImageReadMode.RGB))
    elif isinstance(feature, bytes):
        example["image"] = to_pil_image(tv_io.decode_image(torch.frombuffer(feature, dtype=torch.uint8), mode=tv_io.ImageReadMode.RGB))
    elif isinstance(feature, dict):
        if "bytes" in feature and feature["bytes"] is not None:
            byte_buffer = torch.frombuffer(feature["bytes"], dtype=torch.uint8)
            tensor = tv_io.decode_image(byte_buffer, mode=tv_io.ImageReadMode.RGB)
            example["image"] = to_pil_image(tensor)
        elif "path" in feature and feature["path"] is not None:
            tensor = tv_io.read_image(feature["path"], mode=tv_io.ImageReadMode.RGB)
            example["image"] = to_pil_image(tensor)

    elif isinstance(feature, torch.Tensor):
        example["image"] = to_pil_image(feature)
    elif hasattr(feature, "size") and hasattr(feature, "mode"):
        pass
    else:
        raise ValueError("Unsupported image format")
    return example

def get_datasets(buffer_size:int=1000):
    print("loading arabic")
    arabic_train = cast(
        IterableDataset,
        load_dataset("MohamedRashad/arabic-img2md", split="train", streaming=True)
    ).rename_column("markdown", "text").select_columns(["image", "text"]).map(_all_same_type)
    arabic_test = cast(
        IterableDataset,
        load_dataset("MohamedRashad/arabic-img2md", split="test", streaming=True)
    ).rename_column("markdown", "text").select_columns(["image", "text"]).map(_all_same_type)
    print("finish loading arabic")

    print("loading persian")
    parsynth_train = cast(
        IterableDataset,
        load_dataset("hezarai/parsynth-ocr-200k", split="train", streaming=True)
    ).rename_column("image_path", "image").select_columns(["image", "text"]).map(_all_same_type)
    parsynth_test = cast(
        IterableDataset,
        load_dataset("hezarai/parsynth-ocr-200k", split="test", streaming=True)
    ).rename_column("image_path", "image").select_columns(["image", "text"]).map(_all_same_type)

    persian_pixel = cast(
        IterableDataset,
        load_dataset("Omarrran/Persian_Pixel", name="full", split="train", streaming=True)
    ).select_columns(["image", "text"]).map(_all_same_type)
    print("finish loading persian")

    print("loading urdu")
    nastaliq_raw_train = cast(
        IterableDataset,
        load_dataset("PuristanLabs1/urdu-ocr-1M", name="nastaliq", split="train", streaming=True)
    ).select_columns(["image", "text"]).map(_all_same_type)
    nastaliq_raw_val = cast(
        IterableDataset,
        load_dataset("PuristanLabs1/urdu-ocr-1M", name="nastaliq", split="val", streaming=True)
    ).select_columns(["image", "text"]).map(_all_same_type)

    naskh_raw_train = cast(
        IterableDataset,
        load_dataset("PuristanLabs1/urdu-ocr-1M", name="naskh", split="train", streaming=True)
    ).select_columns(["image", "text"]).map(_all_same_type)
    naskh_raw_test = cast(
        IterableDataset,
        load_dataset("PuristanLabs1/urdu-ocr-1M", name="naskh", split="val", streaming=True)
    ).select_columns(["image", "text"]).map(_all_same_type)

    urdu_news_train = cast(
        IterableDataset,
        load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="train", streaming=True)
    ).select_columns(["image", "text"]).map(_all_same_type)
    urdu_news_test = cast(
        IterableDataset,
        load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="test", streaming=True)
    ).select_columns(["image", "text"]).map(_all_same_type)
    urdu_news_val = cast(
        IterableDataset,
        load_dataset("oddadmix/qari-0.2.2-news-dataset-large", split="validation", streaming=True)
    ).select_columns(["image", "text"]).map(_all_same_type)
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
