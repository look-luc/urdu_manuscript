import base64
from typing import cast

import datasets
import torch
import torchvision.io as tv_io
from datasets import IterableDataset, interleave_datasets, load_dataset


def to_raw_bytes(example):
    """Normalizes image data to raw binary bytes without PIL."""
    img_data = example.get("image") or example.get("image_path") or example.get("image_base64")
    txt_data = example.get("text") or example.get("markdown") or example.get("chunk")

    if isinstance(img_data, str) and (img_data.startswith("data:image") or "image_base64" in example):
        if "," in img_data:
            img_data = img_data.split(",", 1)[1]
        raw_bytes = base64.b64decode(img_data)
    elif isinstance(img_data, dict) and "bytes" in img_data and img_data["bytes"]:
        raw_bytes = img_data["bytes"]
    elif isinstance(img_data, str):
        with open(img_data, "rb") as f:
            raw_bytes = f.read()
    else:
        raw_bytes = img_data

    return {"image_bytes": raw_bytes, "text": str(txt_data)}


def preprocess_sample(example, processor, prompt: str, assistant_start_tensor: torch.Tensor):
    """Transforms raw bytes into model-ready PyTorch tensors using torchvision."""
    raw_bytes = example["image_bytes"]

    # Decode raw bytes using torchvision
    byte_tensor = torch.frombuffer(bytearray(raw_bytes), dtype=torch.uint8)
    img_tensor = tv_io.decode_image(byte_tensor, mode=tv_io.ImageReadMode.RGB)

    txt_content = example.get("text", "")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_tensor},
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": txt_content},
            ],
        },
    ]

    formatted_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    # Encode through AutoProcessor
    inputs = processor(
        text=[formatted_text],
        images=[img_tensor],
        padding=False,
        return_tensors="pt",
    )

    # Squeeze batch dimension for dataset mapping
    input_ids = inputs["input_ids"].squeeze(0)
    attention_mask = inputs["attention_mask"].squeeze(0)
    pixel_values = inputs["pixel_values"].squeeze(0)
    image_grid_thw = inputs["image_grid_thw"].squeeze(0)

    # Calculate labels and mask user prompt
    labels = input_ids.clone()
    pat_len = assistant_start_tensor.size(0)

    match_idx = -1
    if input_ids.size(0) >= pat_len:
        windows = input_ids.unfold(0, pat_len, 1)
        matches = (windows == assistant_start_tensor).all(dim=1)
        indices = torch.nonzero(matches, as_tuple=True)[0]
        if len(indices) > 0:
            match_idx = indices[0].item()

    if match_idx != -1:
        labels[: match_idx + pat_len] = -100
    else:
        labels[:] = -100

    labels[labels == processor.tokenizer.pad_token_id] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "labels": labels,
    }


def _prepare_stream(dataset_name: str, split: str, name: str = "") -> IterableDataset:
    kwargs = {"split": split, "streaming": True}
    if name:
        kwargs["name"] = name

    ds = load_dataset(dataset_name, **kwargs)

    if "image" in ds.features:
        ds = ds.cast_column("image", datasets.Image(decode=False))

    return cast(
        IterableDataset,
        ds.map(to_raw_bytes).select_columns(["image_bytes", "text"])
    )


def get_datasets(processor, prompt: str, buffer_size: int = 100):
    print(f"Loading datasets and mapping tensors upstream (buffer_size={buffer_size})...")

    assistant_tokens = processor.tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )
    assistant_start_tensor = torch.tensor(assistant_tokens, dtype=torch.long)

    arabic_train = _prepare_stream("MohamedRashad/arabic-img2md", split="train")
    arabic_test = _prepare_stream("MohamedRashad/arabic-img2md", split="test")

    parsynth_train = _prepare_stream("hezarai/parsynth-ocr-200k", split="train")
    parsynth_test = _prepare_stream("hezarai/parsynth-ocr-200k", split="test")

    persian_pixel = _prepare_stream("Omarrran/Persian_Pixel", name="full", split="train")

    nastaliq_raw = _prepare_stream("PuristanLabs1/urdu-ocr-1M", name="nastaliq", split="train")
    naskh_raw = _prepare_stream("PuristanLabs1/urdu-ocr-1M", name="naskh", split="train")

    urdu_news_train = _prepare_stream("oddadmix/qari-0.2.2-news-dataset-large", split="train")
    urdu_news_test = _prepare_stream("oddadmix/qari-0.2.2-news-dataset-large", split="test")

    urdu_ds_train = interleave_datasets(
        [nastaliq_raw.skip(1000), naskh_raw.skip(1000), urdu_news_train],
        seed=42,
    )
    urdu_ds_test = interleave_datasets(
        [nastaliq_raw.take(1000), naskh_raw.take(1000), urdu_news_test],
        seed=42,
    )

    test_sources = [
        arabic_test,
        parsynth_test,
        persian_pixel.take(100000),
        urdu_ds_test,
    ]
    test_dataset = interleave_datasets(test_sources, seed=42)

    train_sources = [
        arabic_train,
        parsynth_train,
        persian_pixel.skip(100000),
        urdu_ds_train,
    ]
    train_dataset = interleave_datasets(
        datasets=train_sources,
        probabilities=[0.15, 0.175, 0.175, 0.50],
        stopping_strategy="all_exhausted",
        seed=42,
    ).shuffle(seed=42, buffer_size=buffer_size)

    # Apply preprocessing upstream in dataset.map()
    map_fn = lambda x: preprocess_sample(x, processor, prompt, assistant_start_tensor)

    train_dataset = train_dataset.map(map_fn).select_columns(
        ["input_ids", "attention_mask", "pixel_values", "image_grid_thw", "labels"]
    )
    test_dataset = test_dataset.map(map_fn).select_columns(
        ["input_ids", "attention_mask", "pixel_values", "image_grid_thw", "labels"]
    )

    return {"train": train_dataset, "test": test_dataset}
