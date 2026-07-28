from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import tifffile
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode


ORION_CHANNEL_ORDER = [
    "Hoechst",
    "CD31",
    "CD45",
    "CD68",
    "CD4",
    "FOXP3",
    "CD8a",
    "CD45RO",
    "CD20",
    "PD-L1",
    "CD3e",
    "CD163",
    "E-cadherin",
    "PD-1",
    "Ki67",
    "Pan-CK",
    "SMA",
]

DEFAULT_ORION_MARKERS = [
    "Hoechst",
    "CD31",
    "CD45",
    "CD68",
    "CD4",
    "FOXP3",
    "CD8a",
    "CD45RO",
    "CD20",
    "PD-L1",
    "CD3e",
    "CD163",
    "E-cadherin",
    "Ki67",
    "Pan-CK",
    "SMA",
]


def _read_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _resize_square(image: Image.Image, image_size: int) -> Image.Image:
    image = TF.center_crop(image, min(image.size))
    return TF.resize(image, [image_size, image_size], interpolation=InterpolationMode.BICUBIC)


class OrionMIFDataset(Dataset):
    """ORION-CRC H&E to marker-wise IF dataset.

    Expected root layout:
      root/
        train_dataframe.csv, val_dataframe.csv, test_dataframe.csv
        he/*.jpeg
        if/*.tiff

    The dataframes must contain `image_path` and `target_path` columns. Paths can be
    relative to root or absolute.
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        markers: list[str] | None = None,
        image_size: int = 256,
        max_rows: int | None = None,
        augmented_dir: str | None = None,
        augmented_prob: float = 0.0,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = int(image_size)
        self.augmented_dir = Path(augmented_dir) if augmented_dir else None
        self.augmented_prob = float(augmented_prob)

        dataframe_path = self.root_dir / f"{split}_dataframe.csv"
        self.df = pd.read_csv(dataframe_path)
        if max_rows is not None:
            self.df = self.df.head(max_rows).copy()
        self.df = self.df.reset_index(drop=True)

        self.markers = list(markers or DEFAULT_ORION_MARKERS)
        self.marker_to_channel = {name: idx for idx, name in enumerate(ORION_CHANNEL_ORDER)}
        for marker in self.markers:
            if marker not in self.marker_to_channel:
                raise ValueError(f"Unknown ORION marker {marker!r}. Available: {ORION_CHANNEL_ORDER}")

    def __len__(self) -> int:
        return len(self.df) * len(self.markers)

    def _path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root_dir / path

    def _resolve_he_path(self, rel_path: str) -> Path:
        path = self._path(rel_path)
        if self.augmented_dir is None or self.augmented_prob <= 0:
            return path
        aug_path = self.augmented_dir / Path(rel_path).name
        if aug_path.exists() and torch.rand(1).item() < self.augmented_prob:
            return aug_path
        return path

    def _load_he(self, rel_path: str) -> torch.Tensor:
        image = _read_rgb(self._resolve_he_path(rel_path))
        return TF.to_tensor(_resize_square(image, self.image_size))

    def _load_marker_rgb(self, rel_path: str, channel_idx: int) -> torch.Tensor:
        arr = tifffile.imread(self._path(rel_path))
        channel = arr[..., channel_idx]
        image = Image.fromarray(channel, mode="L")
        gray = TF.to_tensor(_resize_square(image, self.image_size))
        return gray.repeat(3, 1, 1)

    def __getitem__(self, index: int) -> dict:
        row_idx = index // len(self.markers)
        marker_idx = index % len(self.markers)
        row = self.df.iloc[row_idx]
        marker_name = self.markers[marker_idx]
        return {
            "source": self._load_he(str(row["image_path"])),
            "target": self._load_marker_rgb(str(row["target_path"]), self.marker_to_channel[marker_name]),
            "marker_id": torch.tensor(marker_idx, dtype=torch.long),
            "marker_name": marker_name,
            "source_path": str(row["image_path"]),
            "target_path": str(row["target_path"]),
        }


@dataclass(frozen=True)
class HEMITExample:
    source_path: Path
    target_path: Path
    pair_id: str


class HEMITDataset(Dataset):
    """HEMIT paired translation dataset.

    Expected root layout:
      root/{train,val,test}/input/*
      root/{train,val,test}/label/*

    This loader treats HEMIT as a single target domain. The single marker token is
    named `HEMIT` by default so the same marker-conditioned Diffusion-FT code path
    can be used for both ORION and HEMIT.
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        image_size: int = 512,
        marker_name: str = "HEMIT",
        max_rows: int | None = None,
        random_flip: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = int(image_size)
        self.marker_name = marker_name
        self.random_flip = bool(random_flip and split == "train")
        input_dir = self.root_dir / split / "input"
        label_dir = self.root_dir / split / "label"
        names = sorted(p.name for p in input_dir.iterdir() if p.is_file() and not p.name.startswith("."))
        if max_rows is not None:
            names = names[:max_rows]
        self.examples = [
            HEMITExample(input_dir / name, label_dir / name, name)
            for name in names
            if (label_dir / name).exists()
        ]
        if not self.examples:
            raise RuntimeError(f"No HEMIT pairs found under {self.root_dir}/{split}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        item = self.examples[index]
        source = _read_rgb(item.source_path)
        target = _read_rgb(item.target_path)
        if source.size != target.size:
            raise ValueError(f"Mismatched HEMIT pair dimensions: {item.source_path} and {item.target_path}")

        if self.random_flip:
            if random.random() < 0.5:
                source = TF.hflip(source)
                target = TF.hflip(target)
            if random.random() < 0.5:
                angle = random.choice([90, 180, 270])
                source = TF.rotate(source, angle)
                target = TF.rotate(target, angle)

        if self.split == "train":
            width, height = source.size
            if width < self.image_size or height < self.image_size:
                raise ValueError(
                    f"HEMIT training tile is smaller than the {self.image_size}x{self.image_size} crop: "
                    f"{item.source_path}"
                )
            left = random.randint(0, width - self.image_size)
            top = random.randint(0, height - self.image_size)
            crop_box = (left, top, left + self.image_size, top + self.image_size)
            source = source.crop(crop_box)
            target = target.crop(crop_box)

        else:
            source = _resize_square(source, self.image_size)
            target = _resize_square(target, self.image_size)
        return {
            "source": TF.to_tensor(source),
            "target": TF.to_tensor(target),
            "marker_id": torch.tensor(0, dtype=torch.long),
            "marker_name": self.marker_name,
            "source_path": str(item.source_path),
            "target_path": str(item.target_path),
        }


def build_dataset(
    dataset: str,
    root_dir: str,
    split: str,
    image_size: int,
    markers: list[str] | None = None,
    max_rows: int | None = None,
    augmented_dir: str | None = None,
    augmented_prob: float = 0.0,
) -> tuple[Dataset, list[str]]:
    dataset_key = dataset.lower()
    if dataset_key == "orion":
        ds = OrionMIFDataset(
            root_dir=root_dir,
            split=split,
            markers=markers,
            image_size=image_size,
            max_rows=max_rows,
            augmented_dir=augmented_dir,
            augmented_prob=augmented_prob,
        )
        return ds, list(ds.markers)
    if dataset_key == "hemit":
        marker_names = markers or ["HEMIT"]
        if len(marker_names) != 1:
            raise ValueError("HEMIT uses a single target token; pass zero or one marker name.")
        ds = HEMITDataset(
            root_dir=root_dir,
            split=split,
            image_size=image_size,
            marker_name=marker_names[0],
            max_rows=max_rows,
        )
        return ds, list(marker_names)
    raise ValueError(f"Unsupported dataset: {dataset}. Expected 'orion' or 'hemit'.")
