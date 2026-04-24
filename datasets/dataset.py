import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class WikiArtDataset(Dataset):
    # reads image_path + label from CSV; root_dir is prepended if set

    def __init__(
        self,
        csv_file: str,
        root_dir: str = "",
        transform=None,
    ) -> None:
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        assert "image_path" in self.data.columns, f"no image_path column in {csv_file}"
        assert "label" in self.data.columns, f"no label column in {csv_file}"

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]

        image_path = row["image_path"]
        label = int(row["label"])

        if self.root_dir:
            image_path = os.path.join(self.root_dir, image_path)

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class WikiArtMultiTaskDataset(Dataset):
    # image_path plus artist/style/genre label columns

    def __init__(
        self,
        csv_file: str,
        root_dir: str = "",
        transform=None,
    ) -> None:
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        assert "image_path" in self.data.columns
        assert "artist_label" in self.data.columns, f"no artist_label column in {csv_file}"
        assert "style_label" in self.data.columns, f"no style_label column in {csv_file}"
        assert "genre_label" in self.data.columns, f"no genre_label column in {csv_file}"

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]

        image_path = row["image_path"]
        if self.root_dir:
            image_path = os.path.join(self.root_dir, image_path)

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        labels = {
            "artist": int(row["artist_label"]),
            "style": int(row["style_label"]),
            "genre": int(row["genre_label"]),
        }

        return image, labels
