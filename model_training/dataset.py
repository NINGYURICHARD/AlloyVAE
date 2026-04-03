from typing import Any, Callable, Optional, Tuple

import torch
from torch.utils.data import Dataset


class MapDataset(Dataset):
    def __init__(
        self,
        x_data,
        y_data,
        transform: Optional[Callable[[Any, Any], Tuple[Any, Any]]] = None,
    ):
        if len(x_data) != len(y_data):
            raise ValueError(f"x_data and y_data must have the same length, got {len(x_data)} and {len(y_data)}")

        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform

    def __len__(self) -> int:
        return len(self.y_data)

    def __getitem__(self, idx: int):
        x_item = self.x_data[idx]
        y_item = self.y_data[idx]

        if self.transform is not None:
            x_item, y_item = self.transform(x_item, y_item)

        return x_item, y_item


class TensorMapDataset(MapDataset):
    def __getitem__(self, idx: int):
        x_item, y_item = super().__getitem__(idx)
        return torch.as_tensor(x_item, dtype=torch.float32), torch.as_tensor(y_item, dtype=torch.float32)
