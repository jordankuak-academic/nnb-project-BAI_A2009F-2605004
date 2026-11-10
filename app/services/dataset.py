from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset


class ManifestImageDataset(Dataset):
    def __init__(self, root: Path, samples: list[dict], include_paths: bool = False) -> None:
        self.__root = root
        self.__samples = list(samples)
        self.__include_paths = include_paths
    
    def __len__(self) -> int:
        return len(self.__samples)
    
    def __getitem__(self, index: int) -> tuple[Tensor, int] | tuple[Tensor, int, str]:
        sample = self.__samples[index]
        path = self.__root / sample["path"]
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        target = int(sample["class_index"])
        if self.__include_paths:
            return tensor, target, str(path)
        return tensor, target
