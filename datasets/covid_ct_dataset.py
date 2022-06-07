from torchvision import datasets, transforms
from PIL import Image

import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import numpy as np

def ct_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        # return img.convert('L')
        return img.convert('RGB')

class COVID_CT_Dataset(datasets.ImageFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = ct_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(COVID_CT_Dataset, self).__init__(root, loader=loader,transform=transform, target_transform=target_transform, is_valid_file=is_valid_file)

