import numpy as np
import torch
from albumentations import TransformType
from albumentations.pytorch import ToTensorV2
from numpy.typing import NDArray
from torch.utils.data import Dataset

from patch_camelyon.typing import Sample


class PatchCamelyon(Dataset[Sample]):
    def __init__(
        self, path_x: str, path_y: str, transforms: TransformType | None = None
    ) -> None:
        """Initialize the dataset.

        Args:
            path_x: Path to the numpy file containing the images.
            path_y: Path to the numpy file containing the labels.
            transforms: Optional albumentations transform to apply to the images.
        """
        super().__init__()
        self.x = np.load(path_x, mmap_mode="r")
        self.y = np.load(path_y, mmap_mode="r")
        self.transforms = transforms
        self.to_tensor = ToTensorV2()

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.x)

    def __getitem__(self, index: int) -> Sample:
        """Return the sample at the given index.

        Args:
            index: Index of the sample to return.

        Returns:
            Tuple containing the image and the label.
            Image is a tensor of shape (C, H, W) and label is a tensor of shape (1,).
        """
        image: NDArray[np.uint8] = self.x[index]  # (96, 96, 3)
        label: NDArray[np.uint8] = self.y[index]  # (1,)

        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        return self.to_tensor(image=image)["image"], torch.tensor(label).float()
