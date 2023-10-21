import torch
from typing import Any, Callable, Tuple
from torch import Tensor
from torchvision.transforms import v2, InterpolationMode
from torchvision.datasets import CIFAR10


class ContrastiveCIFAR10(CIFAR10):
    def __init__(
        self,
        root: str,
        transform: Callable[..., Any] | None = None,
        train: bool = True,
        download: bool = False,
    ) -> None:
        transforms = self._get_transforms() if transform is None else transform
        super().__init__(root, train, transforms, None, download)

    def _get_transforms(self):
        transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=True),
                # geom invariances
                v2.RandomResizedCrop((32, 32), (0.3, 1.0), interpolation=InterpolationMode.BICUBIC, antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                # color invariances
                v2.RandomApply([v2.ColorJitter(brightness=0.4, hue=0.1, contrast=0.1)], p=0.8),
                v2.RandomGrayscale(p=0.1),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.5,), std=(0.5,)),
            ]
        )

        return transforms

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        # instead of returning one transformed image along with its label, it returns a pair of images transformed
        # independently
        img = self.data[index]
        img_1 = self.transform(img)
        img_2 = self.transform(img)

        return img_1, img_2
