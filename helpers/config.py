from dataclasses import dataclass
import torch
from torch import get_num_threads


@dataclass
class TrainingConfig:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size: int = 16
    num_workers: int = get_num_threads()
    learning_rate: float = 0.002
    weight_decay: float = 1e-6
    max_epoch: int = 100


@dataclass
class ResNetConfig:
    in_channels: int
    num_stage: int
    num_block_per_stage: int
    out_channels_first: int = 64
    kernel_size_first: int = 7


@dataclass
class SSLConfig:
    dim_embedding: int
    num_hidden_proj: int
    dim_hidden_proj: int | None = None
