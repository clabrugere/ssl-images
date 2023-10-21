from typing import Tuple
from torch import Tensor, inference_mode
from torch.nn import Module
import torch.nn.functional as F
from models.mlp import MLP


class SSLModel(Module):
    def __init__(self, encoder: Module, dim_embedding: int, num_hidden_proj: int, dim_hidden_proj: int = None) -> None:
        super().__init__()
        self.encoder = encoder
        dim_hidden_proj = dim_hidden_proj if dim_hidden_proj else 2 * dim_embedding
        self.projector = MLP(dim_embedding, dim_hidden_proj, num_hidden_proj)

    def forward(self, x_1: Tensor, x_2: Tensor) -> Tuple[Tensor, Tensor]:
        z_1, z_2 = self.encoder(x_1), self.encoder(x_2)  # (bs, dim_emb), (bs, dim_emb)
        z_1, z_2 = self.projector(z_1), self.projector(z_2)  # (bs, dim_hidden_proj), (bs, dim_hidden_proj)

        return z_1, z_2

    @inference_mode()
    def get_embeddings(self, x: Tensor, normalize: bool = True) -> Tensor:
        z = self.encoder(x)  # (bs, dim_emb)
        if normalize:
            z = F.normalize(z)

        return z
