import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module


class NTXent(Module):
    # adapted from https://github.com/clabrugere/pytorch-scarf/blob/master/scarf/loss.py but rewrite the loss to avoid
    # explicit log(exp(.)) and log(sum(exp(.))) operations to improve numerical stability.
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_1, z_2):
        batch_size, device = z_1.size(0), z_1.device

        # compute similarities between all the 2N views
        z = torch.cat([z_1, z_2], dim=0)  # (2 * bs, dim_emb)
        similarity = F.cosine_similarity(z[:, None], z[None, :], dim=2) / self.temperature  # (2 * bs, 2 * bs)
        sim_ij = torch.diag(similarity, batch_size)  # (bs,)
        sim_ji = torch.diag(similarity, -batch_size)  # (bs,)

        # positive contains the 2N similarities between two views of the same sample
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # (2 * bs,)

        # negative contains the (2N, 2N - 1) similarities between the view of a sample and all the other views that are
        # not from that same sample
        mask = ~torch.eye(2 * batch_size, 2 * batch_size, dtype=torch.bool, device=device)  # (2 * bs, 2 * bs)
        negatives = similarity[mask].view(2 * batch_size, 2 * batch_size - 1)  # (2 * bs, 2 * bs - 1)

        # the loss can be rewritten as the sum of the alignement loss making the two representations of the same
        # sample closer, and the distribution loss making the representations of different samples farther
        loss_alignement = -torch.mean(positives)
        loss_distribution = torch.mean(torch.logsumexp(negatives, dim=1))
        loss = loss_alignement + loss_distribution

        return loss


class DCL(Module):
    # adapted from https://github.com/raminnakhli/Decoupled-Contrastive-Learning/blob/main/loss/dcl.py
    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor: Tensor, positive: Tensor) -> Tensor:
        # assumes that anchor and positive are tensors of size (bs, dim_emb)
        batch_size, device = anchor.size(0), anchor.device

        # compute cosine similarity
        anchor = F.normalize(anchor, p=2, dim=1)  # (bs, dim_emb)
        positive = F.normalize(positive, p=2, dim=1)  # (bs, dim_emb)
        similarity = anchor @ positive.T  # (bs, bs)

        # compute losses
        positive_loss = -torch.diag(similarity) / self.temperature  # (1, bs)
        neg_similarity = torch.concat((anchor @ anchor.T, similarity), dim=1) / self.temperature  # (bs, 2 * bs)
        neg_mask = torch.eye(batch_size, dtype=anchor.dtype, device=device).repeat_interleave(2, dim=1)  # (bs, 2 * bs)
        negative_loss = torch.logsumexp(neg_similarity + neg_mask, dim=1)  # (bs, 1)

        return torch.mean(positive_loss + negative_loss)


class VICReg(Module):
    # adapted from the pseudo-code in the paper: https://arxiv.org/pdf/2105.04906.pdf
    # https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
    def __init__(
        self,
        lamb: float = 25.0,
        mu: float = 25.0,
        nu: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
        normalize: bool = False,
    ) -> None:
        super().__init__()
        self.lamb = lamb
        self.mu = mu
        self.nu = nu
        self.gamma = gamma
        self.eps = eps
        self.normalize = normalize

    @staticmethod
    def _center(x: Tensor) -> Tensor:
        return x - torch.mean(x, dim=0)

    @staticmethod
    def _representation_loss(x: Tensor, y: Tensor) -> Tensor:
        # minimizes the difference between the embeddings of the two slightly different views of the same sample
        return F.mse_loss(x, y)

    def _variance_loss(self, x: Tensor) -> Tensor:
        # forces std of each embedding dimension to be closer to gamma (> 0) to avoid mapping samples to the same
        # embedding vector, which would result in dimensional collapse
        x_std = torch.sqrt(torch.var(x, dim=0) + self.eps)  # (dim_emb,)
        loss = torch.mean(F.relu(self.gamma - x_std))

        return loss

    @staticmethod
    def _covariance_loss(x: Tensor) -> Tensor:
        # forces the covariance of embedding dimensions to be diagonal in order to make each dimension as independent
        # as possible to prevent them from encoding the same information
        batch_size, dim_emb = x.size()
        covariance = (x.T @ x) / (batch_size - 1)  # (dim_emb, dim_emb)
        covariance.fill_diagonal_(0.0)
        covariance.pow_(2)
        loss = torch.sum(covariance) / dim_emb

        return loss

    def forward(self, z_1: Tensor, z_2: Tensor) -> Tensor:
        # assumes that z_1 and z_2 are tensors of size (bs, dim_emb)
        if self.normalize:
            z_1, z_2 = F.normalize(z_1), F.normalize(z_2)

        representation_loss = self._representation_loss(z_1, z_2)

        z_1, z_2 = self._center(z_1), self._center(z_2)
        variance_loss = 0.5 * (self._variance_loss(z_1) + self._variance_loss(z_2))
        covariance_loss = self._covariance_loss(z_1) + self._covariance_loss(z_2)

        return self.lamb * representation_loss + self.mu * variance_loss + self.nu * covariance_loss
