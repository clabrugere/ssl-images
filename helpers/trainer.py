from collections import defaultdict

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


def train(
    model: Module,
    loss_func: Module,
    dl_train: DataLoader,
    lr: float,
    max_epoch: int,
    device: torch.device,
    weight_decay: float = 0.0,
) -> defaultdict:
    model.to(device)
    model.train()

    optimizer = Adam(model.parameters(), lr=10 * lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(dl_train) * max_epoch, eta_min=lr, verbose=False)
    loss_history = []

    for epoch in range(max_epoch):
        running_loss = 0.0

        for step, (x_1, x_2) in enumerate(dl_train):
            optimizer.zero_grad(set_to_none=True)

            x_1, x_2 = x_1.to(device, non_blocking=True), x_2.to(device, non_blocking=True)
            z_1, z_2 = model(x_1), model(x_2)

            loss = loss_func(z_1, z_2)
            loss.backward()

            optimizer.step()
            scheduler.step()

            running_loss += loss.detach().cpu()
            print(
                f"Epoch {epoch + 1} - step {step + 1}/{len(dl_train)} - lr: {scheduler.get_last_lr()[-1]:.4f} - ",
                f"loss: {running_loss / (step + 1):.4f}",
                end="\r",
            )

        print()
        loss_history.append(running_loss / (step + 1))
