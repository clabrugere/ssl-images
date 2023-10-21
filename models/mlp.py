from torch import Tensor
from torch.nn import Module, Sequential, Linear, BatchNorm1d, ReLU, Dropout


class MLP(Module):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        num_hidden,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self._layers = Sequential()
        for _ in range(num_hidden - 1):
            self._layers.append(Linear(dim_in, dim_hidden))
            self._layers.append(BatchNorm1d(dim_hidden))
            self._layers.append(ReLU(inplace=True))
            if dropout > 0.0:
                self._layers.append(Dropout(dropout))
            dim_in = dim_hidden

        self._layers.append(Linear(dim_in, dim_hidden, bias=False))

    def forward(self, x: Tensor) -> Tensor:
        return self._layers(x)
