"""Implementations of RNN cells."""
import torch
from torch import nn


class GRUCell(nn.Module):
    """Custom implementation of GRU cell."""

    def __init__(self, num_inputs, num_hidden):
        """Inits GRUCell."""
        super().__init__()

        self._num_inputs = num_inputs
        self._num_hidden = num_hidden

        self._update_gate = Gate(num_inputs, num_hidden, 'sigmoid')
        self._reset_gate = Gate(num_inputs, num_hidden, 'sigmoid')

        self._candidate_gate = Gate(num_inputs, num_hidden, 'tanh')

    def init_hidden(self, batch_size):
        """Returns initialization vector for hidden state."""
        return torch.zeros(
            size=(batch_size, self._num_hidden),
            dtype=torch.float,
            device=torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            ),
        )

    def forward(self, x, h_t_1=None):
        """Performs forward pass of GRU cell."""
        if h_t_1 is None:
            h_t_1 = self.init_hidden(batch_size=x.size(0))

        z = self._update_gate(x, h_t_1)
        r = self._reset_gate(x, h_t_1)

        h_hat = self._candidate_gate(x, (r * h_t_1))

        h_t = z * h_t_1 + (1 - z) * h_hat

        return h_t, h_t


class Gate(nn.Module):
    """Implementation of RNN gate."""

    def __init__(self, num_inputs: int, num_hidden: int, activation: str):
        """Inits RNNGate."""
        super().__init__()
        self._W = _make_param(size=(num_inputs, num_hidden))
        self._U = _make_param(size=(num_hidden, num_hidden))
        self._b = _make_param(size=(num_hidden,))

        if activation == 'sigmoid':
            self._act = nn.Sigmoid()
        elif activation == 'tanh':
            self._act = nn.Tanh()
        else:
            raise RuntimeError(f'Unknown activation: {activation}')

    def forward(self, x, h):
        """Implements forward pass of typical RNN gate."""
        return self._act(x @ self._W + h @ self._U + self._b)


def _make_param(size):
    return torch.nn.Parameter(
        data=torch.randn(
            size=size,
            dtype=torch.float,
            device=torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            ),
        ),
        requires_grad=True,
    )
