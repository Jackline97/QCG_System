from typing import Optional, Callable, List
from torch_geometric.typing import Adj
import copy
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, ReLU

from torch_geometric.nn.conv import MessagePassing, GATConv
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge


class BasicGNN(torch.nn.Module):
    r"""An abstract class for implementing basic GNN models.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (Callable, optional): The non-linear activation function to use.
            (default: :obj:`torch.nn.ReLU(inplace=True)`)
        norm (torch.nn.Module, optional): The normalization operator to use.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode
            (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"lstm"`).
            (default: :obj:`"last"`)
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch_geometric.nn.conv.MessagePassing` layers.
    """
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
                 out_channels: Optional[int] = None, dropout: float = 0.0,
                 act: Optional[Callable] = ReLU(inplace=True),
                 norm: Optional[torch.nn.Module] = None, jk: str = 'last',
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.act = act
        self.jk_mode = jk
        self.has_out_channels = out_channels is not None

        self.convs = ModuleList()
        self.convs.append(
            self.init_conv(in_channels, hidden_channels, **kwargs))
        for _ in range(num_layers - 2):
            self.convs.append(
                self.init_conv(hidden_channels, hidden_channels, **kwargs))
        if self.has_out_channels and self.jk_mode == 'last':
            self.convs.append(
                self.init_conv(hidden_channels, out_channels, **kwargs))
        else:
            self.convs.append(
                self.init_conv(hidden_channels, hidden_channels, **kwargs))

        self.norms = None
        if norm is not None:
            self.norms = ModuleList()
            for _ in range(num_layers - 1):
                self.norms.append(copy.deepcopy(norm))
            if not (self.has_out_channels and self.jk_mode == 'last'):
                self.norms.append(copy.deepcopy(norm))

        if self.jk_mode != 'last':
            self.jk = JumpingKnowledge(jk, hidden_channels, num_layers)

        if self.has_out_channels:
            self.out_channels = out_channels
            if self.jk_mode == 'cat':
                self.lin = Linear(num_layers * hidden_channels, out_channels)
            elif self.jk_mode in {'max', 'lstm'}:
                self.lin = Linear(hidden_channels, out_channels)
        else:
            self.out_channels = hidden_channels
            if self.jk_mode == 'cat':
                self.lin = Linear(num_layers * hidden_channels,
                                  hidden_channels)

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        raise NotImplementedError

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms or []:
            norm.reset_parameters()
        if hasattr(self, 'jk'):
            self.jk.reset_parameters()
        if hasattr(self, 'lin'):
            self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, *args, **kwargs) -> Tensor:
        """"""
        xs: List[Tensor] = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, *args, **kwargs)
            if (i == self.num_layers - 1 and self.has_out_channels
                    and self.jk_mode == 'last'):
                break
            if self.norms is not None:
                x = self.norms[i](x)
            if self.act is not None:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if hasattr(self, 'jk'):
                xs.append(x)
        x = self.jk(xs) if hasattr(self, 'jk') else x
        x = self.lin(x) if hasattr(self, 'lin') else x
        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.num_layers})')


class GAT(BasicGNN):
    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:

        kwargs = copy.copy(kwargs)
        if 'heads' in kwargs and out_channels % kwargs['heads'] != 0:
            kwargs['heads'] = 1
        if 'concat' not in kwargs or kwargs['concat']:
            out_channels = out_channels // kwargs.get('heads', 1)

        return GATConv(in_channels, out_channels, dropout=self.dropout,
                       **kwargs)


