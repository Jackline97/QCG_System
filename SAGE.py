from torch_geometric.nn.conv import SAGEConv
from typing import Optional, Callable, List
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
from torch import Tensor
from torch_geometric.typing import Adj
from torch.nn import ModuleList, Linear, ReLU
import torch

class BasicGNN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
                 out_channels: Optional[int] = None, dropout: float = 0.0,
                 act: Optional[Callable] = ReLU(inplace=True),
                 norm: Optional[torch.nn.Module] = None, jk: str = 'lstm'):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.act = act

        self.convs = ModuleList()

        self.norms = None
        if norm is not None:
            self.norms = ModuleList(
                [copy.deepcopy(norm) for _ in range(num_layers)])

        if jk != 'last':
            self.jk = JumpingKnowledge(jk, hidden_channels, num_layers)

        if out_channels is not None:
            self.out_channels = out_channels
            if jk == 'cat':
                self.lin = Linear(num_layers * hidden_channels, out_channels)
            else:

                self.lin = Linear(hidden_channels, out_channels)
        else:
            if jk == 'cat':
                self.out_channels = num_layers * hidden_channels
            else:
                self.out_channels = hidden_channels

                # 加入layer mean

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
        xs: List[Tensor] = []
        # Add the item itself

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, *args, **kwargs)
            if self.norms is not None:
                x = self.norms[i](x)
            if self.act is not None:
                x = self.act(x)
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
            if hasattr(self, 'jk'):
                xs.append(x)

        x = self.jk(xs) if hasattr(self, 'jk') else x
        x = self.lin(x) if hasattr(self, 'lin') else x
        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.num_layers})')


class GraphSAGE(BasicGNN):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
                 out_channels: Optional[int] = None, dropout: float = 0.0,
                 act: Optional[Callable] = ReLU(inplace=True),
                 norm: Optional[torch.nn.Module] = None, jk: str = 'lstm',
                 **kwargs):
        super().__init__(in_channels, hidden_channels, num_layers,
                         out_channels, dropout, act, norm, jk)

        self.convs.append(SAGEConv(in_channels, hidden_channels, **kwargs))
        for _ in range(1, num_layers):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels, **kwargs))