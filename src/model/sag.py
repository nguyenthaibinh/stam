import torch as th
import torch.nn as nn
import torch.nn.functional as F

from graph.stgcn_block import STGCN_Block
from graph.graph import Graph
from model.attention import Attention

class SAG(nn.Module):
    r"""
    Spatial Attention Graph Convolutional Nets (SAG)

    Args:
        in_dim (int): Number of channels in the input datasets
        z_dim  (int): Dimensionality of output node features
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_dim, T_{in}, V_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes.
    """

    def __init__(
            self, in_dim, z_dim, beta_dim, graph_args,
            edge_importance_weighting=None, **kwargs
    ):
        super(SAG, self).__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = th.tensor(self.graph.A, dtype=th.float, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_dim * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            ## 3 layers  256-dim
            STGCN_Block(in_dim, 64, kernel_size, 1, **kwargs0),
            STGCN_Block(64, 128, kernel_size, 2, **kwargs),
            STGCN_Block(128, z_dim, kernel_size, 2, **kwargs)
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(th.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        self.spatial_attn = Attention(in_dim=z_dim, hid_dim=beta_dim)

        # fcn for prediction
        self.fcn = nn.Conv2d(z_dim, z_dim, kernel_size=1)

    def forward(self, x):
        # datasets normalization
        # datasets normalization
        # N: batch_size
        # C: num_channels
        # T: num_frames
        # V: num_joints
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(N, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        x = x.view(N, c, t, v).permute(0, 1, 2, 3)
        feature = F.avg_pool2d(x, (x.size()[2], 1))
        feature = feature.squeeze(dim=2).permute(0, 2, 1)
        e = self.spatial_attn(feature)
        beta = th.softmax(e, dim=1)
        feature = th.bmm(th.transpose(beta, 1, 2), feature).permute(0, 2, 1)
        feature = feature.unsqueeze(dim=3)

        # prediction
        feature = self.fcn(feature)
        feature = feature.view(feature.size(0), -1)

        return feature, beta