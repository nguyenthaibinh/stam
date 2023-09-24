from torch import nn
import torch as th
from model.sag import SAG
from model.attention import Attention

class STAM(nn.Module):
    """
    Spatio-Temporal Attention for Motion Modeling (STAM)

    Args:
        in_dim (int): Number of channels in the input datasets
        z_dim  (int): Dimensionality of output node features
        graph_args (dict): The arguments for building the graph
        alpha_dim (int): the dimensionality of alpha
        beta_dim (int): the dimensionality of beta
        dropout_context (float): context dropout probability
        dropout_gcn (float): gcn dropout probability

    Shape:
        - Input: :math:`(mini_batch_size, number_of_blocks,
                         number_of_feature_channels, block_size,
                         number_of_joints)`
        - Output: logits, alphas, betas (attention weights).

    """
    def __init__(
            self, in_dim, z_dim, alpha_dim, beta_dim, graph_args,
            dropout_context, dropout_gcn
    ):
        super(STAM, self).__init__()
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.sag = SAG(
            in_dim=in_dim, z_dim=z_dim, beta_dim=beta_dim,
            graph_args=graph_args, dropout=dropout_gcn
        )
        self.bn1 = nn.BatchNorm1d(z_dim)

        self.output = nn.Sequential(
            nn.Dropout(p=dropout_context),
            nn.Linear(in_features=z_dim, out_features=2, bias=True),
        )

        self.temporal_attn = Attention(in_dim=z_dim, hid_dim=alpha_dim)

    def forward(self, x, ret_attn=False):
        # mini-batch size
        b_size = x.size(0)
        # number of block
        n_blocks = x.size(1)
        # node feature dim
        n_channels = x.size(2)
        # size of a block
        block_size = x.size(3)
        # number of joints
        n_joints = x.size(4)
        
        x = x.view(b_size * n_blocks, n_channels, block_size, n_joints)
        
        # spatial attention (joint attention)
        x_emb, beta = self.sag(x)
        beta = beta.view(b_size, n_blocks, -1)
        x_emb = x_emb.view(b_size, n_blocks, self.z_dim).contiguous()
        
        # temporal attention
        e = self.temporal_attn(x_emb)

        alpha = th.softmax(e, dim=1)

        context = th.bmm(th.transpose(alpha, 1, 2), x_emb).squeeze(1)

        logits = self.output(context)

        if ret_attn is False:
            return logits
        else:
            return logits, alpha.squeeze(dim=2), beta


