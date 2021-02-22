from torch import nn
import torch as th
from model.sag import SAG
from model.attention import Attention

class STAM(nn.Module):
    """
    Spatio-Temporal Attention for Motion Modeling (STAM)
    """
    def __init__(self, in_dim, z_dim, alpha_dim, beta_dim, graph_args, dropout_context=0.5, dropout_gcn=0.5):
        super(STAM, self).__init__()
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.sag = SAG(in_dim=in_dim, z_dim=z_dim, beta_dim=beta_dim, graph_args=graph_args, dropout=dropout_gcn)
        self.bn1 = nn.BatchNorm1d(z_dim)

        self.output = nn.Sequential(
            nn.Dropout(p=dropout_context),
            nn.Linear(in_features=z_dim, out_features=2, bias=True),
        )

        self.temporal_attn = Attention(in_dim=z_dim, hid_dim=alpha_dim)

    def forward(self, x, ret_attn=False):
        b_size = x.size(0)
        n_blocks = x.size(1)
        n_channels = x.size(2)
        block_size = x.size(3)
        n_joints = x.size(4)
        x = x.view(b_size * n_blocks, n_channels, block_size, n_joints)
        x_emb, beta = self.sag(x)
        beta = beta.view(b_size, n_blocks, -1)
        x_emb = x_emb.view(b_size, n_blocks, self.z_dim).contiguous()

        e = self.temporal_attn(x_emb)

        alpha = th.softmax(e, dim=1)

        context = th.bmm(th.transpose(alpha, 1, 2), x_emb).squeeze(1)
        # without applying non-linearity
        scores = self.output(context)

        if ret_attn is False:
            return scores
        else:
            return scores, alpha.squeeze(dim=2), beta


