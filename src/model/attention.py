from torch import nn

class Attention(nn.Module):
    r"""
    Spatial Attention Graph Convolutional Nets (SAG)

    Args:
        in_dim (int): Number of channels in the input datasets
        hid_dim  (int): Dimensionality of the hidden layer

    Shape:
        - Input: :math:`(mini_batch_size, number_of_blocks, z_dim)`
        - Output (float): e (attention weight)
    """

    def __init__(self, in_dim, hid_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, 1)
        )

    def forward(self, embeddings):
        # embeddings: batch_size, n_blocks, z_dim
        e = self.attention(embeddings)
        return e