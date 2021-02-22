from torch import nn

class Attention(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, 1)
        )

    def forward(self, embeddings):
        e = self.attention(embeddings)
        return e