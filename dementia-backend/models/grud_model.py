import torch
import torch.nn as nn

class GRUD(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUD, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Learnable parameters
        self.gamma_x = nn.Parameter(torch.ones(input_dim))  # How quickly missing values decay
        self.feature_means = nn.Parameter(torch.zeros(input_dim))  # Normal values to decay towards

        # GRU layer
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, mask, delta):
        """
        x: (batch_size, seq_len, input_dim) - input features
        mask: (batch_size, seq_len, input_dim) - 1 where observed, 0 where missing
        delta: (batch_size, seq_len, input_dim) - time since last observation
        """
        # Handle missing values by decaying towards feature_means
        x_hat = mask * x + (1 - mask) * (self.feature_means + torch.exp(-self.gamma_x * delta) * (x - self.feature_means))

        out, _ = self.gru(x_hat)
        out = self.fc(out)
        out = torch.sigmoid(out)
        return out
