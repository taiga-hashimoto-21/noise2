import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self, weight1=10, weight2=1):
        super().__init__()
        self.weight1 = weight1
        self.weight2 = weight2

    def forward(self, predictions, targets):
        # predictions と targets は shape (batch_size, 2) のテンソルを想定
        squared_diff = (predictions - targets) ** 2
        weighted_squared_diff = torch.cat([
            self.weight1 * squared_diff[:, 0].unsqueeze(1),
            self.weight2 * squared_diff[:, 1].unsqueeze(1)
        ], dim=1)
        return torch.mean(weighted_squared_diff)