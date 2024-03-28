import torch
from torch import nn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        x1= x1 * x
        x2 = x2 * x
        return x1, x2
if __name__ == '__main__':
    x1 = torch.randn(6,10,8,8)
    # x2 = torch.randn(6,8,8,8)
    # model = SpatialAttention()
    # x1,x2 = model(x1,x2)
    up = torch.nn.UpsamplingBilinear2d(scale_factor=2)
    x = up(x1)
    print(x.shape)

