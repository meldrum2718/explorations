import torch
from torch import nn

def inspect(label, im):
    """ Print some basic image stats."""
    if im is None:
      return
    print()
    print(label + ':')
    print('shape:', im.shape)
    print('dtype:', im.dtype)
    print('max:', torch.max(im))
    print('min:', torch.min(im))
    if im.dtype == torch.float32:
      print('mean:', torch.mean(im))
      print('std:', torch.std(im))
    print()


class Conv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation=nn.GELU):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      x = self.conv(x)
      x = self.bn(x)
      x = self.activation(x)
      return x


class DownConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation=nn.GELU):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation=nn.GELU):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class Flatten(nn.Module):
    def __init__(self, activation=nn.GELU, kernel_size=7):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=kernel_size)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.activation(x)
        return x


class Unflatten(nn.Module):
    def __init__(self, in_channels: int, activation=nn.GELU, kernel_size=7):
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=kernel_size,
            padding=0,
        )
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation=nn.GELU):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = Conv(in_channels=in_channels, out_channels=out_channels, activation=activation)
        self.conv2 = Conv(in_channels=out_channels, out_channels=out_channels, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = x + self.conv2(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation=nn.GELU):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = Conv(in_channels=in_channels, out_channels=out_channels, activation=activation)
        self.conv2 = DownConv(in_channels=out_channels, out_channels=out_channels, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation=nn.GELU):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = Conv(in_channels=in_channels, out_channels=out_channels, activation=activation)
        self.conv2 = UpConv(in_channels=out_channels, out_channels=out_channels, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class FCBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(in_features=in_channels, out_features=out_channels)
        self.activation = activation()
        self.fc2 = nn.Linear(in_features=out_channels, out_features=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class TimeConditionalUNet(nn.Module):
    def __init__(
        self,
        C: int,
        H: int,
        W: int,
        T: int,
        D: int = 64,
        activation=nn.GELU,
    ):
        """
        Args:
            C: channels
            H: image height
            W: image width
            T: sequence length (context length + 1)
            D: hidden dimension like
        """
        super().__init__()
        self.C = C
        self.H = H
        self.W = W
        self.T = T
        self.D = D
        
        self.cb1 = ConvBlock(in_channels=T*C, out_channels=D, activation=activation)
        self.db1 = DownBlock(in_channels=D, out_channels=D, activation=activation)
        self.db2 = DownBlock(in_channels=D, out_channels=2*D, activation=activation)
        self.flatten = Flatten(activation=activation, kernel_size=(H//4, W//4))
        self.unflatten = Unflatten(in_channels=2*D, activation=activation, kernel_size=(H//4, W//4))
        self.ub2 = UpBlock(in_channels=4*D, out_channels=D, activation=activation)
        self.ub1 = UpBlock(in_channels=2*D, out_channels=D, activation=activation)
        self.cb2 = ConvBlock(in_channels=2*D, out_channels=D, activation=activation)
        self.conv_out = nn.Conv2d(in_channels=D, out_channels=C, kernel_size=3, stride=1, padding=1)

        # time conditioning (so we can walk along a diffusion trajectory)
        self.fc_unflat_t = FCBlock(in_channels=1, out_channels=2*D, activation=activation)
        self.fc_ub2_t = FCBlock(in_channels=1, out_channels=D, activation=activation)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input tensor.
            c: (B, T-1, C, H, W) input tensor.
            t: (B,) normalized time tensor.

        Returns:
            (B, C, H, W) output tensor.
        """

        B, T, C, H, W = x.size(0), self.T, self.C, self.H, self.W

        assert x.shape == (B, C, H, W)
        assert c.shape == (B, T-1, C, H, W)

        x = torch.cat((x.unsqueeze(1), c), dim=1) # (B, T, C, H, W)
        x = x.reshape(B, T * C, H, W)

        t = t.unsqueeze(-1)

        unflat_t = self.fc_unflat_t(t).reshape(B, 2*self.D, 1, 1)
        ub2_t = self.fc_ub2_t(t).reshape(B, self.D, 1, 1)

        x0 = self.cb1(x)
        x1 = self.db1(x0)
        x2 = self.db2(x1)
        flat = self.flatten(x2)
        lat2 = self.unflatten(flat)
        lat2 = lat2 + unflat_t
        lat2 = torch.concat((x2, lat2), dim=1)
        lat1 = self.ub2(lat2)

        lat1 = lat1 + ub2_t
        lat1 = torch.concat((x1, lat1), dim=1)
        lat0 = self.ub1(lat1)
        lat0 = torch.concat((x0, lat0), dim=1)
        lat0 = self.cb2(lat0)
        out = self.conv_out(lat0)
        return out
