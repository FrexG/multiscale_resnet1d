import torch
import torch.nn as nn
import torch.functional as F
from collections import OrderedDict


class MSBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        scale,
        stride=1,
        padding=1,
        use_1x1_conv=False,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=scale,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm1d(num_features=out_channel),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=scale,
                stride=stride,
                padding=padding,
            ),
        )
        if use_1x1_conv:
            self.downsample = nn.Conv1d(
                in_channel, out_channel, kernel_size=1, stride=stride, padding=padding
            )
        else:
            self.downsample = None

        self.final_bn_relu = nn.Sequential(
            OrderedDict(
                [
                    ("bn", nn.BatchNorm1d(num_features=out_channel)),
                    ("relu", nn.ReLU()),
                ]
            )
        )

    def forward(self, x: torch.Tensor):
        Y = self.block(x)
        if self.downsample:
            x = self.downsample(x)

        Y += x
        return self.final_bn_relu(Y)


class ResidualBlocks(nn.Module):
    def __init__(self, in_channel, scale, expansion=2, n_blocks=3):
        super().__init__()
        self.expansion = 2
        self.n_blocks = n_blocks
        use_1x1 = True

        num_channel = in_channel
        blocks = []
        for i in range(n_blocks):
            blocks.append(
                MSBlock(
                    in_channel=num_channel,
                    out_channel=num_channel * expansion,
                    scale=scale,
                    stride=1,
                    padding="same",
                    use_1x1_conv=use_1x1,
                )
            )

            num_channel = num_channel * expansion

        self.resnet_blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for i, block in enumerate(self.resnet_blocks):
            x = block(x)

        return x


class MSResnet(nn.Module):
    def __init__(self, in_channels, n_blocks=3, scale_list: tuple = (3, 5, 7)) -> None:
        """1D Multi-scale Resitual Network
        Input-shape: (B x C x F) -> batch, num_channels, feature_length

        Args:
            in_channels(_type_): num of input channels
            n_blocks (int, optional):The depth of the residual blocks for each scalse. Defaults to 3.
            scale_list (tuple, optional): A list of kernel sizes. Defaults to (3, 5, 7).
        """
        super().__init__()

        self.input_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=7, stride=2),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.muti_scale_resnets = nn.ModuleList(
            ResidualBlocks(in_channel=in_channels, scale=s, n_blocks=n_blocks)
            for s in scale_list
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(32)

        self.fcn = nn.Sequential(nn.Linear(len(scale_list) * (2**n_blocks) * 32, 1))

    def forward(self, x):
        x = self.input_conv(x)
        outs = []
        for res_blocks in self.muti_scale_resnets:
            res_block_out = res_blocks(x)
            avg_pool = self.avg_pool(res_block_out)
            outs.append(avg_pool)

        features = torch.cat(outs, dim=1)

        probs = self.fcn(features.view(x.shape[0], -1))
        return probs

