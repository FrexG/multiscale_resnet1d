# multiscale_resnet1d
My re-write of the multi-scale-1d-resnet implemented originally [here](https://github.com/geekfeiw/Multi-Scale-1D-ResNet/).
![architecture](msresnet.png)

## Usage
``` python
import torch

from multi_scale_resnet1d import MSResnet

x = torch.randn(8,32,256) # B,C,T

in_channels = 32
out_channels = 64
scale_list = (3,5,7) # multi sclase
n = 3 # number of conv1d blocks in each residual block

model = MSResnet(in_channels,
                  out_channels,
                  n,
                  scale_list
                  )

out = model(x) # (B,C)
```

## Requirements
- PyTorch > 1.8.x
- Python > 3.8.x