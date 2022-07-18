import numpy as np
import scipy.signal as sps
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from typing import Type, Any, Callable, Union, List, Optional, Tuple, cast



def conv3x3(in_channels: int, out_channels: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        bias=False
    )

class Attention2d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_kernels: int,
                 kernel_size: int,
                 padding_size: int,
                 ) -> None:

        super(Attention2d, self).__init__()

        self.conv_depth = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * num_kernels,
            kernel_size=kernel_size,
            padding=padding_size,
            groups=in_channels
        )
        self.conv_point = nn.Conv2d(
            in_channels=in_channels * num_kernels,
            out_channels=out_channels,
            kernel_size=(1, 1)
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor, size: torch.Size) -> torch.Tensor:
        x = F.adaptive_max_pool2d(x, size)
        x = self.conv_depth(x)
        x = self.conv_point(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


def scale(old_value, old_min, old_max, new_min, new_max):
    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    new_value = (((old_value - old_min) * new_range) / old_range) + new_min

    return new_value





class BasicBlock(nn.Module):

    expansion: int = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 ) -> None:

        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(torch.nn.Module):

    expansion: int = 4

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 ) -> None:
                 
        super(Bottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetWithAttention(nn.Module):

    def __init__(self,
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: List[int],
                 num_classes: int = 1000,
                 apply_attention: bool = False,
                 num_channels: int = 3,
                 zero_init_residual: bool = False,
                 groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation: Optional[List[bool]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 ) -> None:

        super(ResNetWithAttention, self).__init__()

        self.apply_attention = apply_attention  

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                f'replace_stride_with_dilation should be None or a 3-element tuple, got {replace_stride_with_dilation}'
            )

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(num_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        if self.apply_attention:
            self.att1 = Attention2d(
                in_channels=64,
                out_channels=64 * block.expansion,
                num_kernels=1,
                kernel_size=(3, 1),
                padding_size=(1, 0)
            )

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        if self.apply_attention:
            self.att2 = Attention2d(
                in_channels=64 * block.expansion,
                out_channels=128 * block.expansion,
                num_kernels=1,
                kernel_size=(1, 5),
                padding_size=(0, 2)
            )

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        if self.apply_attention:
            self.att3 = Attention2d(
                in_channels=128 * block.expansion,
                out_channels=256 * block.expansion,
                num_kernels=1,
                kernel_size=(3, 1),
                padding_size=(1, 0)
            )

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        if self.apply_attention:
            self.att4 = Attention2d(
                in_channels=256 * block.expansion,
                out_channels=512 * block.expansion,
                num_kernels=1,
                kernel_size=(1, 5),
                padding_size=(0, 2)
            )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if self.apply_attention:
            self.att5 = Attention2d(
                in_channels=512 * block.expansion,
                out_channels=512 * block.expansion,
                num_kernels=1,
                kernel_size=(3, 5),
                padding_size=(1, 2)
            )

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self,
                    block: Type[Union[BasicBlock, Bottleneck]],
                    planes: int,
                    blocks: int,
                    stride: int = 1,
                    dilate: bool = False,
                    ) -> nn.Sequential:

        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(
            self.inplanes,
            planes,
            stride,
            downsample,
            self.groups,
            self.base_width,
            previous_dilation,
            norm_layer
        ))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(
                self.inplanes,
                planes,
                groups=self.groups,
                base_width=self.base_width,
                dilation=self.dilation,
                norm_layer=norm_layer
            ))

        return nn.Sequential(*layers)

    def _forward_pre_processing(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.get_default_dtype())

        return x

    def _forward_pre_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        return x

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_pre_features(x)

        if self.apply_attention:
            x_att = x.clone()
            x = self.layer1(x)
            x_att = self.att1(x_att, x.shape[-2:])
            x = x * x_att

            x_att = x.clone()
            x = self.layer2(x)
            x_att = self.att2(x_att, x.shape[-2:])
            x = x * x_att

            x_att = x.clone()
            x = self.layer3(x)
            x_att = self.att3(x_att, x.shape[-2:])
            x = x * x_att

            x_att = x.clone()
            x = self.layer4(x)
            x_att = self.att4(x_att, x.shape[-2:])
            x = x * x_att
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

        return x

    def _forward_reduction(self, x: torch.Tensor) -> torch.Tensor:
        if self.apply_attention:
            x_att = x.clone()
            x = self.avgpool(x)
            x_att = self.att5(x_att, x.shape[-2:])
            x = x * x_att
        else:
            x = self.avgpool(x)

        x = torch.flatten(x, 1)

        return x

    def _forward_classifier(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)

        return x


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_pre_processing(x)
        x = self._forward_features(x)
        x = self._forward_reduction(x)
        y_pred = self._forward_classifier(x)
        return y_pred
