from typing import Any, Callable, List, Optional, Sequence, Type, Union

import equinox as eqx
import equinox.nn as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array


def _conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, key=None):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        use_bias=False,
        dilation=dilation,
        key=key,
    )


def _conv1x1(in_planes, out_planes, stride=1, key=None):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, use_bias=False, key=key
    )


class _ResNetBasicBlock(eqx.nn.StatefulLayer):
    expansion: int
    conv1: eqx.Module
    bn1: eqx.Module
    relu: Callable
    conv2: eqx.Module
    bn2: eqx.Module
    downsample: eqx.Module
    stride: int

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        key=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        keys = jrandom.split(key, 2)
        self.expansion = 1
        self.conv1 = _conv3x3(inplanes, planes, stride, key=keys[0])
        self.bn1 = norm_layer(planes, axis_name="batch")
        self.relu = jnn.relu
        self.conv2 = _conv3x3(planes, planes, key=keys[1])
        self.bn2 = norm_layer(planes, axis_name="batch")
        if downsample:
            self.downsample = downsample
        else:
            self.downsample = nn.Identity()
        self.stride = stride

    def __call__(
        self,
        x: Array,
        state: nn.State,
        *,
        key: Optional["jax.random.PRNGKey"] = None,
    ) -> Array:
        out = self.conv1(x)
        out, state = self.bn1(out, state)
        out = self.relu(out)
        out = self.conv2(out)
        out, state = self.bn2(out, state)
        identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out, state


class _ResNetBottleneck(eqx.nn.StatefulLayer):
    expansion: int
    conv1: eqx.Module
    bn1: eqx.Module
    conv2: eqx.Module
    bn2: eqx.Module
    conv3: eqx.Module
    bn3: eqx.Module
    relu: Callable
    downsample: eqx.Module
    stride: int
    has_downsample: bool

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        key=None,
    ):
        super(_ResNetBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm
        self.expansion = 4
        keys = jrandom.split(key, 3)
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = _conv1x1(inplanes, width, key=keys[0])
        self.bn1 = norm_layer(width, axis_name="batch")
        self.conv2 = _conv3x3(width, width, stride, groups, dilation, key=keys[1])
        self.bn2 = norm_layer(width, axis_name="batch")
        self.conv3 = _conv1x1(width, planes * self.expansion, key=keys[2])
        self.bn3 = norm_layer(planes * self.expansion, axis_name="batch")
        self.relu = jnn.relu
        if downsample:
            self.has_downsample = True
            self.downsample = downsample
        else:
            self.has_downsample = False
            self.downsample = nn.Identity()
        self.stride = stride

    def __call__(
        self,
        x: Array,
        state: nn.State,
        *,
        key: Optional["jax.random.PRNGKey"] = None,
    ):
        out = self.conv1(x)
        out, state = self.bn1(out, state)
        out = self.relu(out)

        out = self.conv2(out)
        out, state = self.bn2(out, state)
        out = self.relu(out)

        out = self.conv3(out)
        out, state = self.bn3(out, state)

        if self.has_downsample:
            identity, state = self.downsample(x, state)
        else:
            identity = x

        out += identity
        out = self.relu(out)

        return out, state


EXPANSIONS = {_ResNetBasicBlock: 1, _ResNetBottleneck: 4}


class ResNet(eqx.Module):
    """A simple port of `torchvision.models.resnet`"""

    inplanes: int
    dilation: int
    groups: Sequence[int]
    base_width: int
    conv1: eqx.Module
    bn1: eqx.Module
    relu: jnn.relu
    maxpool: eqx.Module
    layer1: eqx.Module
    layer2: eqx.Module
    layer3: eqx.Module
    layer4: eqx.Module
    avgpool: eqx.Module
    fc: eqx.Module

    def __init__(
        self,
        block: Type[Union["_ResNetBasicBlock", "_ResNetBottleneck"]],
        layers: List[int],
        num_classes: int = 1000,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: List[bool] = None,
        norm_layer: Any = None,
        *,
        key: Optional["jax.random.PRNGKey"] = None,
    ):
        super(ResNet, self).__init__()
        norm_layer = nn.BatchNorm

        if key is None:
            key = jrandom.PRNGKey(0)

        keys = jrandom.split(key, 6)
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            use_bias=False,
            key=keys[0],
        )
        self.bn1 = norm_layer(input_size=self.inplanes, axis_name="batch")
        self.relu = jnn.relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer, key=keys[1])
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            norm_layer,
            stride=2,
            dilate=replace_stride_with_dilation[0],
            key=keys[2],
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            norm_layer,
            stride=2,
            dilate=replace_stride_with_dilation[1],
            key=keys[3],
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            norm_layer,
            stride=2,
            dilate=replace_stride_with_dilation[2],
            key=keys[4],
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * EXPANSIONS[block], num_classes, key=keys[5])

    def _make_layer(
        self, block, planes, blocks, norm_layer, stride=1, dilate=False, key=None
    ):
        keys = jrandom.split(key, blocks + 1)
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * EXPANSIONS[block]:
            downsample = nn.Sequential(
                [
                    _conv1x1(
                        self.inplanes, planes * EXPANSIONS[block], stride, key=keys[0]
                    ),
                    norm_layer(planes * EXPANSIONS[block], axis_name="batch"),
                ]
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                key=keys[1],
            )
        )
        self.inplanes = planes * EXPANSIONS[block]
        for block_idx in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    key=keys[block_idx + 1],
                )
            )

        return nn.Sequential(layers)

    def __call__(self, x: Array, state: nn.State) -> Array:
        x = self.conv1(x)
        x, state = self.bn1(x, state)
        x = self.relu(x)
        x = self.maxpool(x)

        x, state = self.layer1(x, state)
        x, state = self.layer2(x, state)
        x, state = self.layer3(x, state)
        x, state = self.layer4(x, state)

        x = self.avgpool(x)
        x = jnp.ravel(x)
        x = self.fc(x)

        return x, state


def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(**kwargs) -> ResNet:
    model = _resnet(_ResNetBasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs) -> ResNet:
    model = _resnet(_ResNetBasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs) -> ResNet:
    model = _resnet(_ResNetBottleneck, [3, 4, 6, 3], **kwargs)
    return model
