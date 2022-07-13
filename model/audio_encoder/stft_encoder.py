import torch
import torch.nn as nn
import numpy as np
import torchvision as tv
from typing import Type, Any, Callable, Union, List, Optional, Tuple, cast
import scipy.signal as sps

from .AttentionResNet import ResNetWithAttention, BasicBlock, Bottleneck,scale



class _STFTResNetWithAttention(ResNetWithAttention):

    @staticmethod
    def loading_function(*args, **kwargs) -> torch.nn.Module:
        raise NotImplementedError

    def __init__(self,
                 block: Type[Union[BasicBlock, Bottleneck]],
                 layers: List[int],
                 apply_attention: bool = False,
                 n_fft: int = 256,
                 hop_length: Optional[int] = None,
                 win_length: Optional[int] = None,
                 window: Optional[str] = None,
                 normalized: bool = False,
                 onesided: bool = True,
                 spec_height: int = 224,
                 spec_width: int = 224,
                 num_classes: int = 1000,
                 pretrained: Union[bool, str] = False,
                 lock_pretrained: Optional[Union[bool, List[str]]] = None,
                 zero_init_residual: bool = False,
                 groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation: bool = None,
                 norm_layer: Optional[Type[torch.nn.Module]] = None):

        super(_STFTResNetWithAttention, self).__init__(
            block=block,
            layers=layers,
            apply_attention=apply_attention,
            num_channels=3,
            num_classes=num_classes,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer
        )

        self.num_classes = num_classes

        self.fc = torch.nn.Linear(
            in_features=self.fc.in_features,
            out_features=self.num_classes,
            bias=self.fc.bias is not None
        )

        if hop_length is None:
            hop_length = int(np.floor(n_fft / 4))

        if win_length is None:
            win_length = n_fft

        if window is None:
            window = 'boxcar'

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        self.normalized = normalized
        self.onesided = onesided

        self.spec_height = spec_height
        self.spec_width = spec_width

        self.pretrained = pretrained
        self._inject_members()
        if pretrained:
            err_msg = self.load_pretrained()

            unlocked_weights = list()

            for name, p in self.named_parameters():
                unlock = True
                if isinstance(lock_pretrained, bool):
                    if lock_pretrained and name not in err_msg:
                        unlock = False
                elif isinstance(lock_pretrained, list):
                    if name in lock_pretrained:
                        unlock = False

                p.requires_grad_(unlock)
                if unlock:
                    unlocked_weights.append(name)

            print(f'Following weights are unlocked: {unlocked_weights}')

        window_buffer: torch.Tensor = torch.from_numpy(sps.get_window(window=window, Nx=win_length, fftbins=True)).to(torch.get_default_dtype())
        self.register_buffer('window', window_buffer)

        self.log10_eps = 1e-18

        if self.apply_attention and pretrained and not isinstance(pretrained, str):
            self._reset_attention()

    def _inject_members(self):
        pass

    def _reset_attention(self):

        self.att1.bn.weight.data.fill_(1.0)
        self.att1.bn.bias.data.fill_(1.0)

        self.att2.bn.weight.data.fill_(1.0)
        self.att2.bn.bias.data.fill_(1.0)

        self.att3.bn.weight.data.fill_(1.0)
        self.att3.bn.bias.data.fill_(1.0)

        self.att4.bn.weight.data.fill_(1.0)
        self.att4.bn.bias.data.fill_(1.0)

        self.att5.bn.weight.data.fill_(1.0)
        self.att5.bn.bias.data.fill_(1.0)

    def load_pretrained(self) -> str:
        if isinstance(self.pretrained, bool):
            state_dict = self.loading_func(pretrained=True).state_dict()
        else:
            state_dict = torch.load(self.pretrained, map_location='cpu')

        err_msg = ''
        try:
            self.load_state_dict(state_dict=state_dict, strict=True)
        except RuntimeError as ex:
            err_msg += f'While loading some errors occurred.\n{ex}'

        return err_msg

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            x.view(-1, x.shape[-1]),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            pad_mode='reflect',
            normalized=self.normalized,
            onesided=True
        )

        if not self.onesided:
            spec = torch.cat((torch.flip(spec, dims=(-3,)), spec), dim=-3)

        return spec

    def split_spectrogram(self, spec: torch.Tensor, batch_size: int) -> torch.Tensor:
        spec_height_per_band = spec.shape[-3] // self.conv1.in_channels
        spec_height_single_band = self.conv1.in_channels * spec_height_per_band
        spec = spec[:, :spec_height_single_band]

        spec = spec.reshape(batch_size, -1, spec.shape[-3] // self.conv1.in_channels, *spec.shape[-2:])

        return spec

    def spectrogram_to_power(self, spec: torch.Tensor) -> torch.Tensor:
        spec_height = spec.shape[-3] if self.spec_height < 1 else self.spec_height
        spec_width = spec.shape[-2] if self.spec_width < 1 else self.spec_width

        pow_spec = spec[..., 0] ** 2 + spec[..., 1] ** 2

        if spec_height != pow_spec.shape[-2] or spec_width != pow_spec.shape[-1]:
            pow_spec = F.interpolate(
                pow_spec,
                size=(spec_height, spec_width),
                mode='bilinear',
                align_corners=True
            )

        return pow_spec

    def _forward_pre_processing(self, x: torch.Tensor) -> torch.Tensor:
        x = super(_STFTResNetWithAttention, self)._forward_pre_processing(x)
        x = scale(x, -32768.0, 32767, -1.0, 1.0)

        spec = self.spectrogram(x)
        spec_split_ch = self.split_spectrogram(spec, x.shape[0])
        pow_spec_split_ch = self.spectrogram_to_power(spec_split_ch)
        pow_spec_split_ch = torch.where(
            cast(torch.Tensor, pow_spec_split_ch > 0.0),
            pow_spec_split_ch,
            torch.full_like(pow_spec_split_ch, self.log10_eps)
        )
        pow_spec_split_ch = pow_spec_split_ch.reshape(
            x.shape[0], -1, self.conv1.in_channels, *pow_spec_split_ch.shape[-2:]
        )
        x_db = torch.log10(pow_spec_split_ch).mul(10.0)

        return x_db

    def _forward_features(self, x_db: torch.Tensor) -> List[torch.Tensor]:
        outputs = list()
        for ch_idx in range(x_db.shape[1]):
            ch = x_db[:, ch_idx]
            out = super(_STFTResNetWithAttention, self)._forward_features(ch)
            outputs.append(out)

        return outputs

    def _forward_reduction(self, x: List[torch.Tensor]) -> torch.Tensor:
        outputs = list()
        for ch in x:
            out = super(_STFTResNetWithAttention, self)._forward_reduction(ch)
            outputs.append(out)
        outputs = torch.stack(outputs, dim=-1).sum(dim=-1)

        return outputs


class STFTResNetWithAttention(_STFTResNetWithAttention):

    loading_func = staticmethod(tv.models.resnet50)

    def __init__(self,
                 n_fft: int = 256,
                 hop_length: Optional[int] = None,
                 win_length: Optional[int] = None,
                 window: Optional[str] = None,
                 normalized: bool = False,
                 onesided: bool = True,
                 spec_height: int = 224,
                 spec_width: int = 224,
                 num_classes: int = 1000,
                 apply_attention: bool = False,
                 pretrained: bool = False,
                 lock_pretrained: Optional[Union[bool, List[str]]] = None):

        super(STFTResNetWithAttention, self).__init__(
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            apply_attention=apply_attention,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            normalized=normalized,
            onesided=onesided,
            spec_height=spec_height,
            spec_width=spec_width,
            num_classes=num_classes,
            pretrained=pretrained,
            lock_pretrained=lock_pretrained
        )


class STFTResNeXtWithAttention(_STFTResNetWithAttention):

    loading_func = staticmethod(tv.models.resnext50_32x4d)

    def __init__(self,
                 n_fft: int = 256,
                 hop_length: Optional[int] = None,
                 win_length: Optional[int] = None,
                 window: Optional[str] = None,
                 normalized: bool = False,
                 onesided: bool = True,
                 spec_height: int = 224,
                 spec_width: int = 224,
                 num_classes: int = 1000,
                 apply_attention: bool = False,
                 pretrained: Union[bool, str] = False,
                 lock_pretrained: Optional[Union[bool, List[str]]] = None):

        super(STFTResNeXtWithAttention, self).__init__(
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            apply_attention=apply_attention,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            normalized=normalized,
            onesided=onesided,
            spec_height=spec_height,
            spec_width=spec_width,
            num_classes=num_classes,
            pretrained=pretrained,
            lock_pretrained=lock_pretrained,
            groups=32,
            width_per_group=4
        )