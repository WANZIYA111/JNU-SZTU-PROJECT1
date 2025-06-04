"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import fastmri
from fastmri.data import transforms
# from fast_sense import SENSE
from .unet import Unet

import torch
import math
# from functorch import vmap
from torch.func import vmap
def process_batch(csm_b, acc_factor, noise_matrix_inv, regularization_factor):
    unmix = vmap(
    lambda csm_slice: ismrm_calculate_sense_unmixing_1d(
        acc_factor,
        csm_slice,
        noise_matrix_inv,
        regularization_factor
    ),
    in_dims=1,    # 对csm_b的第1维(Y)做向量化
    out_dims=1    # 保持结果在第二维(Y)
    )(csm_b)
    return unmix

def ismrm_calculate_sense_unmixing_general2(acc_factor, csm, noise_matrix=None, regularization_factor=None):
    # 参数维度检查调整为4维
    assert csm.dim() == 4, "coil sensitivity map must have 4 dimensions [batch, X, Y, COIL]"
    
    if noise_matrix is None:
        num_coils = csm.size(3)  # 现在形状是 [batch, X, Y, COIL]
        noise_matrix = torch.eye(num_coils,dtype=csm.dtype,device=csm.device)
    
    if regularization_factor is None:
        regularization_factor = 0.00
    device=csm.device    
    noise_matrix_inv = torch.linalg.pinv(noise_matrix).to(device)
    
    # 添加batch维度处理
    batch_size = csm.size(0)
    unmix = torch.zeros_like(csm)
    
    unmix = vmap(
        process_batch,
        in_dims=(0, None, None, None)  # 对 csm 的 batch 维度（dim=0）做向量化
        )(csm, acc_factor, noise_matrix_inv, regularization_factor)
    return unmix
    
def ismrm_calculate_sense_unmixing_1d(acc_factor, csm1d, noise_matrix_inv, regularization_factor):
    # 输入csm1d形状应为 [kx, coil]
    # print(csm1d.shape)   #torch.Size([384, 16])
    ny, nc = csm1d.shape
    if ny % acc_factor != 0:
        raise ValueError("ny must be a multiple of acc_factor")

    n_blocks = ny // acc_factor
    base_indices = torch.arange(n_blocks.item(), device=csm1d.device)
    offsets = torch.arange(0, ny, n_blocks.item(), device=csm1d.device)
    indices = base_indices[:, None] + offsets[None, :]##torch.Size([128, 3])
    block_csm = csm1d[indices,:]##torch.Size([128, 3, 16])
    def process_block(block_csm1d):
        A = block_csm1d.mT  # [nc, k]=[16,3]
        AHA = A.conj().T @ noise_matrix_inv @ A
        diag_AHA = torch.diag(AHA)
        reduced_eye = torch.diag((torch.abs(diag_AHA) > 0).float())
        
        n_alias = torch.sum(reduced_eye)
        scaled_reg_factor = regularization_factor * torch.trace(AHA) / n_alias
        
        inv_term = torch.linalg.pinv(AHA + reduced_eye * scaled_reg_factor)
        return inv_term @ A.conj().T @ noise_matrix_inv

    all_blocks = vmap(process_block,in_dims=0)(block_csm) #torch.Size([128, 3, 16])
    unmix1d = torch.zeros((ny, nc), dtype=csm1d.dtype, device=csm1d.device)
    unmix1d = all_blocks.permute(1, 0, 2).reshape(ny, nc)
    
    return unmix1d

def ismrm_transform_kspace_to_image(k, dim=[2,3], img_shape=None):
    if img_shape is None:
        img_shape = k.shape  # 保持batch维度不变
    else:
        img_shape = tuple(img_shape)
    
    img = k.clone()
    for d in dim:
        # 调整dim参数处理，跳过batch维度
        img = torch.fft.ifftshift(img, dim=d)  # dim+1跳过batch维度
        img = torch.fft.ifft(img, n=img_shape[d], dim=d, norm="ortho")
        img = torch.fft.fftshift(img, dim=d)
    
    return img

def SENSE(inp, csm, acc_factor, replicas=100, reg=None):
    # 输入inp形状应为 [batch, kx, ky, coil]
    # 输入csm形状应为 [batch, kx, ky, coil]
    device=inp.device
    csm = csm.to(device)
    acc_factor = acc_factor.to(device)
    unmix_sense = ismrm_calculate_sense_unmixing_general2(acc_factor, csm, None, reg).to(device)
    
    scaling_factor = torch.sqrt(torch.prod(acc_factor)).to(device)
    
    # 处理batch维度的FFT
    img_alias = scaling_factor * ismrm_transform_kspace_to_image(inp.to(device), [1,2])
    
    # 合并通道时保持batch维度
    img = torch.sum(img_alias * unmix_sense, dim=3)  # 在dim=3(coil)上求和
    return img

class NormUnet(nn.Module):
    """
    Normalized U-Net model.

    This is the same as a regular U-Net, but with normalization applied to the
    input before the U-Net. This keeps the values more numerically stable
    during training.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, 2, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        x = self.unet(x)

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        return x


class SensitivityModel(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.

    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end variational network.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        mask_center: bool = True,
    ):
       
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()
        self.mask_center = mask_center
        self.norm_unet = NormUnet(
            chans,
            num_pools,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=drop_prob,
        )

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape

        return x.view(b * c, 1, h, w, comp), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size

        return x.view(batch_size, c, h, w, comp)

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        return x / fastmri.rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    def get_pad_and_num_low_freqs(
        self, mask: torch.Tensor, num_low_frequencies: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_low_frequencies is None or num_low_frequencies == 0:
            # get low frequency line locations and mask them out
            squeezed_mask = mask[:, 0, 0, :, 0].to(torch.int8)
            cent = squeezed_mask.shape[1] // 2
            # running argmin returns the first non-zero
            left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
            right = torch.argmin(squeezed_mask[:, cent:], dim=1)
            num_low_frequencies_tensor = torch.max(
                2 * torch.min(left, right), torch.ones_like(left)
            )  # force a symmetric center unless 1
        else:
            num_low_frequencies_tensor = num_low_frequencies * torch.ones(
                mask.shape[0], dtype=mask.dtype, device=mask.device
            )

        pad = (mask.shape[-2] - num_low_frequencies_tensor + 1) // 2

        return pad.type(torch.long), num_low_frequencies_tensor.type(torch.long)

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ) -> torch.Tensor:
        device = masked_kspace.device
        if self.mask_center:
            pad, num_low_freqs = self.get_pad_and_num_low_freqs(
                mask, num_low_frequencies
            )
            masked_kspace = transforms.batched_mask_center(
                masked_kspace, pad, pad + num_low_freqs
            )
        pd = 30
        ACS_MASK = torch.zeros(masked_kspace.shape).to(device)
        h = int(masked_kspace.shape[-3])
        w = int(masked_kspace.shape[-2])
        ACS_MASK[...,int(h/2)-pd:int(h/2)+pd,int(w/2)-pd:int(w/2)+pd,:] = 1
        ACS_kspace = masked_kspace*ACS_MASK
        # convert to image space
        images, batches = self.chans_to_batch_dim(fastmri.ifft2c(ACS_kspace.to(device)))

        # estimate sensitivities
        return self.divide_root_sum_of_squares(
            self.batch_chans_to_chan_dim(self.norm_unet(images), batches)
        ),ACS_kspace


class VarNet(nn.Module):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        mask_center: bool = True,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()

        self.sens_net = SensitivityModel(
            chans=sens_chans,
            num_pools=sens_pools,
            mask_center=mask_center,
        )
        

    def forward(
        self,
        masked_kspace: torch.Tensor,
        real_masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        real_mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ) -> torch.Tensor:
        device = real_masked_kspace.device
        sens_maps,ACS_kspace = self.sens_net(masked_kspace, mask, num_low_frequencies)
#         kspace_pred = real_masked_kspace.clone()

#         for cascade in self.cascades:
#             kspace_pred = cascade(kspace_pred,real_masked_kspace, real_mask, sens_maps)
        #real_masked_kspace (1, 16, 384, 384,2)
        #real_mask (1, 16, 384, 384,2)
        #acc_factor 3
        #sens_map(1, 16, 384, 384,2)
        R = 3
        acc_factor = torch.tensor(R)
        sense_kspace = torch.view_as_complex(real_masked_kspace).permute(0,2,3,1)
        # sense_mask = torch.view_as_complex(real_mask).transpose(0,2,3,1)
        sense_sens = torch.view_as_complex(sens_maps).permute(0,2,3,1)
        image = SENSE(sense_kspace,sense_sens,acc_factor)#image [batch,x,y]
        ## 输入inp和csm形状应为 [batch, kx, ky, coil]
        return (abs(image).float())/(abs(image).max()),sens_maps,ACS_kspace,real_masked_kspace


