"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser
import numpy as np
import torch
import os
import fastmrisense
from fastmrisense.data import transforms
from fastmrisense.models import VarNet

from .mri_module import MriModule


class VarNetModule(MriModule):
    """
    VarNet training module.

    This can be used to train variational networks from the paper:

    A. Sriram et al. End-to-end variational networks for accelerated MRI
    reconstruction. In International Conference on Medical Image Computing and
    Computer-Assisted Intervention, 2020.

    which was inspired by the earlier paper:

    K. Hammernik et al. Learning a variational network for reconstruction of
    accelerated MRI data. Magnetic Resonance inMedicine, 79(6):3055–3071, 2018.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        pools: int = 4,
        chans: int = 18,
        sens_pools: int = 4,
        sens_chans: int = 8,
        lr: float = 0.0003,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊")
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            chans: Number of channels for cascade U-Net.
            sens_pools: Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            sens_chans: Number of channels for sensitivity map U-Net.
            lr: Learning rate.
            lr_step_size: Learning rate step size.
            lr_gamma: Learning rate gamma decay.
            weight_decay: Parameter for penalizing weights norm.
            num_sense_lines: Number of low-frequency lines to use for sensitivity map
                computation, must be even or `None`. Default `None` will automatically
                compute the number from masks. Default behaviour may cause some slices to
                use more low-frequency lines than others, when used in conjunction with
                e.g. the EquispacedMaskFunc defaults. To prevent this, either set
                `num_sense_lines`, or set `skip_low_freqs` and `skip_around_low_freqs`
                to `True` in the EquispacedMaskFunc. Note that setting this value may
                lead to undesired behaviour when training on multiple accelerations
                simultaneously.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.num_cascades = num_cascades
        self.pools = pools
        self.chans = chans
        self.sens_pools = sens_pools
        self.sens_chans = sens_chans
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        self.varnet = VarNet(
            num_cascades=self.num_cascades,
            sens_chans=self.sens_chans,
            sens_pools=self.sens_pools,
            chans=self.chans,
            pools=self.pools,
        )

        self.loss = fastmrisense.SSIMLoss()

    def forward(self, masked_kspace,real_masked_kspace, mask, real_mask,num_low_frequencies):
        return self.varnet(masked_kspace,real_masked_kspace, mask, real_mask,num_low_frequencies)

    def training_step(self, batch, batch_idx):
        output,sens_maps,acs_kspace0,masked_kspace0= self(batch.masked_kspace, batch.real_masked_kspace,batch.mask, batch.real_mask,batch.num_low_frequencies)

        target, output = transforms.center_crop_to_smallest(batch.target, output)
        # loss = self.loss(
        #     output.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value
        # )
        #sens_maps --(1,16,384,384,2)--*(batch,coil,kx,ky,complex)
        sens_real = torch.view_as_complex(sens_maps).real
        sens_imag = torch.view_as_complex(sens_maps).imag
        gold_real = (batch.gold_sens).real
        gold_imag = (batch.gold_sens).imag
        
        mse_real = torch.nn.functional.mse_loss(sens_real, gold_real)
        mse_imag = torch.nn.functional.mse_loss(sens_imag, gold_imag)
        mse_loss = mse_real + mse_imag
        
        self.log("train_loss", mse_loss)

        return mse_loss

    def validation_step(self, batch, batch_idx):
        output,sens_maps,acs_kspace0,masked_kspace0= self.forward(batch.masked_kspace, batch.real_masked_kspace,batch.mask, batch.real_mask,batch.num_low_frequencies)
        target, output = transforms.center_crop_to_smallest(batch.target, output)
        # print("batch.gold_sens",batch.gold_sens.shape)
        # print("sens_maps",sens_maps.shape)
        output_dir = "output"
        sens_maps_dir = os.path.join(output_dir, "sens_maps")
        recon_dir = os.path.join(output_dir, "recon")
        target_dir = os.path.join(output_dir, "target")

        # 创建目录（如果不存在）
        os.makedirs(sens_maps_dir, exist_ok=True)
        os.makedirs(recon_dir, exist_ok=True)
        os.makedirs(target_dir, exist_ok=True)

        # 构造文件路径
        sens_filename = os.path.join(sens_maps_dir, f'{batch.fname[0]}_{batch.slice_num[0]}.npy')
        recon_filename = os.path.join(recon_dir, f'{batch.fname[0]}_{batch.slice_num[0]}.npy')
        GT_filename = os.path.join(target_dir, f'{batch.fname[0]}_{batch.slice_num[0]}.npy')

        # 保存文件
        np.save(sens_filename, torch.view_as_complex(sens_maps).detach().cpu().numpy())
        np.save(recon_filename, output.detach().cpu().numpy())
        np.save(GT_filename, target.detach().cpu().numpy())
        
        sens_real = torch.view_as_complex(sens_maps).real
        sens_imag = torch.view_as_complex(sens_maps).imag
        gold_real = (batch.gold_sens).real
        gold_imag = (batch.gold_sens).imag
        
        mse_real = torch.nn.functional.mse_loss(sens_real, gold_real)
        mse_imag = torch.nn.functional.mse_loss(sens_imag, gold_imag)
        mse_loss = mse_real + mse_imag
        
        self.log("train_loss", mse_loss)
        np.savez('tmp.npz',sens_maps=torch.view_as_complex(sens_maps).detach().cpu().numpy(),gold_sens=(batch.gold_sens).detach().cpu().numpy(),recon=output.detach().cpu().numpy(),kspace_input_sensmap=torch.view_as_complex(acs_kspace0).detach().cpu().numpy(),kspace_input_varnet=torch.view_as_complex(masked_kspace0).detach().cpu().numpy(),target = target.detach().cpu().numpy())
        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output,
            "target": target,
            "val_loss": mse_loss,
        }

    def test_step(self, batch, batch_idx):
        output,sens_maps,acs_kspace0,masked_kspace0= self(batch.masked_kspace, batch.real_masked_kspace,batch.mask, batch.real_mask,batch.num_low_frequencies)

        # check for FLAIR 203
        if output.shape[-1] < batch.crop_size[1]:
            crop_size = (output.shape[-1], output.shape[-1])
        else:
            crop_size = batch.crop_size

        output = transforms.center_crop(output, crop_size)

        return {
            "fname": batch.fname,
            "slice": batch.slice_num,
            "output": output.cpu().numpy(),
        }

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # param overwrites

        # network params
        parser.add_argument(
            "--num_cascades",
            default=12,
            type=int,
            help="Number of VarNet cascades",
        )
        parser.add_argument(
            "--pools",
            default=4,
            type=int,
            help="Number of U-Net pooling layers in VarNet blocks",
        )
        parser.add_argument(
            "--chans",
            default=18,
            type=int,
            help="Number of channels for U-Net in VarNet blocks",
        )
        parser.add_argument(
            "--sens_pools",
            default=4,
            type=int,
            help="Number of pooling layers for sense map estimation U-Net in VarNet",
        )
        parser.add_argument(
            "--sens_chans",
            default=8,
            type=float,
            help="Number of channels for sense map estimation U-Net in VarNet",
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=0.0003, type=float, help="Adam learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma",
            default=0.1,
            type=float,
            help="Extent to which step size should be decreased",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )

        return parser
