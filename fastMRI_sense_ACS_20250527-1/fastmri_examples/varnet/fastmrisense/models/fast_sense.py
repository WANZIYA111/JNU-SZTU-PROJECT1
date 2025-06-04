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
