# coding: UTF-8
"""
    @date:  2025.01
    @func:  ESD Loss calculation for Infinity Autoregressive Model
            将 EraseAnything 的 ESD loss 迁移到 Infinity 模型
"""

import random
import torch
import torch.nn.functional as F
from typing import Tuple, List
from .esd_utils import autoregressive_sample, predict_next_scale


def calculate_esd_loss(
    args,
    batch: dict,
    gpt_model,           # Infinity transformer model
    vae_model,           # Infinity VAE model  
    text_features: torch.Tensor,        # T5 encoded text features
    text_lens: List[int],               # text sequence lengths
    cu_seqlens_k: torch.Tensor,         # cumulative sequence lengths
    criteria: torch.nn.Module,          # loss function (MSE)
    negative_guidance: float = 1.0,
    start_guidance: float = 3.0,
    ddim_steps: int = 28,
    device: str = "cuda",
    weight_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Calculate ESD (Erased Stable Diffusion) loss for Infinity model
    
    ESD loss 的核心思想：
    e_n = predict_noise(x_t, prompt)  # 当前模型预测
    e_0 = predict_noise(x_t, "")      # 无条件预测
    e_p = predict_noise(x_t, prompt)  # 原始预测（不需要梯度）
    
    Loss = MSE(e_n, e_0 - negative_guidance * (e_p - e_0))
    这样可以让模型学习到：当看到要擦除的概念时，输出接近无条件生成
    
    Args:
        args: 配置参数
        batch: 数据批次，包含 pixel_values, prompts 等
        gpt_model: Infinity GPT 模型
        vae_model: Infinity VAE 模型
        text_features: T5 编码的文本特征
        text_lens: 文本长度列表
        cu_seqlens_k: 累积序列长度
        criteria: 损失函数
        negative_guidance: 负向引导强度
        start_guidance: 起始引导强度
        ddim_steps: DDIM 采样步数
        device: 设备
        weight_dtype: 权重数据类型
        
    Returns:
        esd_loss: ESD 损失值
    """
    
    # ==================== 1. 图像编码到 latent space ====================
    pixel_values = batch["pixel_values"].to(dtype=weight_dtype, device=device)
    B = pixel_values.shape[0]
    
    with torch.no_grad():
        # Infinity VAE encode: 获取多尺度 latent features
        # Infinity 使用 multi-scale 编码，与 Flux 不同
        vae_model.eval()
        # 根据 Infinity 的编码方式获取 latent
        if hasattr(vae_model, 'encode_for_raw_features'):
            # 使用 Infinity 的多尺度编码
            scale_schedule = args.scale_schedule if hasattr(args, 'scale_schedule') else [(1, 1, 1), (1, 2, 2), (1, 4, 4), (1, 8, 8)]
            raw_features, _, _ = vae_model.encode_for_raw_features(pixel_values, scale_schedule=scale_schedule)
            # 量化得到 latent codes
            ms_idx_Bl = vae_model.get_next_autoregressive_input(pixel_values)
        else:
            # 如果没有这个方法，使用基本编码
            latent_dist = vae_model.encode(pixel_values)
            model_input = latent_dist if hasattr(latent_dist, 'sample') else latent_dist
    
    # ==================== 2. 准备文本条件 ====================
    # emb_0: 空文本条件（用于 ESD）
    # emb_p: 正常文本条件（包含要擦除的概念）
    
    # 获取空文本的 embedding
    empty_prompts = [""] * B
    with torch.no_grad():
        # 假设已经通过 T5 编码得到 text_features
        # 需要获取空文本的 features
        # 这里需要外部传入或者使用 gpt_model 的 cfg_uncond
        if hasattr(gpt_model, 'cfg_uncond'):
            # Infinity 模型有 cfg_uncond buffer
            text_features_0 = gpt_model.cfg_uncond.unsqueeze(0).expand(B, -1, -1).to(device)
            text_lens_0 = [gpt_model.cfg_uncond.shape[0]] * B
            cu_seqlens_k_0 = torch.cat([torch.tensor([0]), torch.tensor(text_lens_0).cumsum(0)], dim=0).to(device=device, dtype=torch.int32)
        else:
            # 如果没有，使用全零
            text_features_0 = torch.zeros_like(text_features)
            text_lens_0 = text_lens
            cu_seqlens_k_0 = cu_seqlens_k
    
    text_features_p = text_features.clone()
    text_lens_p = text_lens.copy()
    cu_seqlens_k_p = cu_seqlens_k.clone()
    
    # ==================== 3. 采样一个随机时间步 ====================
    # 类似 Flux 的 ESD，我们需要在某个时间步进行预测
    # 对于 Infinity，时间步对应的是 scale index
    t_enc = torch.randint(1, ddim_steps, (1,), device=device)
    
    # Infinity 使用 scale-wise 生成，不是 timestep-based
    # 我们需要采样到某个中间 scale
    # 假设有 10 个 scales，我们随机选一个中间的 scale
    num_scales = len(args.scale_schedule) if hasattr(args, 'scale_schedule') else 10
    target_scale = random.randint(1, min(5, num_scales - 1))  # 选择前半部分的 scale
    
    # ==================== 4. 生成中间状态 z (类似 DDIM 采样到 t) ====================
    with torch.no_grad():
        # 使用 autoregressive sampling 生成到 target_scale
        # 这部分需要实现 Infinity 的部分采样
        z_intermediate, prev_tokens = autoregressive_sample(
            gpt_model=gpt_model,
            vae_model=vae_model,
            text_cond_tuple=(text_features_p, text_lens_p, cu_seqlens_k_p, max(text_lens_p)),
            target_scale=target_scale,
            scale_schedule=args.scale_schedule if hasattr(args, 'scale_schedule') else None,
            device=device,
            dtype=weight_dtype,
            cfg_scale=start_guidance,
            batch_size=B,
            resolution=args.resolution,
        )
        
        # ==================== 5. 预测下一个 scale 的 tokens ====================
        # e_0: 无条件预测
        e_0 = predict_next_scale(
            gpt_model=gpt_model,
            prev_tokens=prev_tokens,
            text_cond_tuple=(text_features_0, text_lens_0, cu_seqlens_k_0, max(text_lens_0)),
            current_scale=target_scale,
            scale_schedule=args.scale_schedule if hasattr(args, 'scale_schedule') else None,
            cfg_scale=start_guidance,
            device=device,
            dtype=weight_dtype,
        )
        
        # e_p: 有条件预测（原始模型，不需要梯度）
        e_p = predict_next_scale(
            gpt_model=gpt_model,
            prev_tokens=prev_tokens,
            text_cond_tuple=(text_features_p, text_lens_p, cu_seqlens_k_p, max(text_lens_p)),
            current_scale=target_scale,
            scale_schedule=args.scale_schedule if hasattr(args, 'scale_schedule') else None,
            cfg_scale=start_guidance,
            device=device,
            dtype=weight_dtype,
        )
        
        e_0.requires_grad = False
        e_p.requires_grad = False
    
    # ==================== 6. 当前模型的预测（需要梯度）====================
    # e_n: 当前正在训练的模型的预测
    e_n = predict_next_scale(
        gpt_model=gpt_model,
        prev_tokens=prev_tokens,
        text_cond_tuple=(text_features_p, text_lens_p, cu_seqlens_k_p, max(text_lens_p)),
        current_scale=target_scale,
        scale_schedule=args.scale_schedule if hasattr(args, 'scale_schedule') else None,
        cfg_scale=start_guidance,
        device=device,
        dtype=weight_dtype,
    )
    
    # ==================== 7. 计算 ESD loss ====================
    # 目标：让 e_n 接近 e_0 - negative_guidance * (e_p - e_0)
    # 这样当模型看到要擦除的概念时，会输出更接近无条件生成的结果
    target = e_0 - negative_guidance * (e_p - e_0)
    
    # 根据 Infinity 使用 bit prediction 或 token prediction
    if args.use_bit_label:
        # bit-wise loss
        loss_esd = criteria(e_n, target)
        loss_esd = loss_esd.mean()
    else:
        # token-wise loss
        loss_esd = criteria(e_n, target)
        loss_esd = loss_esd.mean()
    
    return loss_esd


def calculate_esd_loss_simple(
    args,
    batch: dict,
    gpt_model,
    vae_model,
    text_cond_tuple: Tuple,
    criteria: torch.nn.Module,
    negative_guidance: float = 1.0,
    device: str = "cuda",
    weight_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    简化版 ESD loss - 直接在训练时的 forward pass 上计算
    
    这个版本不需要采样，而是直接在训练的 forward pass 中：
    1. 同时用有条件和无条件文本进行 forward
    2. 计算 ESD loss
    
    这样更高效，适合 Infinity 的训练流程
    """
    pixel_values = batch["pixel_values"].to(dtype=weight_dtype, device=device)
    B = pixel_values.shape[0]
    
    # 解包 text condition tuple
    text_features, text_lens, cu_seqlens_k, Ltext = text_cond_tuple
    
    # ==================== 1. 编码图像 ====================
    with torch.no_grad():
        vae_model.eval()
        # 获取 ground truth tokens
        gt_ms_idx_Bl = vae_model.get_next_autoregressive_input(pixel_values)
        
        # 获取输入特征（前一个 scale 的特征）
        if hasattr(vae_model, 'encode_for_raw_features'):
            scale_schedule = args.scale_schedule if hasattr(args, 'scale_schedule') else [(1,1,1), (1,2,2), (1,4,4)]
            raw_features, _, _ = vae_model.encode_for_raw_features(pixel_values, scale_schedule=scale_schedule)
            
            # 使用 bitwise self-correction 获取输入
            from infinity.models.bitwise_self_correction import BitwiseSelfCorrection
            bsc = BitwiseSelfCorrection(vae_model, args)
            x_BLC_wo_prefix, _ = bsc.flip_requant(scale_schedule, pixel_values, raw_features, device)
        else:
            # 简化版：直接使用 tokens 的 embedding
            gt_BL = torch.cat(gt_ms_idx_Bl, dim=1)
            x_BLC_wo_prefix = gpt_model.word_embed(gt_BL[:, :-1])  # 去掉最后一个 token
    
    # ==================== 2. 准备文本条件 ====================
    # 获取无条件文本特征
    if hasattr(gpt_model, 'cfg_uncond'):
        text_features_uncond = gpt_model.cfg_uncond.unsqueeze(0).expand(B, -1, -1)
        # 创建对应的 lens 和 cu_seqlens
        uncond_len = text_features_uncond.shape[1]
        text_lens_uncond = [uncond_len] * B
        cu_seqlens_k_uncond = F.pad(torch.tensor(text_lens_uncond).to(dtype=torch.int32).cumsum_(0), (1, 0)).to(device)
        Ltext_uncond = uncond_len
        text_cond_tuple_uncond = (text_features_uncond, text_lens_uncond, cu_seqlens_k_uncond, Ltext_uncond)
    else:
        # 如果没有 cfg_uncond，使用零向量
        text_features_uncond = torch.zeros_like(text_features)
        text_cond_tuple_uncond = (text_features_uncond, text_lens, cu_seqlens_k, Ltext)
    
    # ==================== 3. Forward pass ====================
    # 随机选择一个训练的 scale
    if hasattr(args, 'scale_schedule'):
        training_scales = min(args.always_training_scales, len(args.scale_schedule))
        target_scale_idx = random.randint(1, training_scales - 1)
        scale_schedule = args.scale_schedule[:target_scale_idx + 1]
    else:
        scale_schedule = [(1,1,1), (1,2,2), (1,4,4)]
        target_scale_idx = 2
    
    # 截断输入到目标 scale
    target_len = sum([t*h*w for t,h,w in scale_schedule[:-1]])
    x_BLC = x_BLC_wo_prefix[:, :target_len, :]
    
    # 有条件 forward (需要梯度)
    logits_BLV_cond = gpt_model(text_cond_tuple, x_BLC, scale_schedule=scale_schedule)
    
    # 无条件 forward (不需要梯度)
    with torch.no_grad():
        logits_BLV_uncond = gpt_model(text_cond_tuple_uncond, x_BLC, scale_schedule=scale_schedule)
    
    # ==================== 4. 计算 ESD loss ====================
    # 获取目标 scale 的预测
    # 只计算最后一个 scale 的 loss
    last_scale_len = scale_schedule[-1][0] * scale_schedule[-1][1] * scale_schedule[-1][2]
    
    # 提取最后一个 scale 的 logits
    logits_cond = logits_BLV_cond[:, -last_scale_len:, :]
    logits_uncond = logits_BLV_uncond[:, -last_scale_len:, :]
    
    # ESD 目标：让有条件的预测接近无条件预测
    # 使用 KL 散度或 MSE
    if args.use_bit_label:
        # bit-wise prediction
        # 转换为 probability
        prob_cond = F.softmax(logits_cond, dim=-1)
        prob_uncond = F.softmax(logits_uncond, dim=-1)
        # MSE on probabilities
        loss_esd = F.mse_loss(prob_cond, prob_uncond.detach())
    else:
        # token-wise prediction  
        # 使用 KL divergence
        log_prob_cond = F.log_softmax(logits_cond, dim=-1)
        prob_uncond = F.softmax(logits_uncond, dim=-1)
        loss_esd = F.kl_div(log_prob_cond, prob_uncond.detach(), reduction='batchmean')
    
    return loss_esd

