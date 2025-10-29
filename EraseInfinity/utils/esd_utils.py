# coding: UTF-8
"""
    @date:  2025.01
    @func:  ESD utilities for Infinity Autoregressive Model
            为 Infinity 模型实现 ESD 相关的辅助函数
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional


def autoregressive_sample(
    gpt_model,
    vae_model,
    text_cond_tuple: Tuple,
    target_scale: int,
    scale_schedule: Optional[List] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    cfg_scale: float = 7.5,
    batch_size: int = 1,
    resolution: int = 256,
    top_p: float = 0.9,
    top_k: int = 900,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Infinity 模型的自回归采样到目标 scale
    
    Args:
        gpt_model: Infinity GPT transformer 模型
        vae_model: Infinity VAE 模型
        text_cond_tuple: 文本条件 (text_features, text_lens, cu_seqlens_k, Ltext)
        target_scale: 目标 scale 索引
        scale_schedule: scale 调度表
        device: 设备
        dtype: 数据类型
        cfg_scale: classifier-free guidance scale
        batch_size: batch size
        resolution: 图像分辨率
        top_p: top-p sampling
        top_k: top-k sampling
        
    Returns:
        z_intermediate: 中间状态的 latent
        prev_tokens: 之前生成的所有 tokens
    """
    
    if scale_schedule is None:
        # 默认 scale schedule
        scale_schedule = [(1, 1, 1), (1, 2, 2), (1, 3, 3), (1, 4, 4), (1, 6, 6), (1, 8, 8), (1, 10, 10), (1, 13, 13), (1, 16, 16)]
    
    # 确保 target_scale 不超过 schedule 长度
    target_scale = min(target_scale, len(scale_schedule) - 1)
    
    # 准备初始输入
    text_features, text_lens, cu_seqlens_k, Ltext = text_cond_tuple
    
    # ==================== 自回归生成 ====================
    gpt_model.eval()
    
    with torch.no_grad():
        # 初始化：第一个 scale 是 SOS token
        # Infinity 从最粗糙的 scale 开始生成
        all_tokens = []
        
        # 逐 scale 生成
        for scale_idx in range(target_scale + 1):
            current_scale_schedule = scale_schedule[:scale_idx + 1]
            t, h, w = scale_schedule[scale_idx]
            scale_len = t * h * w
            
            if scale_idx == 0:
                # 第一个 scale：使用 text pooling 作为 SOS
                # 创建空输入
                x_BLC = torch.zeros(batch_size, 0, gpt_model.C, device=device, dtype=dtype)
            else:
                # 后续 scale：使用之前生成的 tokens
                # 将 tokens 转换为 embeddings
                prev_tokens_cat = torch.cat(all_tokens, dim=1)
                
                if gpt_model.use_bit_label:
                    # bit label: 需要将 bit predictions 转换为 token indices
                    # prev_tokens_cat shape: [B, L, codebook_dim] (bits)
                    # 转换为 indices
                    token_indices = bits_to_indices(prev_tokens_cat, gpt_model.bit_mask)
                else:
                    token_indices = prev_tokens_cat
                
                # 通过 VAE 的 quantizer 获取 features
                x_BLC = vae_model.quantize.get_next_autoregressive_input(token_indices)
            
            # Forward pass
            logits_BLV = gpt_model(
                text_cond_tuple,
                x_BLC,
                scale_schedule=current_scale_schedule
            )
            
            # 提取当前 scale 的 logits
            current_logits = logits_BLV[:, -scale_len:, :]
            
            # Sampling
            if gpt_model.use_bit_label:
                # bit-wise sampling
                # logits shape: [B, L, codebook_dim * 2]
                B, L, _ = current_logits.shape
                current_logits = current_logits.reshape(B, L, gpt_model.codebook_dim, 2)
                # sample each bit
                sampled_bits = sample_categorical(current_logits, top_p=top_p, top_k=top_k)
                all_tokens.append(sampled_bits)
            else:
                # token-wise sampling
                sampled_tokens = sample_categorical(current_logits, top_p=top_p, top_k=top_k)
                all_tokens.append(sampled_tokens)
        
        # 合并所有生成的 tokens
        prev_tokens = torch.cat(all_tokens, dim=1)
        
        # 将 tokens 解码为 latent (可选)
        if gpt_model.use_bit_label:
            token_indices = bits_to_indices(prev_tokens, gpt_model.bit_mask)
        else:
            token_indices = prev_tokens
        
        # 通过 VAE 解码（可选，用于可视化）
        # z_intermediate = vae_model.decode_to_features(token_indices)
        z_intermediate = None  # 为了节省内存，这里不解码
    
    return z_intermediate, prev_tokens


def predict_next_scale(
    gpt_model,
    prev_tokens: torch.Tensor,
    text_cond_tuple: Tuple,
    current_scale: int,
    scale_schedule: Optional[List] = None,
    cfg_scale: float = 7.5,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    预测下一个 scale 的 tokens
    
    Args:
        gpt_model: Infinity GPT 模型
        prev_tokens: 之前生成的所有 tokens [B, L_prev, ...]
        text_cond_tuple: 文本条件
        current_scale: 当前要预测的 scale 索引
        scale_schedule: scale 调度表
        cfg_scale: classifier-free guidance scale
        device: 设备
        dtype: 数据类型
        
    Returns:
        predicted_logits: 预测的 logits [B, L_current_scale, V]
    """
    
    if scale_schedule is None:
        scale_schedule = [(1, 1, 1), (1, 2, 2), (1, 3, 3), (1, 4, 4), (1, 6, 6), (1, 8, 8)]
    
    text_features, text_lens, cu_seqlens_k, Ltext = text_cond_tuple
    
    # 准备输入：将 prev_tokens 转换为 features
    B = prev_tokens.shape[0]
    
    if gpt_model.use_bit_label:
        # bit label: 需要通过 VAE 转换
        token_indices = bits_to_indices(prev_tokens, gpt_model.bit_mask)
    else:
        token_indices = prev_tokens
    
    # 获取输入 features
    # 这需要 VAE 的 quantizer
    # 简化版：直接使用 word_embed
    if hasattr(gpt_model, 'word_embed'):
        # token_indices: [B, L]
        if gpt_model.use_bit_label:
            # 对于 bit label，需要特殊处理
            x_BLC = gpt_model.word_embed(token_indices)
        else:
            x_BLC = gpt_model.word_embed(token_indices)
    else:
        # 如果没有 word_embed，需要通过 VAE
        raise NotImplementedError("需要实现通过 VAE 获取 features 的逻辑")
    
    # Forward pass
    current_scale_schedule = scale_schedule[:current_scale + 1]
    logits_BLV = gpt_model(
        text_cond_tuple,
        x_BLC,
        scale_schedule=current_scale_schedule
    )
    
    # 提取当前 scale 的预测
    t, h, w = scale_schedule[current_scale]
    scale_len = t * h * w
    predicted_logits = logits_BLV[:, -scale_len:, :]
    
    return predicted_logits


def sample_categorical(
    logits: torch.Tensor,
    top_p: float = 0.9,
    top_k: int = 900,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    从 categorical distribution 中采样
    
    Args:
        logits: [B, L, V] or [B, L, D, 2] (for bit prediction)
        top_p: top-p sampling
        top_k: top-k sampling
        temperature: temperature for softmax
        
    Returns:
        samples: 采样结果
    """
    
    # Apply temperature
    logits = logits / temperature
    
    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
    
    # Top-p filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter sorted tensors back to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
    
    # Sample from the filtered distribution
    probs = F.softmax(logits, dim=-1)
    samples = torch.multinomial(probs.reshape(-1, probs.shape[-1]), num_samples=1)
    samples = samples.reshape(*probs.shape[:-1], 1).squeeze(-1)
    
    return samples


def bits_to_indices(bits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    将 bit representation 转换为 token indices
    
    Args:
        bits: [B, L, codebook_dim] bit values (0 or 1)
        mask: [codebook_dim] mask for each bit position
        
    Returns:
        indices: [B, L] token indices
    """
    # bits: [B, L, D], each element is 0 or 1
    # mask: [D], e.g., [1, 2, 4, 8, 16, ...] for binary positions
    
    # Ensure bits are 0 or 1
    bits = bits.long()
    
    # Convert bits to indices using mask
    # indices = sum(bit_i * mask_i)
    indices = (bits * mask.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
    
    return indices


def indices_to_bits(indices: torch.Tensor, mask: torch.Tensor, codebook_dim: int) -> torch.Tensor:
    """
    将 token indices 转换为 bit representation
    
    Args:
        indices: [B, L] token indices
        mask: [codebook_dim] mask for each bit position
        codebook_dim: number of bits
        
    Returns:
        bits: [B, L, codebook_dim] bit values
    """
    # Extract bits using mask
    # For each bit position, check if (index & mask_i) != 0
    
    B, L = indices.shape
    bits = torch.zeros(B, L, codebook_dim, dtype=torch.long, device=indices.device)
    
    for i in range(codebook_dim):
        bits[:, :, i] = (indices & mask[i]) != 0
    
    return bits


def get_default_scale_schedule(resolution: int = 256) -> List[Tuple[int, int, int]]:
    """
    根据分辨率获取默认的 scale schedule
    
    Args:
        resolution: 图像分辨率
        
    Returns:
        scale_schedule: [(t, h, w), ...] 列表
    """
    
    if resolution == 256:
        # 256x256: pn=0.06M
        scale_schedule = [
            (1, 1, 1),
            (1, 2, 2),
            (1, 3, 3),
            (1, 4, 4),
            (1, 6, 6),
            (1, 8, 8),
            (1, 10, 10),
            (1, 13, 13),
            (1, 16, 16),
        ]
    elif resolution == 512:
        # 512x512: pn=0.25M
        scale_schedule = [
            (1, 1, 1),
            (1, 2, 2),
            (1, 3, 3),
            (1, 4, 4),
            (1, 6, 6),
            (1, 8, 8),
            (1, 10, 10),
            (1, 13, 13),
            (1, 16, 16),
            (1, 20, 20),
            (1, 24, 24),
            (1, 28, 28),
            (1, 32, 32),
        ]
    elif resolution == 1024:
        # 1024x1024: pn=1M
        scale_schedule = [
            (1, 1, 1),
            (1, 2, 2),
            (1, 3, 3),
            (1, 4, 4),
            (1, 5, 5),
            (1, 6, 6),
            (1, 8, 8),
            (1, 10, 10),
            (1, 13, 13),
            (1, 16, 16),
            (1, 20, 20),
            (1, 25, 25),
            (1, 32, 32),
            (1, 40, 40),
            (1, 50, 50),
            (1, 64, 64),
        ]
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")
    
    return scale_schedule

