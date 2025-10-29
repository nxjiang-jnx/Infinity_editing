#!/usr/bin/env python
# coding: UTF-8
"""
    @date:  2025.01
    @func:  Training script for EraseInfinity
            基于 Infinity 自回归模型的 nude 内容擦除训练脚本
"""

import os
import sys
import yaml
import argparse
import random
import time
import math
from pathlib import Path
from typing import List, Tuple, Optional
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5EncoderModel, T5TokenizerFast
from tqdm.auto import tqdm

# 添加 Infinity 到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入 Infinity 相关模块
import infinity.utils.dist as dist
from infinity.models import Infinity
from infinity.utils.load import build_vae_gpt
from infinity.utils import arg_util, misc

# 导入 EraseInfinity 模块
from dataset import EraseInfinityDataset, collate_fn
from utils.calc_loss import calculate_esd_loss_simple
from utils.esd_utils import get_default_scale_schedule

# Wandb
try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False
    print("Wandb not available, logging disabled")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Train EraseInfinity model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="Local rank for distributed training"
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_models(config: dict, device: torch.device):
    """
    构建模型：VAE + GPT + Text Encoder
    """
    print("=" * 80)
    print("Building models...")
    
    # ==================== 加载 VAE ====================
    vae_ckpt_path = config['vae_ckpt']
    print(f"Loading VAE from {vae_ckpt_path}")
    
    if os.path.exists(vae_ckpt_path):
        vae_ckpt = torch.load(vae_ckpt_path, map_location='cpu')
    else:
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_ckpt_path}")
    
    # ==================== 加载 GPT ====================
    gpt_ckpt_path = config['gpt_ckpt']
    print(f"Loading GPT from {gpt_ckpt_path}")
    
    # 创建临时 args 用于 build_vae_gpt
    class TempArgs:
        def __init__(self, config):
            self.vae_type = 'infinity'
            self.model_init_device = 'cpu'
            self.pn = config.get('pn', '0.06M')
            self.model = config.get('model_name', '2bc8')
            self.tlen = 256
            self.cond_drop_rate = 0.1
            self.device = device
            self.use_bit_label = 1
            self.apply_spatial_patchify = 0
            self.always_training_scales = 20
            # 添加其他必要的参数
            self.diva = 1
            self.alng = 1e-5
            self.aln = True
            self.hd0 = 1
            self.online_t5 = True
            self.t5_path = config.get('t5_path', 'google/flan-t5-xl')
    
    temp_args = TempArgs(config)
    
    # 使用 Infinity 的 build_vae_gpt 函数
    vae_local, gpt_wo_ddp, _ = build_vae_gpt(temp_args, vae_ckpt, skip_gpt=False, device='cpu')
    
    # 加载 GPT checkpoint
    if os.path.exists(gpt_ckpt_path):
        print(f"Loading GPT checkpoint...")
        gpt_state = torch.load(gpt_ckpt_path, map_location='cpu')
        
        # 处理不同的 checkpoint 格式
        if 'trainer' in gpt_state:
            # 训练中的 checkpoint
            if 'gpt_wo_ddp' in gpt_state['trainer']:
                gpt_state_dict = gpt_state['trainer']['gpt_wo_ddp']
            elif 'gpt_fsdp' in gpt_state['trainer']:
                gpt_state_dict = gpt_state['trainer']['gpt_fsdp']
            else:
                gpt_state_dict = gpt_state
        else:
            gpt_state_dict = gpt_state
        
        # 加载 state dict
        missing, unexpected = gpt_wo_ddp.load_state_dict(gpt_state_dict, strict=False)
        print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    else:
        print(f"Warning: GPT checkpoint not found at {gpt_ckpt_path}, using random initialization")
    
    # 移动到设备
    vae_local = vae_local.to(device)
    gpt_wo_ddp = gpt_wo_ddp.to(device)
    
    # 冻结 VAE
    vae_local.requires_grad_(False)
    vae_local.eval()
    
    # ==================== 添加 LoRA 到 GPT ====================
    if config.get('use_lora', True):
        print("Adding LoRA adapters to GPT...")
        from peft import LoraConfig, get_peft_model
        
        # 确定目标模块
        # Infinity 模型的注意力层在 blocks 中
        target_modules = []
        
        # 遍历模型找到所有 attention 相关的层
        for name, module in gpt_wo_ddp.named_modules():
            # Self-attention 的 QKV 投影
            if 'attn_blocks' in name and 'qkv' in name:
                target_modules.append(name.split('.')[-1])
            # Cross-attention 的投影
            elif 'cross_attn_blocks' in name and 'mat_qkv' in name:
                target_modules.append(name.split('.')[-1])
            # FFN 层
            elif 'ffn' in name and ('fc1' in name or 'fc2' in name):
                target_modules.append(name.split('.')[-1])
        
        # 去重
        target_modules = list(set(target_modules))
        print(f"LoRA target modules: {target_modules}")
        
        # 创建 LoRA config
        lora_config = LoraConfig(
            r=config.get('lora_rank', 8),
            lora_alpha=config.get('lora_alpha', 8),
            lora_dropout=config.get('lora_dropout', 0.0),
            target_modules=target_modules if target_modules else ["qkv", "fc1", "fc2"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # 应用 LoRA
        try:
            gpt_wo_ddp = get_peft_model(gpt_wo_ddp, lora_config)
            print(f"LoRA adapters added successfully")
            gpt_wo_ddp.print_trainable_parameters()
        except Exception as e:
            print(f"Failed to add LoRA: {e}")
            print("Falling back to full model fine-tuning")
            # 如果 LoRA 失败，冻结大部分参数，只训练部分层
            gpt_wo_ddp.requires_grad_(False)
            # 只训练最后几层
            for name, param in gpt_wo_ddp.named_parameters():
                if 'blocks' in name:
                    # 解析 block 索引
                    try:
                        block_idx = int(name.split('.')[1])
                        # 只训练后 25% 的层
                        if block_idx >= int(len(gpt_wo_ddp.blocks) * 0.75):
                            param.requires_grad = True
                    except:
                        pass
    else:
        # 不使用 LoRA，训练整个模型
        gpt_wo_ddp.requires_grad_(True)
    
    # 打印可训练参数数量
    trainable_params = sum(p.numel() for p in gpt_wo_ddp.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in gpt_wo_ddp.parameters())
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M / {total_params / 1e6:.2f}M ({trainable_params / total_params * 100:.2f}%)")
    
    # ==================== 加载 Text Encoder ====================
    print(f"Loading T5 text encoder from {config['t5_path']}")
    text_tokenizer = T5TokenizerFast.from_pretrained(config['t5_path'], legacy=True)
    text_tokenizer.model_max_length = 256
    
    text_encoder = T5EncoderModel.from_pretrained(config['t5_path'], torch_dtype=torch.float16)
    text_encoder = text_encoder.to(device)
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    
    print("Models built successfully!")
    print("=" * 80)
    
    return vae_local, gpt_wo_ddp, text_tokenizer, text_encoder


def build_optimizer(config: dict, model: nn.Module):
    """构建优化器"""
    optimizer_name = config.get('optimizer', 'adamw').lower()
    learning_rate = config.get('learning_rate', 1e-4)
    
    # 获取可训练参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            betas=(config.get('adam_beta1', 0.9), config.get('adam_beta2', 0.999)),
            eps=config.get('adam_epsilon', 1e-8),
            weight_decay=config.get('adam_weight_decay', 1e-4),
        )
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=learning_rate,
            betas=(config.get('adam_beta1', 0.9), config.get('adam_beta2', 0.999)),
            eps=config.get('adam_epsilon', 1e-8),
        )
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=config.get('adam_weight_decay', 1e-4),
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def build_dataloader(config: dict):
    """构建数据加载器"""
    print("=" * 80)
    print("Building dataloader...")
    
    dataset = EraseInfinityDataset(
        instance_data_root=config['instance_data_dir'],
        instance_prompt=config['instance_prompt'],
        key_word=config['key_word'],
        size=config['resolution'],
        repeats=config.get('repeats', 1),
        center_crop=config.get('center_crop', False),
        random_flip=config.get('random_flip', False),
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['train_batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.get('dataloader_num_workers', 4),
        pin_memory=True,
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataloader batches: {len(dataloader)}")
    print("=" * 80)
    
    return dataloader


def train_one_epoch(
    epoch: int,
    model: nn.Module,
    vae: nn.Module,
    text_encoder: nn.Module,
    text_tokenizer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: dict,
    device: torch.device,
):
    """训练一个 epoch"""
    
    model.train()
    vae.eval()
    text_encoder.eval()
    
    # 准备参数
    weight_dtype = torch.bfloat16 if config.get('mixed_precision') == 'bf16' else torch.float32
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    max_grad_norm = config.get('max_grad_norm', 1.0)
    
    # 获取 scale schedule
    scale_schedule = get_default_scale_schedule(config['resolution'])
    
    # 添加到 config（用于 loss 计算）
    class ConfigWithSchedule:
        def __init__(self, config_dict, scale_schedule):
            for k, v in config_dict.items():
                setattr(self, k, v)
            self.scale_schedule = scale_schedule
            self.always_training_scales = len(scale_schedule)
    
    config_obj = ConfigWithSchedule(config, scale_schedule)
    
    # Loss function
    criteria = nn.MSELoss()
    
    # Progress bar
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    total_loss = 0.0
    num_steps = 0
    
    for step, batch in enumerate(progress_bar):
        # ==================== 编码文本 ====================
        prompts = batch['prompts']
        
        with torch.no_grad():
            # Tokenize
            tokens = text_tokenizer(
                text=prompts,
                max_length=text_tokenizer.model_max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = tokens.input_ids.to(device, non_blocking=True)
            mask = tokens.attention_mask.to(device, non_blocking=True)
            
            # Encode
            text_features = text_encoder(input_ids=input_ids, attention_mask=mask)['last_hidden_state'].float()
            
            # 准备 text_cond_tuple
            lens = mask.sum(dim=-1).tolist()
            cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
            Ltext = max(lens)
            
            text_cond_tuple = (text_features, lens, cu_seqlens_k, Ltext)
        
        # ==================== 计算 ESD Loss ====================
        loss = calculate_esd_loss_simple(
            args=config_obj,
            batch=batch,
            gpt_model=model,
            vae_model=vae,
            text_cond_tuple=text_cond_tuple,
            criteria=criteria,
            negative_guidance=config.get('negative_guidance', 1.0),
            device=device,
            weight_dtype=weight_dtype,
        )
        
        # 缩放 loss
        loss = loss / gradient_accumulation_steps
        
        # Backward
        loss.backward()
        
        # 梯度累积
        if (step + 1) % gradient_accumulation_steps == 0:
            # 梯度裁剪
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # 优化器步骤
            optimizer.step()
            optimizer.zero_grad()
            
            # 更新统计
            total_loss += loss.item() * gradient_accumulation_steps
            num_steps += 1
            
            # 更新进度条
            avg_loss = total_loss / num_steps
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # Wandb logging
            if has_wandb and config.get('report_to') == 'wandb':
                wandb.log({
                    'train/loss': loss.item() * gradient_accumulation_steps,
                    'train/avg_loss': avg_loss,
                    'train/lr': optimizer.param_groups[0]['lr'],
                    'train/epoch': epoch,
                    'train/step': step,
                })
    
    avg_loss = total_loss / num_steps if num_steps > 0 else 0.0
    return avg_loss


def main():
    """主函数"""
    
    # ==================== 解析参数 ====================
    args = parse_args()
    config = load_config(args.config)
    
    # ==================== 设置设备 ====================
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ==================== 设置随机种子 ====================
    setup_seed(config.get('seed', 42))
    
    # ==================== 创建输出目录 ====================
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # ==================== 初始化 Wandb ====================
    if has_wandb and config.get('report_to') == 'wandb':
        wandb.init(
            project=config.get('project_name', 'EraseInfinity'),
            name=config.get('exp_name', 'erase_nude'),
            config=config,
        )
    
    # ==================== 构建模型 ====================
    vae, model, text_tokenizer, text_encoder = build_models(config, device)
    
    # ==================== 构建优化器 ====================
    optimizer = build_optimizer(config, model)
    
    # ==================== 构建数据加载器 ====================
    dataloader = build_dataloader(config)
    
    # ==================== 训练循环 ====================
    num_epochs = config.get('num_train_epochs', 5)
    max_train_steps = config.get('max_train_steps', 1000)
    save_steps = config.get('save_steps', 200)
    
    print("=" * 80)
    print("Starting training...")
    print(f"Total epochs: {num_epochs}")
    print(f"Max steps: {max_train_steps}")
    print("=" * 80)
    
    global_step = 0
    
    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(
            epoch=epoch,
            model=model,
            vae=vae,
            text_encoder=text_encoder,
            text_tokenizer=text_tokenizer,
            dataloader=dataloader,
            optimizer=optimizer,
            config=config,
            device=device,
        )
        
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
        
        # 保存 checkpoint
        if (epoch + 1) % (num_epochs // 5) == 0 or (epoch + 1) == num_epochs:
            save_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            
            # 保存模型
            if config.get('use_lora', True):
                # 保存 LoRA 权重
                model.save_pretrained(os.path.join(output_dir, f"lora_epoch_{epoch+1}"))
            else:
                # 保存整个模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'config': config,
                }, save_path)
            
            print(f"Checkpoint saved to {save_path}")
        
        global_step += len(dataloader)
        if global_step >= max_train_steps:
            print(f"Reached max_train_steps ({max_train_steps}), stopping training.")
            break
    
    # ==================== 保存最终模型 ====================
    final_save_path = os.path.join(output_dir, "final_model.pth")
    
    if config.get('use_lora', True):
        model.save_pretrained(os.path.join(output_dir, "lora_final"))
    else:
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
        }, final_save_path)
    
    print(f"Final model saved to {final_save_path}")
    print("=" * 80)
    print("Training completed!")
    print("=" * 80)
    
    # ==================== 关闭 Wandb ====================
    if has_wandb and config.get('report_to') == 'wandb':
        wandb.finish()


if __name__ == "__main__":
    main()

