#!/bin/bash
# ===================================================================
# EraseInfinity Training Script
# 基于 Infinity 自回归模型的 nude 内容擦除训练脚本
# ===================================================================

set -e  # 遇到错误立即退出

# ==================== 配置路径 ====================
# 配置文件路径
CONFIG_FILE="config/erase_nude.yaml"

# 工作目录（EraseInfinity 根目录）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "EraseInfinity Training Script"
echo "Working directory: $SCRIPT_DIR"
echo "Config file: $CONFIG_FILE"
echo "=========================================="

# ==================== 检查配置文件 ====================
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "✓ Config file found"

# ==================== 检查数据目录 ====================
# 从 config 中读取数据目录（这里简化处理，实际可以用 yq 或 python 读取）
DATA_DIR=$(grep "instance_data_dir:" "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')

if [ -z "$DATA_DIR" ]; then
    echo "Warning: Could not read instance_data_dir from config"
else
    echo "Data directory: $DATA_DIR"
    if [ ! -d "$DATA_DIR" ]; then
        echo "Warning: Data directory does not exist: $DATA_DIR"
        echo "Please update the path in $CONFIG_FILE"
    else
        echo "✓ Data directory found"
    fi
fi

# ==================== 检查模型权重 ====================
VAE_CKPT=$(grep "vae_ckpt:" "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')
GPT_CKPT=$(grep "gpt_ckpt:" "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')

echo "Checking model checkpoints..."
echo "  VAE checkpoint: $VAE_CKPT"
echo "  GPT checkpoint: $GPT_CKPT"

if [ ! -f "$VAE_CKPT" ]; then
    echo "Warning: VAE checkpoint not found: $VAE_CKPT"
    echo "Please download it from https://huggingface.co/FoundationVision/infinity"
fi

if [ ! -f "$GPT_CKPT" ]; then
    echo "Warning: GPT checkpoint not found: $GPT_CKPT"
    echo "Please download it from https://huggingface.co/FoundationVision/infinity"
fi

# ==================== 环境检查 ====================
echo "=========================================="
echo "Environment Check"
echo "=========================================="

# 检查 Python
if ! command -v python &> /dev/null; then
    echo "Error: Python not found"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# 检查 CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✓ CUDA available"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "Warning: nvidia-smi not found, GPU may not be available"
fi

# 检查必要的 Python 包
echo "Checking Python packages..."
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')" || echo "Error: PyTorch not installed"
python -c "import transformers; print(f'✓ Transformers: {transformers.__version__}')" || echo "Error: Transformers not installed"
python -c "import peft; print(f'✓ PEFT: {peft.__version__}')" || echo "Warning: PEFT not installed (needed for LoRA)"
python -c "import wandb; print(f'✓ Wandb: {wandb.__version__}')" || echo "Warning: Wandb not installed (logging disabled)"

# ==================== 创建输出目录 ====================
OUTPUT_DIR=$(grep "output_dir:" "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')
if [ -n "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "✓ Output directory: $OUTPUT_DIR"
fi

# ==================== 设置环境变量 ====================
# Hugging Face 镜像（如果在国内）
# export HF_ENDPOINT="https://hf-mirror.com"

# 防止 tokenizers 并行警告
export TOKENIZERS_PARALLELISM=false

# CUDA 设备（从 config 读取或使用默认值）
DEVICES=$(grep "devices:" "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')
if [ -z "$DEVICES" ]; then
    DEVICES="0"
fi
export CUDA_VISIBLE_DEVICES="$DEVICES"

echo "=========================================="
echo "Training Configuration"
echo "=========================================="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Config: $CONFIG_FILE"
echo "=========================================="

# ==================== 开始训练 ====================
echo ""
echo "Starting training..."
echo "Press Ctrl+C to stop"
echo ""

# 使用 python 运行训练脚本
python train_erase.py \
    --config "$CONFIG_FILE" \
    2>&1 | tee "${OUTPUT_DIR}/training.log"

# ==================== 训练完成 ====================
echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Log file: ${OUTPUT_DIR}/training.log"
echo ""
echo "To resume training or load the model, use:"
echo "  - LoRA weights: ${OUTPUT_DIR}/lora_final/"
echo "  - Full checkpoint: ${OUTPUT_DIR}/final_model.pth"
echo ""
echo "=========================================="

