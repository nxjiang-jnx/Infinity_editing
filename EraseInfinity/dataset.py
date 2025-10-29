# coding: UTF-8
"""
    @date:  2025.01
    @func:  Dataset for EraseInfinity
            用于 nude 内容擦除的数据集加载器
"""

import os
import random
import itertools
from pathlib import Path
from PIL import Image
from PIL.ImageOps import exif_transpose
from typing import Optional, List

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop

# 导入 nltk 用于同义词查找
import nltk
import sys

# 设置 NLTK 缓存目录
nltk_data_dir = os.path.join(os.path.expanduser("~"), ".cache", "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)

# 检查 wordnet 是否已经存在
if not os.path.exists(os.path.join(nltk_data_dir, 'corpora', 'wordnet')):
    print("wordnet不存在，开始下载...")
    try:
        nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)
        print("wordnet下载完成")
    except Exception as e:
        print(f"wordnet下载失败: {e}")

# 设置 NLTK 数据路径
nltk.data.path.append(nltk_data_dir)

try:
    from nltk.corpus import wordnet
except Exception as e:
    print(f"无法加载 wordnet: {e}")
    wordnet = None


def get_synonyms(word: str) -> set:
    """
    获取单词的同义词
    """
    if wordnet is None:
        return {word}
    
    try:
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name().replace('_', ' '))
        return set(synonyms) if synonyms else {word}
    except Exception as e:
        print(f"获取同义词失败 ({word}): {e}")
        return {word}


class EraseInfinityDataset(Dataset):
    """
    EraseInfinity 数据集
    
    用于加载包含需要擦除内容的图像和对应的 prompts
    支持数据增强和 prompt 随机化
    """
    
    def __init__(
        self,
        instance_data_root: str,
        instance_prompt: str,
        key_word: str,
        size: int = 256,
        repeats: int = 1,
        center_crop: bool = False,
        random_flip: bool = False,
    ):
        """
        Args:
            instance_data_root: 图像数据根目录
            instance_prompt: 基础 prompt 模板
            key_word: 要擦除的关键词
            size: 图像大小
            repeats: 数据重复次数
            center_crop: 是否使用中心裁剪
            random_flip: 是否随机翻转
        """
        
        self.size = size
        self.center_crop = center_crop
        self.random_flip = random_flip
        
        # 检查关键词是否在 prompt 中
        if key_word in instance_prompt:
            self.key_word = key_word
        else:
            self.key_word = None
            print(f"Warning: key_word '{key_word}' not found in instance_prompt '{instance_prompt}'")
        
        self.instance_prompt = instance_prompt
        
        # 加载图像路径
        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance images root doesn't exist: {instance_data_root}")
        
        # 获取所有图像文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        instance_images = []
        for ext in image_extensions:
            instance_images.extend(list(self.instance_data_root.glob(f'*{ext}')))
            instance_images.extend(list(self.instance_data_root.glob(f'*{ext.upper()}')))
        
        if len(instance_images) == 0:
            raise ValueError(f"No images found in {instance_data_root}")
        
        print(f"Found {len(instance_images)} images in {instance_data_root}")
        
        # 加载图像并重复
        self.instance_images = []
        for img_path in instance_images:
            try:
                img = Image.open(img_path)
                img = exif_transpose(img)
                if not img.mode == "RGB":
                    img = img.convert("RGB")
                self.instance_images.extend(itertools.repeat(img, repeats))
            except Exception as e:
                print(f"Failed to load image {img_path}: {e}")
        
        self.num_instance_images = len(self.instance_images)
        print(f"Total samples after repeat: {self.num_instance_images}")
        
        # 预处理图像
        self.pixel_values = []
        train_resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        train_crop = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        train_flip = transforms.RandomHorizontalFlip(p=1.0)
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        for image in self.instance_images:
            image = train_resize(image)
            
            if random_flip and random.random() < 0.5:
                image = train_flip(image)
            
            if center_crop:
                y1 = max(0, int(round((image.height - size) / 2.0)))
                x1 = max(0, int(round((image.width - size) / 2.0)))
                image = train_crop(image)
            else:
                y1, x1, h, w = train_crop.get_params(image, (size, size))
                image = crop(image, y1, x1, h, w)
            
            image = train_transforms(image)
            self.pixel_values.append(image)
        
        print(f"Preprocessed {len(self.pixel_values)} images")
    
    def __len__(self):
        return self.num_instance_images
    
    def __getitem__(self, index):
        example = {}
        
        # 获取图像
        instance_image = self.pixel_values[index % self.num_instance_images]
        example["pixel_values"] = instance_image
        
        # 获取 prompt
        prompt = self.instance_prompt
        
        # Prompt 增强：随机打乱单词顺序（10% 概率保持原样）
        if random.random() >= 0.1:
            words = prompt.split(" ")
            random.shuffle(words)
            prompt = ' '.join(words)
        
        example["prompt"] = prompt
        
        # 获取同义词
        if self.key_word is not None:
            synonym_list = list(get_synonyms(self.key_word))
            if len(synonym_list) > 0:
                synonym = random.choice(synonym_list)
            else:
                synonym = self.key_word
            
            example["synonym"] = synonym
            
            # 50% 概率用同义词替换关键词
            if random.random() >= 0.5 and self.key_word in prompt:
                prompt = prompt.replace(self.key_word, synonym)
                example["prompt"] = prompt
        else:
            example["synonym"] = ""
        
        # 添加 key_word 用于标记
        example["key_word"] = self.key_word if self.key_word is not None else ""
        
        return example


def collate_fn(examples: List[dict]) -> dict:
    """
    Collate function for DataLoader
    """
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    
    prompts = [example["prompt"] for example in examples]
    synonyms = [example["synonym"] for example in examples]
    key_words = [example["key_word"] for example in examples]
    
    batch = {
        "pixel_values": pixel_values,
        "prompts": prompts,
        "synonyms": synonyms,
        "key_words": key_words,
    }
    
    return batch


if __name__ == "__main__":
    # 测试数据集
    print("Testing EraseInfinityDataset...")
    
    dataset = EraseInfinityDataset(
        instance_data_root="/path/to/nude/images",
        instance_prompt="nude person with exposed body",
        key_word="nude",
        size=256,
        repeats=2,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # 测试获取样本
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Pixel values shape: {sample['pixel_values'].shape}")
    print(f"Prompt: {sample['prompt']}")
    print(f"Synonym: {sample['synonym']}")
    print(f"Key word: {sample['key_word']}")
    
    # 测试 DataLoader
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    for batch in dataloader:
        print(f"Batch pixel_values shape: {batch['pixel_values'].shape}")
        print(f"Batch prompts: {batch['prompts']}")
        print(f"Batch synonyms: {batch['synonyms']}")
        break
    
    print("Dataset test completed!")

