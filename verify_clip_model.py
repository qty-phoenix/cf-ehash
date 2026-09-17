#!/usr/bin/env python3
"""验证CLIP模型文件是否完整"""

import os
import json

def verify_clip_model(model_path):
    """验证CLIP模型文件"""
    print(f"检查模型路径: {os.path.abspath(model_path)}\n")
    
    required_files = [
        "config.json",
        "merges.txt",
        "pytorch_model.bin",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json"
    ]
    
    # 预期文件大小（大致范围）
    expected_sizes = {
        "config.json": (1000, 10000),  # 1KB-10KB
        "merges.txt": (400000, 500000),  # 400KB-500KB
        "pytorch_model.bin": (500000000, 700000000),  # 500MB-700MB
        "special_tokens_map.json": (100, 10000),
        "tokenizer_config.json": (100, 10000),
        "tokenizer.json": (1000000, 3000000),  # 1MB-3MB
        "vocab.json": (800000, 1000000)  # 800KB-1MB
    }
    
    all_good = True
    
    for filename in required_files:
        filepath = os.path.join(model_path, filename)
        
        if not os.path.exists(filepath):
            print(f"✗ 缺失: {filename}")
            all_good = False
        else:
            size = os.path.getsize(filepath)
            size_mb = size / 1024 / 1024
            
            # 检查文件大小是否合理
            if filename in expected_sizes:
                min_size, max_size = expected_sizes[filename]
                if min_size <= size <= max_size:
                    status = "✓"
                else:
                    status = "⚠"
                    all_good = False
            else:
                status = "✓"
            
            if size_mb > 1:
                print(f"{status} {filename:<30} {size_mb:>8.2f} MB")
            else:
                print(f"{status} {filename:<30} {size/1024:>8.2f} KB")
    
    print("\n" + "=" * 60)
    
    if all_good:
        print("✓ 所有文件验证通过！")
        
        # 尝试加载模型
        print("\n尝试加载模型...")
        try:
            from transformers import CLIPTextModel, CLIPTokenizer
            
            tokenizer = CLIPTokenizer.from_pretrained(model_path)
            model = CLIPTextModel.from_pretrained(model_path)
            
            print("✓ 模型加载成功！")
            print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")
            print(f"  词汇表大小: {len(tokenizer)}")
            
            # 测试推理
            test_text = ["a photo of a cat"]
            inputs = tokenizer(test_text, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            print(f"✓ 模型推理测试通过！")
            print(f"  输出维度: {outputs.pooler_output.shape}")
            
            print("\n可以开始训练了！🎉")
            return True
            
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            print("\n可能的问题:")
            print("  1. 文件损坏，请重新下载")
            print("  2. transformers版本不兼容，运行: pip install --upgrade transformers")
            return False
    else:
        print("✗ 文件验证失败！")
        print("\n请检查:")
        print("  1. 所有文件是否都已下载")
        print("  2. 文件是否下载完整（特别是pytorch_model.bin）")
        print("  3. 文件是否放在正确的目录")
        return False


if __name__ == "__main__":
    model_path = "pretrained_models/clip-vit-base-patch32"
    verify_clip_model(model_path)

