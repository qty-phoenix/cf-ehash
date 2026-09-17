#!/usr/bin/env python3
"""单独下载 merges.txt 文件"""

import os
import requests
from tqdm import tqdm

def download_merges_txt(save_dir="pretrained_models/clip-vit-base-patch32"):
    """下载 merges.txt 文件"""
    
    url = "https://hf-mirror.com/openai/clip-vit-base-patch32/resolve/main/merges.txt"
    save_path = os.path.join(save_dir, "merges.txt")
    
    print("=" * 60)
    print("下载 merges.txt 文件")
    print("=" * 60)
    print(f"\n源地址: {url}")
    print(f"保存到: {os.path.abspath(save_path)}\n")
    
    # 创建目录
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # 下载文件
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(save_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc='merges.txt') as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        # 验证文件
        file_size = os.path.getsize(save_path)
        print(f"\n✓ 下载成功！")
        print(f"  文件大小: {file_size / 1024:.2f} KB")
        
        # 检查文件内容
        with open(save_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"  行数: {len(lines)}")
            if len(lines) > 0:
                print(f"  首行示例: {lines[0].strip()[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        print("\n备用下载方法:")
        print("1. 手动访问链接下载:")
        print(f"   {url}")
        print(f"2. 保存到: {os.path.abspath(save_path)}")
        return False


if __name__ == "__main__":
    download_merges_txt()

