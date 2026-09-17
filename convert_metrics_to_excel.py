#!/usr/bin/env python3
"""
将 test_metrics.json 转换为 Excel 文件
"""

import json
import pandas as pd
import sys
import os

def convert_json_to_excel(json_file, excel_file=None):
    """
    将 test_metrics.json 转换为 Excel 文件
    
    Args:
        json_file: JSON 文件路径
        excel_file: 输出 Excel 文件路径（如果为 None，则自动生成）
    """
    # 检查 JSON 文件是否存在
    if not os.path.exists(json_file):
        print(f"错误: 文件不存在: {json_file}")
        return
    
    # 读取 JSON 文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not data:
        print(f"警告: JSON 文件为空: {json_file}")
        return
    
    # 转换为 DataFrame
    df = pd.DataFrame(data)
    
    # 如果未指定输出文件，自动生成
    if excel_file is None:
        base_name = os.path.splitext(json_file)[0]
        excel_file = f"{base_name}.xlsx"
    
    # 保存为 Excel
    df.to_excel(excel_file, index=False, engine='openpyxl')
    print(f"✅ 成功转换: {json_file} -> {excel_file}")
    print(f"   共 {len(df)} 条记录")
    print(f"   列: {', '.join(df.columns.tolist())}")
    
    return excel_file


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python convert_metrics_to_excel.py <json_file> [excel_file]")
        print("示例: python convert_metrics_to_excel.py log/ablation_balance_loss/test_metrics.json")
        print("      python convert_metrics_to_excel.py log/ablation_balance_loss/test_metrics.json output.xlsx")
        sys.exit(1)
    
    json_file = sys.argv[1]
    excel_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_json_to_excel(json_file, excel_file)
