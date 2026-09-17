#!/usr/bin/env python3
"""
可视化专家增长过程的工具
用于分析训练日志，展示专家数量变化、损失曲线、专家使用情况等
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def parse_training_log(log_file):
    """
    解析训练日志文件
    
    Returns:
        epochs: epoch列表
        losses: 损失值列表
        n_experts: 每个epoch的专家数量
        expert_usage: 专家使用情况
        growth_events: 增长事件列表
        psnr: PSNR值列表
        ssim: SSIM值列表
    """
    epochs = []
    losses = []
    n_experts_list = []
    expert_usage_history = []
    growth_events = []
    psnr_list = []
    ssim_list = []
    
    current_n_experts = 2  # 默认初始值
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 解析epoch信息（新格式包含PSNR和SSIM）
            # 格式: "Epoch  100/3000 | 专家数: 2 | Loss: 0.045623 | ... | PSNR: 28.50dB | SSIM: 0.8512"
            epoch_match = re.search(r'Epoch\s+(\d+)/\d+\s+\|\s+专家数:\s+(\d+)\s+\|\s+Loss:\s+([\d.]+)', line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                n_experts = int(epoch_match.group(2))
                loss = float(epoch_match.group(3))
                
                epochs.append(epoch)
                losses.append(loss)
                n_experts_list.append(n_experts)
                current_n_experts = n_experts
                
                # 尝试解析PSNR和SSIM
                psnr_match = re.search(r'PSNR:\s+([\d.]+)dB', line)
                if psnr_match:
                    psnr_list.append(float(psnr_match.group(1)))
                
                ssim_match = re.search(r'SSIM:\s+([\d.]+)', line)
                if ssim_match:
                    ssim_list.append(float(ssim_match.group(1)))
            
            # 解析专家使用情况
            # 格式: "  专家使用: E0: 0.234 | E1: 0.345 | E2: 0.421"
            usage_match = re.search(r'专家使用:\s+(.*)', line)
            if usage_match:
                usage_str = usage_match.group(1)
                # 提取每个专家的使用率
                usage_values = []
                for expert_usage in re.finditer(r'E\d+:\s+([\d.]+)', usage_str):
                    usage_values.append(float(expert_usage.group(1)))
                
                if usage_values and len(epochs) > 0:
                    expert_usage_history.append({
                        'epoch': epochs[-1],
                        'usage': usage_values
                    })
            
            # 解析增长事件
            # 格式: "   专家数量: 2 -> 3"
            growth_match = re.search(r'专家数量:\s+(\d+)\s+->\s+(\d+)', line)
            if growth_match:
                # 向前查找epoch信息
                epoch_line_match = re.search(r'Epoch:\s+(\d+)', line)
                if epoch_line_match:
                    growth_epoch = int(epoch_line_match.group(1))
                elif len(epochs) > 0:
                    growth_epoch = epochs[-1]
                else:
                    growth_epoch = 0
                
                from_experts = int(growth_match.group(1))
                to_experts = int(growth_match.group(2))
                
                growth_events.append({
                    'epoch': growth_epoch,
                    'from': from_experts,
                    'to': to_experts
                })
    
    return {
        'epochs': epochs,
        'losses': losses,
        'n_experts': n_experts_list,
        'expert_usage': expert_usage_history,
        'growth_events': growth_events,
        'psnr': psnr_list,
        'ssim': ssim_list
    }


def plot_training_analysis(data, output_dir='./'):
    """
    创建综合分析图表
    
    包含：
    1. 损失曲线 + 专家数量变化
    2. PSNR和SSIM曲线
    3. 专家使用情况热力图
    4. 增长事件标记
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建大图，包含多个子图
    fig = plt.figure(figsize=(16, 14))
    
    # ========== 子图1: 损失曲线 + 专家数量 ==========
    ax1 = plt.subplot(4, 1, 1)
    
    epochs = np.array(data['epochs'])
    losses = np.array(data['losses'])
    n_experts = np.array(data['n_experts'])
    
    # 绘制损失曲线
    color1 = 'tab:blue'
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', color=color1, fontsize=12)
    ax1.plot(epochs, losses, color=color1, linewidth=2, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # 对数尺度更清晰
    
    # 在同一图上绘制专家数量
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Number of Experts', color=color2, fontsize=12)
    ax2.plot(epochs, n_experts, color=color2, linewidth=2, 
             linestyle='--', marker='o', markersize=3, label='Number of Experts')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(bottom=0)
    
    # 标记增长事件
    for event in data['growth_events']:
        ax1.axvline(x=event['epoch'], color='green', linestyle=':', 
                   linewidth=2, alpha=0.7)
        ax1.text(event['epoch'], ax1.get_ylim()[1] * 0.5, 
                f"Growth\n{event['from']}→{event['to']}", 
                rotation=0, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
                fontsize=9)
    
    ax1.set_title('Training Loss and Expert Count Over Time', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # ========== 子图2: PSNR和SSIM曲线 ==========
    if data['psnr'] and data['ssim']:
        ax_psnr = plt.subplot(4, 1, 2)
        
        psnr_array = np.array(data['psnr'])
        ssim_array = np.array(data['ssim'])
        metric_epochs = epochs[:len(psnr_array)]  # PSNR/SSIM可能没有每个epoch都计算
        
        # 绘制PSNR曲线
        color_psnr = 'tab:green'
        ax_psnr.set_xlabel('Epoch', fontsize=12)
        ax_psnr.set_ylabel('PSNR (dB)', color=color_psnr, fontsize=12)
        ax_psnr.plot(metric_epochs, psnr_array, color=color_psnr, linewidth=2, label='PSNR')
        ax_psnr.tick_params(axis='y', labelcolor=color_psnr)
        ax_psnr.grid(True, alpha=0.3)
        
        # 在同一图上绘制SSIM
        ax_ssim = ax_psnr.twinx()
        color_ssim = 'tab:purple'
        ax_ssim.set_ylabel('SSIM', color=color_ssim, fontsize=12)
        ax_ssim.plot(metric_epochs, ssim_array, color=color_ssim, linewidth=2, 
                    linestyle='--', label='SSIM')
        ax_ssim.tick_params(axis='y', labelcolor=color_ssim)
        ax_ssim.set_ylim([0, 1])
        
        # 标记增长事件
        for event in data['growth_events']:
            if event['epoch'] <= metric_epochs[-1]:
                ax_psnr.axvline(x=event['epoch'], color='green', linestyle=':', 
                              linewidth=2, alpha=0.7)
        
        ax_psnr.set_title('Image Quality Metrics (PSNR & SSIM)', fontsize=14, fontweight='bold')
        ax_psnr.legend(loc='lower left')
        ax_ssim.legend(loc='lower right')
    
    # ========== 子图3: 专家使用率随时间变化 ==========
    ax3 = plt.subplot(4, 1, 3)
    
    if data['expert_usage']:
        # 准备数据
        usage_epochs = [u['epoch'] for u in data['expert_usage']]
        max_experts = max(len(u['usage']) for u in data['expert_usage'])
        
        # 创建矩阵 (n_epochs, n_experts)
        usage_matrix = np.zeros((len(usage_epochs), max_experts))
        for i, usage_data in enumerate(data['expert_usage']):
            usage = usage_data['usage']
            usage_matrix[i, :len(usage)] = usage
        
        # 绘制热力图
        im = ax3.imshow(usage_matrix.T, aspect='auto', cmap='YlOrRd', 
                       interpolation='nearest', origin='lower')
        
        ax3.set_xlabel('Training Step', fontsize=12)
        ax3.set_ylabel('Expert Index', fontsize=12)
        ax3.set_title('Expert Usage Heatmap', fontsize=14, fontweight='bold')
        
        # 设置刻度
        n_ticks = min(10, len(usage_epochs))
        tick_indices = np.linspace(0, len(usage_epochs)-1, n_ticks, dtype=int)
        ax3.set_xticks(tick_indices)
        ax3.set_xticklabels([usage_epochs[i] for i in tick_indices])
        
        ax3.set_yticks(range(max_experts))
        ax3.set_yticklabels([f'E{i}' for i in range(max_experts)])
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Usage Frequency', rotation=270, labelpad=20, fontsize=11)
        
        # 标记增长事件
        for event in data['growth_events']:
            # 找到最接近的epoch索引
            idx = np.argmin(np.abs(np.array(usage_epochs) - event['epoch']))
            ax3.axvline(x=idx, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    
    # ========== 子图4: 每个阶段的平均专家使用率 ==========
    ax4 = plt.subplot(4, 1, 4)
    
    if data['expert_usage'] and data['growth_events']:
        # 为每个增长阶段计算平均使用率
        growth_epochs_list = [0] + [e['epoch'] for e in data['growth_events']] + [epochs[-1]]
        
        for i in range(len(growth_epochs_list) - 1):
            start_epoch = growth_epochs_list[i]
            end_epoch = growth_epochs_list[i + 1]
            
            # 找到这个阶段的所有使用率数据
            stage_usage = []
            for usage_data in data['expert_usage']:
                if start_epoch <= usage_data['epoch'] < end_epoch:
                    stage_usage.append(usage_data['usage'])
            
            if stage_usage:
                # 计算平均使用率
                avg_usage = np.mean(stage_usage, axis=0)
                n_experts_stage = len(avg_usage)
                
                # 绘制柱状图
                x_offset = i * (max_experts + 1)
                bars = ax4.bar(np.arange(n_experts_stage) + x_offset, avg_usage, 
                              width=0.8, alpha=0.7, 
                              label=f'Epoch {start_epoch}-{end_epoch} ({n_experts_stage} experts)')
                
                # 在柱状图上标注数值
                for j, (bar, val) in enumerate(zip(bars, avg_usage)):
                    ax4.text(bar.get_x() + bar.get_width()/2, val + 0.01, 
                            f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax4.set_xlabel('Expert Index (grouped by stage)', fontsize=12)
        ax4.set_ylabel('Average Usage', fontsize=12)
        ax4.set_title('Average Expert Usage by Training Stage', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim(0, max(0.5, ax4.get_ylim()[1]))
    
    plt.tight_layout()
    
    # 保存图表
    output_file = output_dir / 'expert_growth_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 分析图表已保存: {output_file}")
    
    plt.show()


def print_summary_statistics(data):
    """打印摘要统计信息"""
    print("\n" + "="*80)
    print("训练摘要统计")
    print("="*80)
    
    epochs = data['epochs']
    losses = data['losses']
    n_experts_list = data['n_experts']
    growth_events = data['growth_events']
    
    print(f"\n总训练Epoch数: {len(epochs)}")
    print(f"初始专家数: {n_experts_list[0]}")
    print(f"最终专家数: {n_experts_list[-1]}")
    print(f"专家增长次数: {len(growth_events)}")
    
    print(f"\n初始损失: {losses[0]:.6f}")
    print(f"最终损失: {losses[-1]:.6f}")
    print(f"损失降低: {(1 - losses[-1]/losses[0])*100:.2f}%")
    
    print(f"\n专家增长历史:")
    for i, event in enumerate(growth_events, 1):
        print(f"  {i}. Epoch {event['epoch']:4d}: {event['from']} -> {event['to']} 专家")
    
    # 分析每个阶段的性能提升
    if growth_events:
        print(f"\n各阶段性能分析:")
        growth_epochs = [0] + [e['epoch'] for e in growth_events] + [epochs[-1]]
        
        for i in range(len(growth_epochs) - 1):
            start_idx = next((j for j, e in enumerate(epochs) if e >= growth_epochs[i]), 0)
            end_idx = next((j for j, e in enumerate(epochs) if e >= growth_epochs[i+1]), len(epochs)-1)
            
            if start_idx < end_idx:
                stage_losses = losses[start_idx:end_idx+1]
                initial_loss = stage_losses[0]
                final_loss = stage_losses[-1]
                improvement = (1 - final_loss/initial_loss) * 100 if initial_loss > 0 else 0
                
                n_experts_stage = n_experts_list[start_idx]
                print(f"  阶段 {i+1} (Epoch {growth_epochs[i]}-{growth_epochs[i+1]}, "
                      f"{n_experts_stage} 专家):")
                print(f"    初始损失: {initial_loss:.6f}")
                print(f"    最终损失: {final_loss:.6f}")
                print(f"    改善程度: {improvement:.2f}%")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='可视化专家增长训练过程'
    )
    parser.add_argument('log_file', type=str,
                       help='训练日志文件路径')
    parser.add_argument('--output-dir', type=str, default='./',
                       help='输出图表保存目录')
    
    args = parser.parse_args()
    
    print(f"正在解析训练日志: {args.log_file}")
    
    try:
        data = parse_training_log(args.log_file)
        
        if not data['epochs']:
            print("错误: 未能从日志文件中提取到训练数据")
            print("请确保日志文件包含正确格式的训练输出")
            return
        
        # 打印摘要统计
        print_summary_statistics(data)
        
        # 生成可视化图表
        print(f"\n正在生成可视化图表...")
        plot_training_analysis(data, args.output_dir)
        
        print("\n✓ 分析完成!")
        
    except FileNotFoundError:
        print(f"错误: 找不到日志文件 {args.log_file}")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()


