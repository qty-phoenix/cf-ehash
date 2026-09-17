import pandas as pd
import matplotlib.pyplot as plt

# 设置全局字体（可选：使用更清晰的字体）
plt.rcParams['font.family'] = 'Arial'  # 或 'Times New Roman', 'DejaVu Sans', 'SimHei'（中文）
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'  # 'normal', 'bold', 'light'

# 读取Excel文件
df_aba = pd.read_excel('aba.xlsx')
df_ours = pd.read_excel('ours.xlsx')

# 每5个点画一次
df_aba_subset = df_aba.iloc[::5]
df_ours_subset = df_ours.iloc[::5]

# 绘制对比图
plt.figure(figsize=(10, 6))
plt.plot(df_aba_subset['epoch'], df_aba_subset['psnr'], marker='o', label='w/o Loss-Free Balancing', 
         linewidth=2, color='blue', markersize=6)
plt.plot(df_ours_subset['epoch'], df_ours_subset['psnr'], marker='s', label='Full Model', 
         linewidth=2, color='orange', markersize=6)

# 设置坐标轴标签（加粗）
plt.xlabel('Epoch', fontsize=14, fontweight='bold')
plt.ylabel('PSNR (dB)', fontsize=14, fontweight='bold')

# 设置标题（加粗，更大字体）
plt.title('PSNR Comparison: Ablation vs Ours', fontsize=16, fontweight='bold', pad=20)

# 设置图例（加粗）
from matplotlib import font_manager
legend_font = font_manager.FontProperties(weight='bold', size=12)
plt.legend(prop=legend_font, frameon=True, fancybox=True, shadow=True)

# 设置网格
plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

# 设置刻度标签字体（可选加粗）
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=11, width=1.5)
# 如果需要刻度标签也加粗，取消下面的注释
# for label in ax.get_xticklabels() + ax.get_yticklabels():
#     label.set_fontweight('bold')

plt.tight_layout()

plt.savefig('psnr_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
