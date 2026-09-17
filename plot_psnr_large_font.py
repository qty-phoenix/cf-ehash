import pandas as pd
import matplotlib.pyplot as plt

# 设置全局字体样式（不加粗，字体调大）
plt.rcParams['font.family'] = 'Arial'  # 可选: 'Times New Roman', 'DejaVu Sans', 'SimHei'（中文）
plt.rcParams['font.size'] = 14  # 全局字体大小调大
plt.rcParams['axes.linewidth'] = 1.5  # 坐标轴线宽

# 读取Excel文件
df_aba = pd.read_excel('aba.xlsx')
df_ours = pd.read_excel('ours.xlsx')

# 每5个点画一次
df_aba_subset = df_aba.iloc[::5]
df_ours_subset = df_ours.iloc[::5]

# 绘制对比图
plt.figure(figsize=(10, 6))
plt.plot(df_aba_subset['epoch'], df_aba_subset['psnr'], marker='o', label='w/o Loss-Free Balancing', 
         linewidth=2.5, color='blue', markersize=6)
plt.plot(df_ours_subset['epoch'], df_ours_subset['psnr'], marker='s', label='Full Model', 
         linewidth=2.5, color='orange', markersize=6)

# 设置坐标轴标签（字体调大，不加粗）
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('PSNR (dB)', fontsize=16)

# 设置标题（字体调大，不加粗）
plt.title('PSNR Comparison: Ablation vs Ours', fontsize=18, pad=20)

# 设置图例（字体调大，放在右下角，不加粗）
plt.legend(fontsize=14, frameon=True, fancybox=True, shadow=True, 
           loc='lower right', framealpha=0.9)

# 设置网格
plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

# 设置刻度标签（字体调大）
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=13, width=1.5)

plt.tight_layout()

plt.savefig('psnr_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
