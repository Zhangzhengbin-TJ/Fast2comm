# from opencood.tools import train_utils
# import opencood.hypes_yaml.yaml_utils as yaml_utils
# from opencood.tools.train import train_parser
# import torch
# import math
# import torch.nn.modules as nn
# position = torch.arange(0., 100).unsqueeze(1)
# div_term = torch.exp(torch.arange(0, 256, 2) *
#                      -(math.log(10000.0) / 256))
# emb = nn.Embedding(100, 256)
# emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(
#     256)
# emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(
#     256)
# print()
# print(emb.weight.data[:, 0::2].shape)
import matplotlib.pyplot as plt
import numpy as np

# 通信量 x 轴（连续）
x = np.array([12, 13, 14, 15, 16, 17, 18, 19, 20])

# 三个数据集的连续 y 值
how2comm_y = [
    [34.2, 35.5, 36.0, 37.8, 40.1, 42.0, 43.5, 44.8, 46.2],
    [45.0, 47.0, 49.5, 51.2, 54.1, 56.3, 58.9, 60.7, 62.8],
    [60.5, 62.1, 63.0, 64.5, 66.0, 67.3, 68.9, 70.0, 71.5]
]

# Where2comm曲线（假数据）
where2comm_y = [
    [33.0, 33.5, 34.2, 35.0, 36.2, 37.0, 38.3, 39.1, 40.0],
    [39.8, 40.0, 41.0, 42.5, 44.0, 45.6, 46.5, 47.2, 48.0],
    [55.5, 56.0, 57.0, 58.2, 59.0, 60.1, 61.0, 62.0, 63.0]
]

# 离散模型对应点（在x=20处）
discrete_methods = {
    "CoBEVT": {"marker": "o", "color": "blue"},
    "DiscoNet": {"marker": "v", "color": "orange"},
    "V2X-ViT": {"marker": "^", "color": "green"},
    "V2VNet": {"marker": "s", "color": "red"},
    "When2comm": {"marker": "*", "color": "purple"},
    "Late Fusion": {"marker": "+", "color": "gray"},
}

discrete_y = [
    [42.1, 42.5, 43.0, 41.5, 34.5, 33.8],
    [58.0, 58.3, 59.5, 57.2, 36.5, 35.8],
    [69.5, 70.1, 70.8, 68.2, 55.5, 54.8]
]

# 图标题和 y 轴标签
titles = ["AP@0.7 on DAIR-V2X", "AP@0.7 on V2XSet", "AP@0.7 on OPV2V"]

# 开始画图
fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

for i in range(3):
    ax = axs[i]
    # 连续曲线1：How2comm
    ax.plot(x, how2comm_y[i], '-o', color='orange', label='How2comm (Ours)')
    # 连续曲线2：Where2comm
    ax.plot(x, where2comm_y[i], '--', color='skyblue', label='Where2comm')

    # 离散点
    for j, (method, style) in enumerate(discrete_methods.items()):
        ax.scatter(20, discrete_y[i][j], label=method, marker=style["marker"], color=style["color"], s=70)

    ax.set_title(titles[i])
    ax.set_xlabel("Communication Volume (log2)")
    ax.set_ylabel("AP@0.7" if i == 0 else "")
    ax.grid(True)

# 设置统一图例
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, fontsize='small')

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.show()
