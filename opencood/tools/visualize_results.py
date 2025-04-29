import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import open3d as o3d
import torch
import numpy as np
import math

def feature_heatmap(x):
    time=datetime.now()
    x=torch.mean(x,dim=0)
    max_value = torch.max(x)
    min_value = torch.min(x)
    normalized_tensor = (x - min_value) / (max_value - min_value)
    # 将归一化后的张量映射到颜色空间
    heatmap = torch.clamp(normalized_tensor, 0, 1)  # 确保值在 [0, 1] 区间内
    heatmap = heatmap.unsqueeze(2).cpu().detach().numpy() # 转换为 HWC 格式
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    # 可视化热图
    time = datetime.now().strftime('%Y%m%d_%H%M%S')  # 格式化时间戳
    plt.imsave(f'picture/heatmap_{time}.png', heatmap, format='png')  # 注意 BGR 转 RGB
    plt.axis('off')
    plt.show()

def visualize_batch_heatmaps(x,  grid_size=(2, 2)):
    time = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 获取 batch 大小和图像尺寸
    B, _ , H, W = x.shape
    rows, cols = grid_size

    # 检查网格大小是否足够显示所有 batch
    if rows * cols < B:
        raise ValueError("网格大小不足以显示所有 batch 样本，请增加 rows 和 cols")

    # 创建绘图窗口
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()  # 将轴对象展平，便于迭代

    # 绘制每个 batch 样本的热力图
    for i in range(B):
        sample = x[i]  # 取出第 i 个 batch 样本，形状为 (H, W)

        sample=torch.mean(sample,dim=0)
        # 计算归一化
        max_value = torch.max(sample)
        min_value = torch.min(sample)
        normalized_tensor = (sample - min_value) / (max_value - min_value)
        heatmap = torch.clamp(normalized_tensor, 0, 1).cpu().detach().numpy()

        # 应用颜色映射
        heatmap = np.uint8(255 * heatmap)  # 映射到 [0, 255]
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # 绘制到子图中
        ax = axes[i]
        ax.imshow(heatmap)
        ax.axis('off')
        ax.set_title(f"Batch {i+1}")

    # 如果 batch 数量少于网格数，隐藏多余的子图
    for j in range(B, rows * cols):
        fig.delaxes(axes[j])

    # 调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig(f'picture/batch_heatmaps_{time}.png', format='png', bbox_inches='tight')
    plt.show()





def bbox_2d(pre_box, gt_box,pcd,save_path):
    x = pcd[:, 0].cpu().numpy()
    y = pcd[:, 1].cpu().numpy()

    # 创建绘图
    plt.figure(figsize=(8, 4))
    plt.scatter(x, y, s=0.1, c='black',alpha=0.2)
    # pre_box=pre_box.cpu().numpy()
    # gt_box=gt_box.cpu().numpy()
    pre_box = pre_box.cpu().detach().numpy()
    gt_box = gt_box.cpu().detach().numpy()
    # 绘制真值检测框
    for j in range(gt_box.shape[0]):
        true_box = gt_box[j]
        for i in range(4):
            plt.plot(
                [true_box[i, 0], true_box[(i + 1) % 4, 0]],
                [true_box[i, 1], true_box[(i + 1) % 4, 1]],
                'g-', linewidth=0.5
            )
            plt.plot(
                [true_box[i + 4, 0], true_box[(i + 1) % 4 + 4, 0]],
                [true_box[i + 4, 1], true_box[(i + 1) % 4 + 4, 1]],
                'g-', linewidth=0.5
            )
            plt.plot(
                [true_box[i, 0], true_box[i + 4, 0]],
                [true_box[i, 1], true_box[i + 4, 1]],
                'g-', linewidth=0.5
            )

    # 绘制预测检测框
    for j in range(pre_box.shape[0]):
        pred_box = pre_box[j]
        for i in range(4):
            plt.plot(
                [pred_box[i, 0], pred_box[(i + 1) % 4, 0]],
                [pred_box[i, 1], pred_box[(i + 1) % 4, 1]],
                'r-', linewidth=0.5,
            )
            plt.plot(
                [pred_box[i + 4, 0], pred_box[(i + 1) % 4 + 4, 0]],
                [pred_box[i + 4, 1], pred_box[(i + 1) % 4 + 4, 1]],
                'r-', linewidth=0.5
            )
            plt.plot(
                [pred_box[i, 0], pred_box[i + 4, 0]],
                [pred_box[i, 1], pred_box[i + 4, 1]],
                'r-', linewidth=0.5
            )

    # 设置图例和显示
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300,bbox_inches='tight',pad_inches=0)
    plt.show()






if __name__ == '__main__':
    # 假设点云数据是N*3的numpy数组
    point_cloud = torch.randn(100, 3)  # 示例点云数据

    # 假设预测检测框是N*8*3的numpy数组
    pred_boxes = torch.randn(1, 8, 3).cpu()  # 示例预测检测框数据

    # 假设真值检测框是N*8*3的numpy数组
    gt_boxes = torch.randn(1, 8, 3)  # 示例真值检测框数据

    bbox_2d(gt_boxes,pred_boxes,point_cloud)



# import matplotlib.pyplot as plt
# import cv2
# from datetime import datetime
# import open3d as o3d
# import torch
# import numpy as np
# import opencood.simple_plot3d.canvas_3d as canvas_3d
# import opencood.simple_plot3d.canvas_bev as canvas_bev
#
#
# def residul_heatmap(x):
#     time = datetime.now()
#     x = torch.mean(x, dim=0)
#     max_value = torch.max(x)
#     min_value = torch.min(x)
#     normalized_tensor = (x - min_value) / (max_value - min_value)
#     # 将归一化后的张量映射到颜色空间
#     heatmap = torch.clamp(normalized_tensor, 0, 1)  # 确保值在 [0, 1] 区间内
#     heatmap = heatmap.unsqueeze(2).cpu().detach().numpy()  # 转换为 HWC 格式
#
#     # heatmap = np.linspace(0, 1, 256).reshape(1, -1)  # 生成一行256列的渐变值
#     # heatmap = np.repeat(heatmap, 50, axis=0)
#
#     heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_CIVIDIS)
#     # 可视化热图
#     time = datetime.now().strftime('%Y%m%d_%H%M%S')  # 格式化时间戳
#     plt.imsave(f'residual_picture/heatmap_{time}.png', heatmap, format='png')  # 注意 BGR 转 RGB
#     plt.axis('off')
#     plt.show()
#
#
# def aggrate_heatmap(x):
#     time = datetime.now()
#     x = torch.mean(x, dim=0)
#     max_value = torch.max(x)
#     min_value = torch.min(x)
#     normalized_tensor = (x - min_value) / (max_value - min_value)
#     # 将归一化后的张量映射到颜色空间
#     heatmap = torch.clamp(normalized_tensor, 0, 1)  # 确保值在 [0, 1] 区间内
#     heatmap = heatmap.unsqueeze(2).cpu().detach().numpy()  # 转换为 HWC 格式
#
#     # heatmap = np.linspace(0, 1, 256).reshape(1, -1)  # 生成一行256列的渐变值
#     # heatmap = np.repeat(heatmap, 50, axis=0)
#
#     heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_CIVIDIS)
#     # 可视化热图
#     time = datetime.now().strftime('%Y%m%d_%H%M%S')  # 格式化时间戳
#     plt.imsave(f'aggrate_picture/heatmap_{time}.png', heatmap, format='png')  # 注意 BGR 转 RGB
#     plt.axis('off')
#     plt.show()
#
#
# def feature_heatmap(x):
#     time = datetime.now()
#     x = torch.mean(x, dim=0)
#     max_value = torch.max(x)
#     min_value = torch.min(x)
#     normalized_tensor = (x - min_value) / (max_value - min_value)
#     # 将归一化后的张量映射到颜色空间
#     heatmap = torch.clamp(normalized_tensor, 0, 1)  # 确保值在 [0, 1] 区间内
#     heatmap = heatmap.unsqueeze(2).cpu().detach().numpy()  # 转换为 HWC 格式
#
#     # heatmap = np.linspace(0, 1, 256).reshape(1, -1)  # 生成一行256列的渐变值
#     # heatmap = np.repeat(heatmap, 50, axis=0)
#
#     heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_CIVIDIS)
#     # 可视化热图
#     time = datetime.now().strftime('%Y%m%d_%H%M%S')  # 格式化时间戳
#     plt.imsave(f'picture/heatmap_{time}.png', heatmap, format='png')  # 注意 BGR 转 RGB
#     plt.axis('off')
#     plt.show()
#
#
# def visualize_batch_heatmaps(x, grid_size=(2, 2)):
#     time = datetime.now().strftime('%Y%m%d_%H%M%S')
#     # 获取 batch 大小和图像尺寸
#     B, _, H, W = x.shape
#     rows, cols = grid_size
#
#     # 检查网格大小是否足够显示所有 batch
#     if rows * cols < B:
#         raise ValueError("网格大小不足以显示所有 batch 样本，请增加 rows 和 cols")
#
#     # 创建绘图窗口
#     fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
#     axes = axes.flatten()  # 将轴对象展平，便于迭代
#
#     # 绘制每个 batch 样本的热力图
#     for i in range(B):
#         sample = x[i]  # 取出第 i 个 batch 样本，形状为 (H, W)
#
#         sample = torch.mean(sample, dim=0)
#         # 计算归一化
#         max_value = torch.max(sample)
#         min_value = torch.min(sample)
#         normalized_tensor = (sample - min_value) / (max_value - min_value)
#         heatmap = torch.clamp(normalized_tensor, 0, 1).cpu().detach().numpy()
#
#         # 应用颜色映射
#         heatmap = np.uint8(255 * heatmap)  # 映射到 [0, 255]
#         heatmap = cv2.applyColorMap(heatmap, cv2.COLOR_BGR2RGB)
#
#         # 绘制到子图中
#         ax = axes[i]
#         ax.imshow(heatmap)
#         ax.axis('off')
#         ax.set_title(f"Batch {i + 1}")
#
#     # 如果 batch 数量少于网格数，隐藏多余的子图
#     for j in range(B, rows * cols):
#         fig.delaxes(axes[j])
#
#     # 调整布局
#     plt.tight_layout()
#
#     # 保存图像
#     plt.savefig(f'picture/batch_heatmaps_{time}.png', format='png', bbox_inches='tight')
#     plt.show()
#
#
# def bbox_2d(pre_box, gt_box, pcd):
#     time = datetime.now()
#     x = pcd[:, 0].cpu().numpy()
#     y = pcd[:, 1].cpu().numpy()
#
#     # 创建绘图
#     plt.figure(figsize=(10, 10))
#     plt.scatter(x, y, s=0.01, c='black', alpha=1)
#     pre_box = pre_box.cpu().numpy()
#     gt_box = gt_box.cpu().numpy()
#
#     # 绘制真值检测框
#     for j in range(gt_box.shape[0]):
#         true_box = gt_box[j]
#         for i in range(4):
#             plt.plot(
#                 [true_box[i, 0], true_box[(i + 1) % 4, 0]],
#                 [true_box[i, 1], true_box[(i + 1) % 4, 1]],
#                 'g-', linewidth=1.5
#             )
#             plt.plot(
#                 [true_box[i + 4, 0], true_box[(i + 1) % 4 + 4, 0]],
#                 [true_box[i + 4, 1], true_box[(i + 1) % 4 + 4, 1]],
#                 'g-', linewidth=1.5
#             )
#             plt.plot(
#                 [true_box[i, 0], true_box[i + 4, 0]],
#                 [true_box[i, 1], true_box[i + 4, 1]],
#                 'g-', linewidth=1.5
#             )
#
#     # 绘制预测检测框
#     for j in range(pre_box.shape[0]):
#         pred_box = pre_box[j]
#         for i in range(4):
#             plt.plot(
#                 [pred_box[i, 0], pred_box[(i + 1) % 4, 0]],
#                 [pred_box[i, 1], pred_box[(i + 1) % 4, 1]],
#                 'r-', linewidth=0.5,
#             )
#             plt.plot(
#                 [pred_box[i + 4, 0], pred_box[(i + 1) % 4 + 4, 0]],
#                 [pred_box[i + 4, 1], pred_box[(i + 1) % 4 + 4, 1]],
#                 'r-', linewidth=0.5
#             )
#             plt.plot(
#                 [pred_box[i, 0], pred_box[i + 4, 0]],
#                 [pred_box[i, 1], pred_box[i + 4, 1]],
#                 'r-', linewidth=0.5
#             )
#
#     # 设置图例和显示
#     plt.legend()
#     plt.axis("equal")
#     plt.savefig(f'box_2d_picture/BirdView_box_{time}.png', format='png', bbox_inches='tight')
#     plt.show()
#
#
# def point_cloud(pcd):
#     time = datetime.now()
#     x = pcd[:, 0].cpu().numpy()
#     y = pcd[:, 1].cpu().numpy()
#
#     # 创建绘图
#     plt.figure(figsize=(10, 10))
#     plt.scatter(x, y, s=0.2, c='black', alpha=0.2)
#
#     plt.axis("equal")
#     plt.savefig(f'point_cloud/point_{time}.png', format='png', bbox_inches='tight')
#     plt.show()
#
#
# def simple_visualize(pred_box_tensor, gt_box_tensor, pcd, pc_range=[-140, -38, -5, 140, 38, 3], method='bev',
#                      left_hand=False):
#     plt.figure(figsize=[(pc_range[3] - pc_range[0]) / 40, (pc_range[4] - pc_range[1]) / 40])
#     pc_range = [int(i) for i in pc_range]
#     pcd_np = pcd.cpu().numpy()
#
#     if pred_box_tensor is not None:
#         pred_box_np = pred_box_tensor.cpu().numpy()
#         # pred_name = ['pred'] * pred_box_np.shape[0]
#
#         # score = infer_result.get("score_tensor", None)
#         # if score is not None:
#         #     score_np = score.cpu().numpy()
#         #     pred_name = [f'score:{score_np[i]:.3f}' for i in range(score_np.shape[0])]
#
#         # uncertainty = infer_result.get("uncertainty_tensor", None)
#         # if uncertainty is not None:
#         #     uncertainty_np = uncertainty.cpu().numpy()
#         #     uncertainty_np = np.exp(uncertainty_np)
#         #     d_a_square = 1.6**2 + 3.9**2
#
#         # if uncertainty_np.shape[1] == 3:
#         #     uncertainty_np[:,:2] *= d_a_square
#         #     uncertainty_np = np.sqrt(uncertainty_np)
#         #     # yaw angle is in radian, it's the same in g2o SE2's setting.
#
#         #     pred_name = [f'x_u:{uncertainty_np[i,0]:.3f} y_u:{uncertainty_np[i,1]:.3f} a_u:{uncertainty_np[i,2]:.3f}' \
#         #                     for i in range(uncertainty_np.shape[0])]
#
#         # elif uncertainty_np.shape[1] == 2:
#         #     uncertainty_np[:,:2] *= d_a_square
#         #     uncertainty_np = np.sqrt(uncertainty_np) # yaw angle is in radian
#
#         #     pred_name = [f'x_u:{uncertainty_np[i,0]:.3f} y_u:{uncertainty_np[i,1]:3f}' \
#         #                     for i in range(uncertainty_np.shape[0])]
#
#         # elif uncertainty_np.shape[1] == 7:
#         #     uncertainty_np[:,:2] *= d_a_square
#         #     uncertainty_np = np.sqrt(uncertainty_np) # yaw angle is in radian
#
#         #     pred_name = [f'x_u:{uncertainty_np[i,0]:.3f} y_u:{uncertainty_np[i,1]:3f} a_u:{uncertainty_np[i,6]:3f}' \
#         #                     for i in range(uncertainty_np.shape[0])]
#
#     if gt_box_tensor is not None:
#         gt_box_np = gt_box_tensor.cpu().numpy()
#         # gt_name = ['gt'] * gt_box_np.shape[0]
#
#     if method == 'bev':
#         canvas = canvas_bev.Canvas_BEV_heading_right(
#             canvas_shape=((pc_range[4] - pc_range[1]) * 10, (pc_range[3] - pc_range[0]) * 10),
#             canvas_x_range=(pc_range[0], pc_range[3]),
#             canvas_y_range=(pc_range[1], pc_range[4]),
#             left_hand=left_hand)
#
#         canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np)  # Get Canvas Coords
#         canvas.draw_canvas_points(canvas_xy[valid_mask])  # Only draw valid points
#         if gt_box_tensor is not None:
#             canvas.draw_boxes(gt_box_np, colors=(0, 255, 0))
#         if pred_box_tensor is not None:
#             canvas.draw_boxes(pred_box_np, colors=(255, 0, 0))
#
#         # heterogeneous
#         # lidar_agent_record = infer_result.get("lidar_agent_record", None)
#         # cav_box_np = infer_result.get("cav_box_np", None)
#         # if lidar_agent_record is not None:
#         #     cav_box_np = copy.deepcopy(cav_box_np)
#         #     for i, islidar in enumerate(lidar_agent_record):
#         #         text = ['lidar'] if islidar else ['camera']
#         #         color = (0,191,255) if islidar else (255,185,15)
#         #         canvas.draw_boxes(cav_box_np[i:i+1], colors=color, texts=text)
#
#
#
#     elif method == '3d':
#         canvas = canvas_3d.Canvas_3D(left_hand=left_hand)
#         canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np)
#         canvas.draw_canvas_points(canvas_xy[valid_mask])
#         if gt_box_tensor is not None:
#             canvas.draw_boxes(gt_box_np, colors=(0, 255, 0))
#         if pred_box_tensor is not None:
#             canvas.draw_boxes(pred_box_np, colors=(255, 0, 0))
#
#         # heterogeneous
#         # lidar_agent_record = infer_result.get("lidar_agent_record", None)
#         # cav_box_np = infer_result.get("cav_box_np", None)
#         # if lidar_agent_record is not None:
#         #     # cav_box_np = copy.deepcopy(cav_box_np)
#         #     for i, islidar in enumerate(lidar_agent_record):
#         #         text = ['lidar'] if islidar else ['camera']
#         #         color = (0,191,255) if islidar else (255,185,15)
#         #         canvas.draw_boxes(cav_box_np[i:i+1], colors=color, texts=text)
#
#     else:
#         raise (f"Not Completed for f{method} visualization.")
#
#     plt.axis("off")
#
#     plt.imshow(canvas.canvas)
#     plt.tight_layout()
#     time = datetime.now()
#     plt.savefig(f'box_2d_picture/BirdView_box_{time}.png', transparent=False, dpi=500)
#     plt.clf()
#     plt.close()
#
#
# if __name__ == '__main__':
#     # 假设点云数据是N*3的numpy数组
#     point_cloud = torch.randn(100, 3)  # 示例点云数据
#
#     # 假设预测检测框是N*8*3的numpy数组
#     pred_boxes = torch.randn(1, 8, 3)  # 示例预测检测框数据
#
#     # 假设真值检测框是N*8*3的numpy数组
#     gt_boxes = torch.randn(1, 8, 3)  # 示例真值检测框数据
#     bbox_2d(gt_boxes, pred_boxes, point_cloud)









