import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from mpl_toolkits.axes_grid1 import ImageGrid


class PointCloudVisualizer:
    def __init__(self, point_cloud_no_delay, point_cloud_with_delay, bev_features_no_delay, bev_features_with_delay):
        """
        变量
        point_cloud_no_delay: 无时延点云数据 (numpy数组, shape=(1, N, 4))
        point_cloud_with_delay: 有时延点云数据 (numpy数组, shape=(1, N, 4))
        :bev_features_no_delay: 无时延BEV特征图 (numpy数组, shape=(N, C, H, W))
        bev_features_with_delay: 有时延BEV特征图 (numpy数组, shape=(N, C, H, W))
        pred_box_tensor: (n, 8, 3)
        gt_boxx_tensor: (n, 8, 3)

        """
        
        self.point_cloud_no_delay = point_cloud_no_delay
        self.point_cloud_with_delay = point_cloud_with_delay
        self.bev_features_no_delay = bev_features_no_delay
        self.bev_features_with_delay = bev_features_with_delay

    def save_predict_result(self,predict_result):
        if len(predict_result)==3:
            self.pred_box_tensor, self.pred_score, self.gt_box_tensor=predict_result
        elif len(predict_result)==6:
            self.non_delay_pred_box_tensor, self.non_delay_pred_score, self.non_delay_gt_box_tensor,\
                self.pred_box_tensor, self.pred_score, self.gt_box_tensor = predict_result
            
    
    @staticmethod
    def _draw_box(ax, corners, color, linewidth=1):
        """
        绘制 3D box 的鸟瞰投影框 (8 个角 -> 4 个角)
        corners: (8, 3)
        """
        # 取 z=0 平面（BEV）投影
        x = corners[:, 0]
        y = corners[:, 1]
        # 8 个顶点的顺序：0-1-2-3-0 组成底面矩形
        for i in range(4):
            ax.plot([x[i], x[(i + 1) % 4]],
                    [y[i], y[(i + 1) % 4]],
                    color=color, linewidth=linewidth)

    # ------------------------------------------------------------------
    # 主可视化函数
    # ------------------------------------------------------------------
    def visualize(self, figsize=(16, 8)):
        """
        range_x / range_y: 鸟瞰图坐标范围
        """
        if self.bev_features_no_delay.shape[0]<=1:
            return 
        assert self.pred_box_tensor is not None, "请先给 pred_box_tensor 赋值"
        assert self.gt_box_tensor is not None, "请先给 gt_boxx_tensor 赋值"

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        # 1. 无时延点云
        ax = axes[0]
        pc = self.point_cloud_no_delay[0].cpu().numpy()  # (N, 4)
        ax.scatter(pc[:, 0], pc[:, 1], s=0.2, c='black')
        for box in self.pred_box_tensor:
            box=box.cpu().numpy()
            self._draw_box(ax, box, 'red')
        for box in self.gt_box_tensor:
            box=box.cpu().numpy()
            self._draw_box(ax, box, 'green')
        ax.set_title('Point Cloud (no delay)')
        ax.set_aspect('equal')

        # 2. 有时延点云
        ax = axes[2]
        pc = self.point_cloud_with_delay[0].cpu().numpy()  # (N, 4)
        ax.scatter(pc[:, 0], pc[:, 1], s=0.2, c='black')
        for box in self.pred_box_tensor:
            box=box.cpu().numpy()
            self._draw_box(ax, box, 'red')
        for box in self.gt_box_tensor:
            box=box.cpu().numpy()
            self._draw_box(ax, box, 'green')
        ax.set_title('Point Cloud (with delay)')
        ax.set_aspect('equal')

        # 3. 无时延 BEV 特征
        ax = axes[1]
        bev = self.bev_features_no_delay[1].cpu().numpy()  # (C, H, W)
        bev_img = np.mean(bev, axis=0)  # (H, W)
        ax.imshow(bev_img, cmap='jet', origin='lower')
        ax.set_title('BEV Feature (no delay)')

        # 4. 有时延 BEV 特征
        ax = axes[3]
        bev = self.bev_features_with_delay[1].cpu().numpy() # (C, H, W)
        bev_img = np.mean(bev, axis=0)  # (H, W)
        ax.imshow(bev_img, cmap='jet', origin='lower')
        ax.set_title('BEV Feature (with delay)')

        # plt.tight_layout()
        # plt.show()
        plt.savefig(f'visualize_result/latency/visualize_{time.time()}.png', dpi=300, bbox_inches='tight')


    def visualize_point_clouds(self, save_path="visualized_point_clouds.png"):
        """
        可视化有时延和无时延的点云数据
        :param save_path: 保存路径
        """
        fig = plt.figure(figsize=(12, 6))

        # 无时延点云
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(self.point_cloud_no_delay[:, 0], self.point_cloud_no_delay[:, 1], self.point_cloud_no_delay[:, 2], c='b', marker='o')
        ax1.set_title("No Delay Point Cloud")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")

        # 有时延点云
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(self.point_cloud_with_delay[:, 0], self.point_cloud_with_delay[:, 1], self.point_cloud_with_delay[:, 2], c='r', marker='o')
        ax2.set_title("With Delay Point Cloud")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

    def visualize_bev_features(self, save_path="visualized_bev_features.png"):
        """
        可视化有时延和无时延的BEV特征图
        :param save_path: 保存路径
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # 无时延BEV特征图
        axes[0].imshow(self.bev_features_no_delay, cmap='viridis')
        axes[0].set_title("No Delay BEV Features")
        axes[0].axis('off')

        # 有时延BEV特征图
        axes[1].imshow(self.bev_features_with_delay, cmap='viridis')
        axes[1].set_title("With Delay BEV Features")
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

# 示例用法
if __name__ == "__main__":
    # 示例数据
    np.random.seed(0)
    point_cloud_no_delay = np.random.rand(100, 3) * 10  # 无时延点云
    point_cloud_with_delay = np.random.rand(100, 3) * 10  # 有时延点云
    bev_features_no_delay = np.random.rand(100, 100)  # 无时延BEV特征图
    bev_features_with_delay = np.random.rand(100, 100)  # 有时延BEV特征图

    visualizer = PointCloudVisualizer(point_cloud_no_delay, point_cloud_with_delay, bev_features_no_delay, bev_features_with_delay)
    visualizer.visualize_point_clouds(save_path="point_clouds.png")
    visualizer.visualize_bev_features(save_path="bev_features.png")