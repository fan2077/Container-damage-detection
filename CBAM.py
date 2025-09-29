import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import map_coordinates


class VectorFieldGenerator:
    @staticmethod
    def generate_fractal_vector_field(shape, octaves=4, persistence=0.5):
        """生成分形向量场"""
        h, w = shape
        x = np.zeros((h, w, 2))

        for _ in range(octaves):
            scale = 2 ** np.random.uniform(1, 3)
            rand_angle = np.random.uniform(0, 2 * np.pi, (int(h / scale) + 2, int(w / scale) + 2))

            # 生成随机角度场
            dx = np.cos(rand_angle)
            dy = np.sin(rand_angle)

            # 双线性插值
            x_coords = np.linspace(0, rand_angle.shape[1] - 1, w)
            y_coords = np.linspace(0, rand_angle.shape[0] - 1, h)
            xx, yy = np.meshgrid(x_coords, y_coords)

            dx = map_coordinates(dx, [yy, xx], order=1, mode='wrap')
            dy = map_coordinates(dy, [yy, xx], order=1, mode='wrap')

            x[..., 0] += dx * persistence ** _
            x[..., 1] += dy * persistence ** _

        return x / octaves


def create_red_blue_colormap():
    """创建经典红蓝渐变色图"""
    return LinearSegmentedColormap.from_list("red_blue", [
        (0.0, "#00008B"), (0.2, "#1E90FF"),
        (0.4, "#00BFFF"), (0.6, "#00FFFF"),
        (0.7, "#00FF00"), (0.8, "#FFFF00"),
        (1.0, "#FF0000")
    ])


def generate_multi_focus_heatmap(
        image_path,
        centers,  # 中心点列表 [{"x":0.3,"y":0.5,"sigma":0.2,"turb":0.4}, ...]
        global_turb=0.2,  # 全局湍流强度
        intensity=0.7,
        blend_mode='hybrid'
):
    # 参数校验
    assert 0 <= global_turb <= 1, "湍流强度需在0-1之间"
    assert len(centers) > 0, "至少需要1个中心点"

    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("图像读取失败")
    h, w = img.shape[:2]

    # 生成全局向量场
    vector_field = VectorFieldGenerator.generate_fractal_vector_field(
        (h, w), octaves=4, persistence=0.6
    ) * global_turb * 50

    # 初始化热力叠加场
    heatmap = np.zeros((h, w))

    for center in centers:
        # 解析单个中心参数
        cx = int(np.clip(center['x'] * w, 0, w - 1))
        cy = int(np.clip(center['y'] * h, 0, h - 1))
        sigma = max(int(center['sigma'] * min(h, w)), 1)
        local_turb = center.get('turb', 0.3)

        # 生成基础椭圆场
        y_grid, x_grid = np.indices((h, w))
        dx = (x_grid - cx) / sigma
        dy = (y_grid - cy) / sigma
        base = np.exp(-(dx ** 2 + dy ** 2))

        # 应用局部扰动
        if local_turb > 0:
            local_field = VectorFieldGenerator.generate_fractal_vector_field(
                (h, w), octaves=3, persistence=0.7
            ) * local_turb * 30

            # 坐标偏移
            warped_x = x_grid + local_field[..., 0]
            warped_y = y_grid + local_field[..., 1]

            # 边界处理
            warped_x = np.clip(warped_x, 0, w - 1)
            warped_y = np.clip(warped_y, 0, h - 1)

            # 重采样
            base = map_coordinates(base, [warped_y, warped_x], order=3)

        # 叠加全局扰动
        warped_x = x_grid + vector_field[..., 0]
        warped_y = y_grid + vector_field[..., 1]
        warped_x = np.clip(warped_x, 0, w - 1)
        warped_y = np.clip(warped_y, 0, h - 1)
        final_base = map_coordinates(base, [warped_y, warped_x], order=3)

        heatmap += final_base

    # 归一化处理
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # 应用颜色映射
    cmap = create_red_blue_colormap()
    heatmap_color = (cmap(heatmap)[..., :3] * 255).astype(np.uint8)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_RGB2BGR)

    # 高级混合
    if blend_mode == 'hybrid':
        # 混合策略：高频部分使用叠加，低频部分使用透明度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200) / 255.0
        mask = np.clip(heatmap[..., None] + edges[..., None], 0, 1)
        blended = img * (1 - mask) + heatmap_color * mask
        blended = blended.astype(np.uint8)
    else:
        blended = cv2.addWeighted(img, 1 - intensity, heatmap_color, intensity, 0)

    return blended


# 使用示例
if __name__ == "__main__":
    config = {
        "image_path": "1F.JPG",
        "centers": [
            {
                "x": 0.2,
                "y": 0.3,
                "sigma": 0.5,  # 基础扩散范围
                "turb": 0.5  # 局部扰动强度
            },
            {
                "x": 0.5,
                "y": 0.6,
                "sigma": 0.5,
                "turb": 0.4
            }
        ],
        "global_turb": 0.3,  # 全局扰动强度
        "intensity": 0.7,
        "blend_mode": "hybrid"
    }

    result = generate_multi_focus_heatmap(**config)
    cv2.imwrite("multi_focus_heatmap.jpg", result)