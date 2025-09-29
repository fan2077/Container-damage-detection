import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


# 定义CBAM模块
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=1):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=1, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        ca_weight = self.ca(x)
        x = x * ca_weight
        sa_weight = self.sa(x)
        x = x * sa_weight
        return x, ca_weight, sa_weight


# 加载和预处理图像
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # 添加批次维度
    return image


# 显示图像
def show_image(tensor, title=None):
    image = tensor.clone().detach().squeeze(0)
    image = image.permute(1, 2, 0)  # 将通道维度移到最后
    image = image * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])  # 反归一化
    image = image.numpy()
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()


# 可视化注意力权重
def visualize_attention(weight, title=None):
    weight = weight.squeeze(0).cpu().detach().numpy()
    plt.imshow(weight, cmap='viridis')
    if title:
        plt.title(title)
    plt.axis('off')
    plt.colorbar()
    plt.show()


# 处理和显示图像
def process_and_show(image_path):
    image = load_image(image_path)

    # 显示原始图像
    show_image(image, title='Original Image')

    # 定义CBAM模块
    cbam = CBAM(in_planes=3, ratio=1, kernel_size=7)

    # 处理图像
    with torch.no_grad():
        output, ca_weight, sa_weight = cbam(image)

    # 显示通道注意力权重
    visualize_attention(ca_weight.mean(dim=1, keepdim=True), title='Channel Attention')

    # 显示空间注意力权重
    visualize_attention(sa_weight, title='Spatial Attention')

    # 显示处理后的图像
    show_image(output, title='CBAM Processed Image')


# 示例图像路径
image_path = '1F.JPG'  # 替换为实际图像路径

# 运行示例
process_and_show(image_path)
