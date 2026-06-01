import cv2
import numpy as np
import os
import glob
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 自动检测 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f" 💡 当前运行设备: {device}")

def cv_imread(file_path, flags=cv2.IMREAD_COLOR):
    """支持读取中文路径的图片"""
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flags)

def cv_imwrite(file_path, image):
    """支持写入中文路径的图片"""
    ext = os.path.splitext(file_path)[1]
    result, nparr = cv2.imencode(ext, image)
    if result:
        nparr.tofile(file_path)

# =========================================================
# 1. GPU 加速底层算子（基于 PyTorch Tensor）
# =========================================================

def gaussian_kernel_gpu(size, sigma=1.0):
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2.0 * sigma**2))
    kernel_1d = g / g.sum()
    kernel_2d = kernel_1d.unsqueeze(1) @ kernel_1d.unsqueeze(0)
    return kernel_2d

def convolve_gpu(image_tensor, kernel_tensor):
    pad = kernel_tensor.shape[0] // 2
    img_pad = F.pad(image_tensor.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), mode='reflect')
    weight = kernel_tensor.unsqueeze(0).unsqueeze(0)
    out = F.conv2d(img_pad, weight)
    return out.squeeze()

def sobel_filters_gpu(image_tensor):
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device)
    
    grad_x = convolve_gpu(image_tensor, kx)
    grad_y = convolve_gpu(image_tensor, ky)
    
    magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    direction = torch.atan2(grad_y, grad_x) * (180.0 / np.pi)
    direction = torch.where(direction < 0, direction + 360.0, direction)
    
    return magnitude, direction

def non_max_suppression_gpu(magnitude, direction):
    H, W = magnitude.shape
    suppressed = torch.zeros_like(magnitude)
    
    angle = direction % 180.0
    m0 = (angle < 22.5) | (angle >= 157.5)
    m45 = (angle >= 22.5) & (angle < 67.5)
    m90 = (angle >= 67.5) & (angle < 112.5)
    m135 = (angle >= 112.5) & (angle < 157.5)
    
    mag_r = torch.roll(magnitude, shifts=1, dims=1)
    mag_l = torch.roll(magnitude, shifts=-1, dims=1)
    mag_u = torch.roll(magnitude, shifts=1, dims=0)
    mag_d = torch.roll(magnitude, shifts=-1, dims=0)
    
    mag_ru = torch.roll(torch.roll(magnitude, shifts=1, dims=0), shifts=-1, dims=1)
    mag_ld = torch.roll(torch.roll(magnitude, shifts=-1, dims=0), shifts=1, dims=1)
    mag_lu = torch.roll(torch.roll(magnitude, shifts=1, dims=0), shifts=1, dims=1)
    mag_rd = torch.roll(torch.roll(magnitude, shifts=-1, dims=0), shifts=-1, dims=1)
    
    suppressed[m0] = magnitude[m0] * ((magnitude[m0] >= mag_r[m0]) & (magnitude[m0] >= mag_l[m0]))
    suppressed[m90] = magnitude[m90] * ((magnitude[m90] >= mag_u[m90]) & (magnitude[m90] >= mag_d[m90]))
    suppressed[m45] = magnitude[m45] * ((magnitude[m45] >= mag_ru[m45]) & (magnitude[m45] >= mag_ld[m45]))
    suppressed[m135] = magnitude[m135] * ((magnitude[m135] >= mag_lu[m135]) & (magnitude[m135] >= mag_rd[m135]))
    
    return suppressed

def threshold_gpu(suppressed, low_ratio, high_ratio):
    high_th = suppressed.max() * high_ratio
    low_th = high_th * low_ratio
    
    thresholded = torch.zeros_like(suppressed, dtype=torch.uint8)
    weak = 75
    strong = 255
    
    thresholded[suppressed >= high_th] = strong
    thresholded[(suppressed >= low_th) & (suppressed < high_th)] = weak
    
    return thresholded, weak, strong

def hysteresis_evaporate(thresholded, weak, strong):
    thresholded_cpu = thresholded.cpu().numpy()
    all_edges_mask = np.uint8(thresholded_cpu > 0)
    
    num_labels, labels = cv2.connectedComponents(all_edges_mask, connectivity=8)
    final_edges = np.zeros_like(thresholded_cpu, dtype=np.uint8)
    
    strong_mask = (thresholded_cpu == strong)
    unique_labels = np.unique(labels[strong_mask])
    
    for label in unique_labels:
        if label == 0: continue
        final_edges[labels == label] = 255
        
    return final_edges

# =========================================================
# 2. 核心流水线
# =========================================================

def canny_edge_detection_pipeline(img_path, g_size, sigma, low_ratio, high_ratio):
    img = cv_imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    
    img_tensor = torch.tensor(img, dtype=torch.float32, device=device)
    
    blurred = convolve_gpu(img_tensor, gaussian_kernel_gpu(g_size, sigma))
    magnitude, direction = sobel_filters_gpu(blurred)
    suppressed = non_max_suppression_gpu(magnitude, direction)
    thresholded, weak, strong = threshold_gpu(suppressed, low_ratio, high_ratio)
    
    final_edges = hysteresis_evaporate(thresholded, weak, strong)
    return final_edges

def calculate_metrics_with_tolerance(pred, gt, tolerance=2):
    gt_bin = (gt > 127).astype(np.uint8)
    pred_bin = (pred > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * tolerance + 1, 2 * tolerance + 1))
    gt_dilated = cv2.dilate(gt_bin, kernel)
    pred_dilated = cv2.dilate(pred_bin, kernel)

    tp = np.sum((pred_bin == 1) & (gt_dilated == 1))
    fp = np.sum((pred_bin == 1) & (gt_dilated == 0))
    fn = np.sum((gt_bin == 1) & (pred_dilated == 0))

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f1

# =========================================================
# 3. 主函数：全量三合一拼接保存与展示
# =========================================================

if __name__ == "__main__":
    # 数据集根目录路径
    base_dir = r"E:\北航\大三下\cv\图像处理与机器视觉_大作业\BIPED"
    img_dir = os.path.join(base_dir, "imgs", "test", "rgbr")
    gt_dir = os.path.join(base_dir, "edge_maps", "test", "rgbr")
    
    # 🎯 定位至当前脚本所在的 Task1 目录下，创建保存文件夹
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_script_dir, "canny_results_output")
    os.makedirs(output_dir, exist_ok=True)

    img_paths = glob.glob(os.path.join(img_dir, "*.jpg"))
    print(f"📊 成功加载 {len(img_paths)} 张测试图片。")
    print(f"📁 [三合一对比图] 将全量自动保存在:\n👉 {output_dir}\n")

    # 超参数配置（可根据需要微调）
    gaussian_size = 5          
    sigma = 1.2                
    low_threshold_ratio = 0.40  
    high_threshold_ratio = 0.24 
    tolerance = 2               

    total_precision, total_recall, total_f1 = 0, 0, 0
    count = 0
    vis_images = [] # 缓存前 3 张用于最后屏幕弹窗展示

    for img_path in tqdm(img_paths, desc="GPU 批量拼接处理中"):
        filename = os.path.basename(img_path)
        gt_name = filename.replace('.jpg', '.png')
        gt_path = os.path.join(gt_dir, gt_name)
        if not os.path.exists(gt_path): continue

        # 1. 读取原彩色图和 GT 灰度图
        img_color = cv_imread(img_path, cv2.IMREAD_COLOR)
        gt_img = cv_imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        # 2. 运行 Canny 算法流水线得到边缘
        pred_edge = canny_edge_detection_pipeline(
            img_path, gaussian_size, sigma, low_threshold_ratio, high_threshold_ratio
        )
        if pred_edge is None: continue

        # 3. 🎯 核心逻辑：将单通道灰度图转换为 3 通道 BGR，用于水平矩阵拼接
        gt_color = cv2.cvtColor(gt_img, cv2.COLOR_GRAY2BGR)
        pred_color = cv2.cvtColor(pred_edge, cv2.COLOR_GRAY2BGR)

        # 4. 🎯 矩阵水平横向拼接：[ 真实原图 | 真值标签 | Canny输出 ]
        combined_result = np.hstack((img_color, gt_color, pred_color))

        # 5. 保存三合一对比图到 Task1/canny_results_output
        save_path = os.path.join(output_dir, filename.replace('.jpg', '_combined.png'))
        cv_imwrite(save_path, combined_result)

        # 指标计算
        p, r, f1 = calculate_metrics_with_tolerance(pred_edge, gt_img, tolerance=tolerance)
        total_precision += p
        total_recall += r
        total_f1 += f1
        count += 1

        # 缓存前 3 张用于 matplotlib 展示 (注意 matplotlib 绘图需要 RGB 格式)
        if count <= 3:
            img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
            gt_rgb = cv2.cvtColor(gt_img, cv2.COLOR_GRAY2RGB)
            pred_rgb = cv2.cvtColor(pred_edge, cv2.COLOR_GRAY2RGB)
            vis_images.append((img_rgb, gt_rgb, pred_rgb, filename))

    if count > 0:
        print(f"\n✨ --- 最终全量测试集指标评估报告 ({count}张) ---")
        print(f"Average Precision: {total_precision / count:.4f}")
        print(f"Average Recall:    {total_recall / count:.4f}")
        print(f"Average F1-Score:  {total_f1 / count:.4f}")

    # 全量处理结束后，弹窗展示前3组拼接效果供确认
    if vis_images:
        print("\n🎨 正在调取窗口展示部分拼接样例...")
        fig, axes = plt.subplots(len(vis_images), 1, figsize=(15, 4 * len(vis_images)))
        if len(vis_images) == 1: axes = [axes]
        
        for i, (img, gt, pred, name) in enumerate(vis_images):
            # 将三个矩阵在内存中再拼一次用于 matplotlib 统一显示
            imshow_combined = np.hstack((img, gt, pred))
            axes[i].imshow(imshow_combined)
            axes[i].set_title(f"样例 {i+1}: {name} (左:原图 | 中:GT | 右:Canny输出)", fontsize=11)
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.show()