import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from sklearn.metrics import roc_curve
from sklearn.metrics import auc


# =====================================================
# 路径配置
# =====================================================

DRIVE_DIR = "DRIVE"

IMAGE_DIR = os.path.join(DRIVE_DIR, "images")
MANUAL_DIR = os.path.join(DRIVE_DIR, "manual")
MASK_DIR = os.path.join(DRIVE_DIR, "mask")

RESULT_DIR = "results"
CURVE_DIR = "curves"

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CURVE_DIR, exist_ok=True)


# =====================================================
# 无监督血管分割
# =====================================================

def vessel_segmentation(rgb_img):

    # -----------------------------
    # 1. 绿色通道
    # -----------------------------

    green = rgb_img[:, :, 1]

    # -----------------------------
    # 2. 中值滤波
    # -----------------------------

    green = cv2.medianBlur(green, 5)

    # -----------------------------
    # 3. CLAHE增强
    # -----------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(green)

    # -----------------------------
    # 4. 反色
    # -----------------------------

    inverted = 255 - enhanced

    # -----------------------------
    # 5. Top-Hat增强
    # -----------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (15, 15)
    )

    vessel_response = cv2.morphologyEx(
        inverted,
        cv2.MORPH_TOPHAT,
        kernel
    )

    # -----------------------------
    # 6. Otsu阈值
    # -----------------------------

    _, binary = cv2.threshold(
        vessel_response,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # -----------------------------
    # 7. 闭运算
    # -----------------------------

    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel_close
    )

    # -----------------------------
    # 8. 连通域过滤
    # -----------------------------

    num_labels, labels, stats, _ = \
        cv2.connectedComponentsWithStats(binary)

    clean = np.zeros_like(binary)

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area >= 20:
            clean[labels == i] = 255

    probability_map = vessel_response.astype(
        np.float32
    ) / 255.0

    prediction = (clean > 0).astype(np.uint8)

    return prediction, probability_map


# =====================================================
# 统计量
# =====================================================

TP = TN = FP = FN = 0

all_gt = []
all_prob = []

print("Processing DRIVE test set...")

# =====================================================
# 测试集
# =====================================================

for i in range(1, 21):

    name = f"{i:02d}"

    print(f"Processing {name}...")

    img_path = os.path.join(
        IMAGE_DIR,
        f"{name}_test.tif"
    )

    gt_path = os.path.join(
        MANUAL_DIR,
        f"{name}_manual1.gif"
    )

    roi_path = os.path.join(
        MASK_DIR,
        f"{name}_mask.gif"
    )

    # -----------------------------
    # 读取图像
    # -----------------------------

    rgb = np.array(
        Image.open(img_path)
    )

    gt = np.array(
        Image.open(gt_path)
    )

    roi = np.array(
        Image.open(roi_path)
    )

    gt = (gt > 0).astype(np.uint8)
    roi = roi > 0

    # -----------------------------
    # 分割
    # -----------------------------

    pred, prob = vessel_segmentation(rgb)

    # -----------------------------
    # ROI区域评价
    # -----------------------------

    gt_roi = gt[roi]
    pred_roi = pred[roi]
    prob_roi = prob[roi]

    TP += np.sum(
        (pred_roi == 1) &
        (gt_roi == 1)
    )

    TN += np.sum(
        (pred_roi == 0) &
        (gt_roi == 0)
    )

    FP += np.sum(
        (pred_roi == 1) &
        (gt_roi == 0)
    )

    FN += np.sum(
        (pred_roi == 0) &
        (gt_roi == 1)
    )

    all_gt.extend(
        gt_roi.flatten()
    )

    all_prob.extend(
        prob_roi.flatten()
    )

    # =================================================
    # 保存对比图
    # =================================================

    fig, ax = plt.subplots(
        1,
        3,
        figsize=(12, 4)
    )

    ax[0].imshow(rgb)
    ax[0].set_title("Original")

    ax[1].imshow(
        gt,
        cmap="gray"
    )
    ax[1].set_title("Ground Truth")

    ax[2].imshow(
        pred,
        cmap="gray"
    )
    ax[2].set_title("Prediction")

    for a in ax:
        a.axis("off")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULT_DIR,
            f"{name}_compare.png"
        ),
        dpi=200
    )

    plt.close()


# =====================================================
# 指标计算
# =====================================================

accuracy = (
    TP + TN
) / (
    TP + TN + FP + FN
)

sensitivity = TP / (
    TP + FN + 1e-8
)

specificity = TN / (
    TN + FP + 1e-8
)

fpr, tpr, _ = roc_curve(
    all_gt,
    all_prob
)

roc_auc = auc(
    fpr,
    tpr
)

# =====================================================
# 输出结果
# =====================================================

print("\n========== TEST RESULT ==========")

print(
    f"Accuracy    : {accuracy:.4f}"
)

print(
    f"Sensitivity : {sensitivity:.4f}"
)

print(
    f"Specificity : {specificity:.4f}"
)

print(
    f"AUC         : {roc_auc:.4f}"
)

# =====================================================
# ROC
# =====================================================

plt.figure(figsize=(6, 6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"AUC={roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    '--'
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve"
)

plt.legend()

plt.grid(True)

plt.savefig(
    os.path.join(
        CURVE_DIR,
        "roc_curve.png"
    ),
    dpi=300
)

plt.show()