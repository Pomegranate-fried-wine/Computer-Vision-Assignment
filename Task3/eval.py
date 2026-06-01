import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from sklearn.metrics import roc_curve, auc

from model import UNet


# =========================
# Config
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = "../DRIVE"

IMG_SIZE = 512

MODEL_PATH = "checkpoints/best_model.pth"


# =========================
# Load Model
# =========================

model = UNet().to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.eval()

print("Model Loaded Successfully")


# =========================
# Output dirs
# =========================

os.makedirs("results", exist_ok=True)
os.makedirs("curves", exist_ok=True)


# =========================
# Metrics
# =========================

TP = TN = FP = FN = 0

all_gt = []
all_prob = []


# =========================
# Eval
# =========================

with torch.no_grad():

    for i in range(1, 21):

        name = f"{i:02d}"
        print(f"Processing {name}...")

        # -------------------------
        # Image
        # -------------------------

        img_path = os.path.join(DATA_ROOT, "images", f"{name}_test.tif")
        rgb = np.array(Image.open(img_path))

        green = rgb[:, :, 1]

        clahe = cv2.createCLAHE(2.0, (8, 8))
        green = clahe.apply(green)

        green = green.astype(np.float32) / 255.0

        # resize
        green = cv2.resize(green, (IMG_SIZE, IMG_SIZE))

        x = torch.tensor(green, dtype=torch.float32)
        x = x.unsqueeze(0).unsqueeze(0).to(DEVICE)

        # -------------------------
        # Prediction
        # -------------------------

        pred = model(x)
        prob = torch.sigmoid(pred)

        prob = prob.squeeze().cpu().numpy()

        binary = (prob > 0.5).astype(np.uint8)

        # -------------------------
        # GT
        # -------------------------

        gt_path = os.path.join(DATA_ROOT, "manual", f"{name}_manual1.gif")
        gt = np.array(Image.open(gt_path))
        gt = (gt > 0).astype(np.uint8)
        gt = cv2.resize(gt, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        # -------------------------
        # ROI
        # -------------------------

        roi_path = os.path.join(DATA_ROOT, "mask", f"{name}_mask.gif")
        roi = np.array(Image.open(roi_path))
        roi = (roi > 0).astype(np.uint8)
        roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        # =========================
        # ★关键：强制三者对齐（最终保险）
        # =========================

        H, W = binary.shape

        gt = gt[:H, :W]
        roi = roi[:H, :W]
        prob = prob[:H, :W]

        # -------------------------
        # ROI evaluation
        # -------------------------

        gt_roi = gt[roi == 1]
        pred_roi = binary[roi == 1]
        prob_roi = prob[roi == 1]

        TP += np.sum((pred_roi == 1) & (gt_roi == 1))
        TN += np.sum((pred_roi == 0) & (gt_roi == 0))
        FP += np.sum((pred_roi == 1) & (gt_roi == 0))
        FN += np.sum((pred_roi == 0) & (gt_roi == 1))

        all_gt.extend(gt_roi.flatten())
        all_prob.extend(prob_roi.flatten())

        # -------------------------
        # Save visualization
        # -------------------------

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))

        ax[0].imshow(rgb)
        ax[0].set_title("Original")

        ax[1].imshow(gt, cmap='gray')
        ax[1].set_title("Ground Truth")

        ax[2].imshow(binary, cmap='gray')
        ax[2].set_title("Prediction")

        for a in ax:
            a.axis('off')

        plt.tight_layout()
        plt.savefig(f"results/{name}_compare.png", dpi=200)
        plt.close()


# =========================
# Metrics
# =========================

accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
sensitivity = TP / (TP + FN + 1e-8)
specificity = TN / (TN + FP + 1e-8)

fpr, tpr, _ = roc_curve(all_gt, all_prob)
roc_auc = auc(fpr, tpr)


# =========================
# Print results
# =========================

print("\n========== TEST RESULT ==========")
print(f"Accuracy    : {accuracy:.4f}")
print(f"Sensitivity : {sensitivity:.4f}")
print(f"Specificity : {specificity:.4f}")
print(f"AUC         : {roc_auc:.4f}")


# =========================
# ROC curve
# =========================

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
plt.plot([0, 1], [0, 1], '--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()

plt.savefig("curves/roc_curve.png", dpi=300)
plt.show()