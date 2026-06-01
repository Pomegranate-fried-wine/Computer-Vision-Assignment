import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from attention_model import AttUNet


# =========================
# Config
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PATCH_SIZE = 128

BATCH_SIZE = 32

EPOCHS = 150

LR = 1e-4

PATCHES_PER_IMAGE = 200

TRAIN_IDS = [f"{i:02d}" for i in range(21, 41)]

DATA_ROOT = "../DRIVE"


# =========================
# Dataset
# =========================

class DrivePatchDataset(Dataset):

    def __init__(self):

        self.images = []
        self.masks = []

        print("Loading DRIVE training set...")

        for name in TRAIN_IDS:

            img_path = os.path.join(
                DATA_ROOT,
                "images",
                f"{name}_training.tif"
            )

            gt_path = os.path.join(
                DATA_ROOT,
                "manual",
                f"{name}_manual1.gif"
            )

            img = np.array(
                Image.open(img_path)
            )

            # Green channel
            img = img[:, :, 1]

            # CLAHE
            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8)
            )

            img = clahe.apply(img)

            img = img.astype(np.float32) / 255.0

            gt = np.array(
                Image.open(gt_path)
            )

            gt = (gt > 0).astype(np.float32)

            self.images.append(img)
            self.masks.append(gt)

        self.dataset_size = (
            len(TRAIN_IDS)
            * PATCHES_PER_IMAGE
        )

        print(
            f"Virtual Patch Number: {self.dataset_size}"
        )

    def __len__(self):
        return self.dataset_size

    def augment(self, img, mask):

        # Horizontal flip
        if random.random() > 0.5:
            img = np.fliplr(img)
            mask = np.fliplr(mask)

        # Vertical flip
        if random.random() > 0.5:
            img = np.flipud(img)
            mask = np.flipud(mask)

        # Rotate
        k = random.randint(0, 3)

        img = np.rot90(img, k)
        mask = np.rot90(mask, k)

        return img.copy(), mask.copy()

    def __getitem__(self, idx):

        img_idx = random.randint(
            0,
            len(self.images) - 1
        )

        img = self.images[img_idx]
        mask = self.masks[img_idx]

        h, w = img.shape

        x = random.randint(
            0,
            w - PATCH_SIZE
        )

        y = random.randint(
            0,
            h - PATCH_SIZE
        )

        img_patch = img[
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE
        ]

        mask_patch = mask[
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE
        ]

        img_patch, mask_patch = self.augment(
            img_patch,
            mask_patch
        )

        img_patch = torch.tensor(
            img_patch,
            dtype=torch.float32
        ).unsqueeze(0)

        mask_patch = torch.tensor(
            mask_patch,
            dtype=torch.float32
        ).unsqueeze(0)

        return img_patch, mask_patch


# =========================
# Dice Loss
# =========================

def dice_loss(pred, target):

    pred = torch.sigmoid(pred)

    smooth = 1e-6

    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (
        pred * target
    ).sum()

    dice = (
        2 * intersection + smooth
    ) / (
        pred.sum()
        + target.sum()
        + smooth
    )

    return 1 - dice


# =========================
# Prepare
# =========================

train_dataset = DrivePatchDataset()

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

model = AttUNet().to(DEVICE)

bce_loss = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=10
)

os.makedirs(
    "checkpoints",
    exist_ok=True
)

os.makedirs(
    "curves",
    exist_ok=True
)

checkpoint_path = (
    "checkpoints/best_model.pth"
)

best_loss = float("inf")

loss_history = []

# =========================
# Resume
# =========================

if os.path.exists(checkpoint_path):

    print(
        "Loading checkpoint..."
    )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=DEVICE
        )
    )


# =========================
# Training
# =========================

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for images, masks in train_loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        pred = model(images)

        loss_bce = bce_loss(
            pred,
            masks
        )

        loss_dice = dice_loss(
            pred,
            masks
        )

        loss = (
            0.3 * loss_bce
            + 0.7 * loss_dice
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = (
        running_loss
        / len(train_loader)
    )

    scheduler.step(epoch_loss)

    loss_history.append(
        epoch_loss
    )

    current_lr = optimizer.param_groups[0]['lr']

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss={epoch_loss:.6f} "
        f"LR={current_lr:.6e}"
    )

    if epoch_loss < best_loss:

        best_loss = epoch_loss

        torch.save(
            model.state_dict(),
            checkpoint_path
        )

        print(
            f"Best model saved "
            f"(Loss={best_loss:.6f})"
        )

# =========================
# Loss Curve
# =========================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    loss_history,
    linewidth=2
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "Training Loss Curve"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "curves/loss_curve.png",
    dpi=300
)

plt.show()

print("\nTraining Finished!")
print(
    f"Best Loss = {best_loss:.6f}"
)