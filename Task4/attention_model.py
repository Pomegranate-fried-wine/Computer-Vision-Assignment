import torch
import torch.nn as nn


# =========================================================
# ✔ 工具函数：解决所有尺寸不匹配问题（核心）
# =========================================================

def crop_to_match(x, ref):

    """
    将x裁剪到ref的尺寸（避免72 vs 73问题）
    """

    _, _, H, W = ref.shape

    return x[:, :, :H, :W]


# =========================================================
# ✔ 基础卷积模块
# =========================================================

class DoubleConv(nn.Module):

    def __init__(self, in_c, out_c):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# =========================================================
# ✔ Attention Gate（血管注意力核心）
# =========================================================

class AttentionGate(nn.Module):

    def __init__(self, F_g, F_l, F_int):

        super().__init__()

        self.Wg = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.Wx = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):

        g1 = self.Wg(g)
        x1 = self.Wx(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


# =========================================================
# ✔ Attention U-Net（稳定版）
# =========================================================

class AttUNet(nn.Module):

    def __init__(self):

        super().__init__()

        # -------------------------
        # Encoder
        # -------------------------

        self.enc1 = DoubleConv(1, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)

        # -------------------------
        # Bottleneck
        # -------------------------

        self.bottom = DoubleConv(256, 512)

        # -------------------------
        # Decoder + Attention
        # -------------------------

        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.att4 = AttentionGate(256, 256, 128)
        self.dec4 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.att3 = AttentionGate(128, 128, 64)
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.att2 = AttentionGate(64, 64, 32)
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.att1 = AttentionGate(32, 32, 16)
        self.dec1 = DoubleConv(64, 32)

        # -------------------------
        # Output
        # -------------------------

        self.out = nn.Conv2d(32, 1, 1)

    # =====================================================
    # forward（已彻底解决尺寸问题）
    # =====================================================

    def forward(self, x):

        # ---------------------
        # Encoder
        # ---------------------

        c1 = self.enc1(x)
        p1 = self.pool(c1)

        c2 = self.enc2(p1)
        p2 = self.pool(c2)

        c3 = self.enc3(p2)
        p3 = self.pool(c3)

        c4 = self.enc4(p3)
        p4 = self.pool(c4)

        b = self.bottom(p4)

        # ---------------------
        # Decoder 4
        # ---------------------

        u4 = self.up4(b)

        c4 = crop_to_match(c4, u4)

        c4 = self.att4(u4, c4)

        u4 = torch.cat([u4, c4], dim=1)

        u4 = self.dec4(u4)

        # ---------------------
        # Decoder 3
        # ---------------------

        u3 = self.up3(u4)

        c3 = crop_to_match(c3, u3)

        c3 = self.att3(u3, c3)

        u3 = torch.cat([u3, c3], dim=1)

        u3 = self.dec3(u3)

        # ---------------------
        # Decoder 2
        # ---------------------

        u2 = self.up2(u3)

        c2 = crop_to_match(c2, u2)

        c2 = self.att2(u2, c2)

        u2 = torch.cat([u2, c2], dim=1)

        u2 = self.dec2(u2)

        # ---------------------
        # Decoder 1
        # ---------------------

        u1 = self.up1(u2)

        c1 = crop_to_match(c1, u1)

        c1 = self.att1(u1, c1)

        u1 = torch.cat([u1, c1], dim=1)

        u1 = self.dec1(u1)

        return self.out(u1)


# =========================================================
# ✔ quick test
# =========================================================

if __name__ == "__main__":

    model = AttUNet()

    x = torch.randn(1, 1, 128, 128)

    y = model(x)

    print("output shape:", y.shape)

    total = sum(p.numel() for p in model.parameters())

    print("params:", total)