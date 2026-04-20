"""
=========================================================
5 CẢI TIẾN SO VỚI NETTOP GỐC:

[A] Multi-scale DW 5×5 trên XZ/YZ (MS_TOP)
    NetTOP gốc: Conv2d(C,C,k=3,groups=C) trên mặt phẳng XZ/YZ
    Cải tiến  : Conv1d k=3 + Conv1d k=5 song song → element-wise add
    Phép toán : DW Conv + add (giống MixConv 2019)

[B] Multi-scale DW 5×5 trên XY (MS_TOP)
    NetTOP gốc: conv3x3_dw_block trên nhánh spatial XY
    Cải tiến  : Conv2d(C,C,k=3,groups=C) + Conv2d(C,C,k=5,groups=C) → add
    Phép toán : DW Conv2d + add (giống Inception parallel branch)

[C] Learnable fusion σ(α⊙xz + β⊙yz) (MS_TOP)
    NetTOP gốc: F.sigmoid(xz * yz)  — nhân cứng
    Cải tiến  : sigmoid(alpha * xz + beta * yz), alpha/beta = nn.Parameter
    Phép toán : weighted sum + sigmoid (giống γ,β trong BatchNorm)

[D] Dual-Statistic SE — DualSE = MAF_ChannelGate (MAF)
    NetTOP gốc: SE = GAP → FC → ReLU → FC → σ
    Cải tiến  : [GAP ; StdPool] → FC → ReLU → FC → σ
    Phép toán : Pooling + cat + Linear + Sigmoid (cùng họ SE, thêm std())

[E] Spatial Gate Conv 7×7 — MAF_SpatialGate (MAF)
    NetTOP gốc: KHÔNG có spatial attention
    Cải tiến  : mean(C) + max(C) → Conv2d(2,1,k=7) → sigmoid → scale
    Phép toán : ChannelPool + Conv2d(99 params) + Sigmoid (= CBAM Spatial)


"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import models.SE_Attention


# =========================================================
# DropPath (Stochastic Depth)
# =========================================================
class DropPath(nn.Module):
    """Drop toàn bộ residual branch theo xác suất p khi training.
    Inference: identity. p=0 → zero overhead."""
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask.div_(keep)

    def extra_repr(self):
        return f"p={self.p:.4f}"


# =========================================================
# Basic layers — self-contained (không cần common.py)
# =========================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p, groups=1,
                 bias=False, use_bn=True, act="relu"):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=bias)
        self.bn   = nn.BatchNorm2d(out_ch) if use_bn else None
        self.act  = nn.ReLU(inplace=True) if act == "relu" else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn  is not None: x = self.bn(x)
        if self.act is not None: x = self.act(x)
        return x


def conv1x1(in_ch, out_ch, stride=1, groups=1, use_bn=True, act="relu"):
    return ConvBlock(in_ch, out_ch, 1, stride, 0, groups, False, use_bn, act)

def conv3x3(in_ch, out_ch, stride=1, groups=1, use_bn=True, act="relu"):
    return ConvBlock(in_ch, out_ch, 3, stride, 1, groups, False, use_bn, act)


class Classifier(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, num_classes, 1, bias=True)

    def forward(self, x):
        return self.conv(x).view(x.size(0), -1)

    def init_params(self):
        nn.init.xavier_normal_(self.conv.weight, gain=1.0)
        nn.init.constant_(self.conv.bias, 0)

class SE(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        s = self.pool(x).view(b, c)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s)).view(b, c, 1, 1)
        return x * s
# =========================================================
# LightSE — SE nhẹ trên nhánh shortcut
# =========================================================
class LightSE(nn.Module):
    """SE 
    Giúp shortcut cũng được attention-guided, ổn định gradient.
    Phép toán: GAP → Linear → ReLU → Linear → Sigmoid → scale."""
    def __init__(self, ch, r=16):
        super().__init__()
        mid = max(ch // r, 4)
        self.fc1 = nn.Linear(ch, mid)
        self.fc2 = nn.Linear(mid, ch)

    def forward(self, x):
        b, c, _, _ = x.size()
        s = F.adaptive_avg_pool2d(x, 1).view(b, c)          # GAP
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s), inplace=True)))
        return x * s.view(b, c, 1, 1)


# =========================================================
# [D] DualSE = MAF_ChannelGate
#     CẢI TIẾN D: GAP + StdPool → FC → ReLU → FC → σ
#     (NetTOP gốc chỉ dùng GAP đơn thuần trong SE)
# =========================================================
class MAF_ChannelGate(nn.Module):
    """Dual-Statistic Channel Attention.

    NetTOP gốc (SE):  GAP(x) → FC → ReLU → FC → σ
    Cải tiến [D]   : [GAP(x) ; Std(x)] → FC → ReLU → FC → σ

    Thêm StdPool bắt được phân tán của feature map —
    channel nào có std cao = thông tin đa dạng → được trọng số cao hơn.
    Phép toán: AdaptiveAvgPool + tensor.std() + cat + Linear×2 + Sigmoid.
    """
    def __init__(self, ch, r=16):
        super().__init__()
        mid = max(ch // r, 4)
        self.fc1 = nn.Linear(ch * 2, mid)   # input: [avg ; std] → 2C
        self.fc2 = nn.Linear(mid, ch)

    def forward(self, x):
        b, c, h, w = x.size()
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)              # (B,C) — GAP
        std = x.view(b, c, -1).std(dim=2, unbiased=False)         # (B,C) — StdPool
        std = std.clamp(min=1e-5)
        desc  = torch.cat([avg, std], dim=1)                       # (B, 2C)
        scale = torch.sigmoid(self.fc2(F.relu(self.fc1(desc), inplace=True)))
        return x * scale.view(b, c, 1, 1)


# =========================================================
# [E] Spatial Gate — MAF_SpatialGate
#     CẢI TIẾN E: ChannelPool → Conv2d(2,1,k=7) → σ → scale
#     (NetTOP gốc KHÔNG có spatial attention)
# =========================================================
class MAF_SpatialGate(nn.Module):
    """Spatial Attention via Conv 7×7 (= CBAM Spatial Branch).

    NetTOP gốc: không có spatial attention.
    Cải tiến [E]: mean(C) + max(C) → Conv2d(2→1, k=7) → sigmoid → scale.

    Params: 2×1×7×7 + 1 bias = 99 params — cực nhẹ.
    Phép toán: Channel pooling (mean+max) + Conv2d + Sigmoid.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)           # (B,1,H,W) — channel avg pool
        mx, _ = x.max(dim=1, keepdim=True)          # (B,1,H,W) — channel max pool
        desc = torch.cat([avg, mx], dim=1)           # (B,2,H,W)
        att  = torch.sigmoid(self.conv(desc))        # (B,1,H,W) — spatial map
        return x * att                               # element-wise scale


# =========================================================
# MAF = DualSE [D] + Spatial [E] — nối tiếp
# =========================================================
class MAF(nn.Module):
    """Multi-scale Attention Fusion (Channel → Spatial).
    Kết hợp cải tiến [D] và [E]:
      Channel: DualSE  (GAP+STD → FC×2 → σ)
      Spatial: SpatialGate (ChannelPool → Conv7×7 → σ)
    """
    def __init__(self, ch, r=16):
        super().__init__()
        self.ch_gate = MAF_ChannelGate(ch, r)   # [D]
        self.sp_gate = MAF_SpatialGate()         # [E]

    def forward(self, x):
        x = self.ch_gate(x)   # channel re-weighting
        x = self.sp_gate(x)   # spatial re-weighting
        return x


# =========================================================
# [A][B][C] MS_TOP — Multi-Scale Three Orthogonal Planes
# =========================================================
class MS_TOP(nn.Module):
    """Multi-Scale TOP Operator — 3 cải tiến A, B, C tích hợp.

    [B] XY branch: DW 3×3 + DW 5×5 → BN → ReLU
        NetTOP gốc: chỉ dùng conv3x3_dw_block (DW k=3 đơn scale)
        Cải tiến  : Conv2d(C,C,k=3,g=C) + Conv2d(C,C,k=5,g=C) → add

    [A] XZ/YZ branch: Conv1d k=3 + Conv1d k=5 → BN → ReLU
        NetTOP gốc: torch.transpose + DW3×3 (sai chiều hình học)
        Cải tiến  : permute+view → Conv1d k=3 + Conv1d k=5 → add

    [C] Fusion: gate = σ(α⊙f_xz + β⊙f_yz)
        NetTOP gốc: F.sigmoid(xz * yz) — cứng
        Cải tiến  : sigmoid(alpha*xz + beta*yz), alpha/beta = nn.Parameter
    """
    def __init__(self, ch, stride=1):
        super().__init__()
        self.stride = stride

        # [B] XY: Multi-scale DW 2D (cải tiến B)
        self.dw_xy_3 = nn.Conv2d(ch, ch, 3, stride, 1, groups=ch, bias=False)
        self.dw_xy_5 = nn.Conv2d(ch, ch, 5, stride, 2, groups=ch, bias=False)
        self.bn_xy   = nn.BatchNorm2d(ch)

        # [A] XZ: Multi-scale DW 1D (cải tiến A)
        self.dw_xz_3 = nn.Conv1d(ch, ch, 3, 1, 1, groups=ch, bias=False)
        self.dw_xz_5 = nn.Conv1d(ch, ch, 5, 1, 2, groups=ch, bias=False)
        self.bn_xz   = nn.BatchNorm2d(ch)

        # [A] YZ: Multi-scale DW 1D (cải tiến A)
        self.dw_yz_3 = nn.Conv1d(ch, ch, 3, 1, 1, groups=ch, bias=False)
        self.dw_yz_5 = nn.Conv1d(ch, ch, 5, 1, 2, groups=ch, bias=False)
        self.bn_yz   = nn.BatchNorm2d(ch)

        # Stride handling cho XZ/YZ
        self.pool = nn.AvgPool2d(stride, stride) if stride > 1 else nn.Identity()

        # [C] Learnable fusion parameters (cải tiến C)
        # Thay F.sigmoid(xz * yz) bằng sigmoid(alpha*xz + beta*yz)
        self.alpha = nn.Parameter(torch.full((1, ch, 1, 1), 0.5))
        self.beta  = nn.Parameter(torch.full((1, ch, 1, 1), 0.5))

    def forward(self, x):
        b, c, h, w = x.size()

        # --- [B] XY branch: DW 3×3 + DW 5×5 song song ---
        f_xy = self.dw_xy_3(x) + self.dw_xy_5(x)          # multi-scale add
        f_xy = F.relu(self.bn_xy(f_xy), inplace=True)

        # --- [A] XZ branch: permute → Conv1d k=3+k=5 → reshape back ---
        # x: (B,C,H,W) → permute(0,2,1,3) → (B,H,C,W) → view → (B*H, C, W)
        t_xz = x.permute(0, 2, 1, 3).contiguous().view(b * h, c, w)
        f_xz = self.dw_xz_3(t_xz) + self.dw_xz_5(t_xz)   # (B*H, C, W)
        f_xz = f_xz.view(b, h, c, w).permute(0, 2, 1, 3).contiguous()
        f_xz = F.relu(self.pool(self.bn_xz(f_xz)), inplace=True)

        # --- [A] YZ branch: permute → Conv1d k=3+k=5 → reshape back ---
        # x: (B,C,H,W) → permute(0,3,1,2) → (B,W,C,H) → view → (B*W, C, H)
        t_yz = x.permute(0, 3, 1, 2).contiguous().view(b * w, c, h)
        f_yz = self.dw_yz_3(t_yz) + self.dw_yz_5(t_yz)   # (B*W, C, H)
        f_yz = f_yz.view(b, w, c, h).permute(0, 2, 3, 1).contiguous()
        f_yz = F.relu(self.pool(self.bn_yz(f_yz)), inplace=True)

        # --- [C] Learnable fusion: σ(α⊙xz + β⊙yz) thay sigmoid(xz*yz) ---
        gate = torch.sigmoid(self.alpha * f_xz + self.beta * f_yz)
        return F.relu(f_xy * gate, inplace=True)


# =========================================================
# FR-PDP Block — tích hợp cả 5 cải tiến
# =========================================================
class FR_PDP_block(nn.Module):
    """Full-Residual Parallel Depthwise + Pointwise Block.

    Luồng:
        Main   : Pw1(no BN/act) → MS_TOP[A,B,C] → Pw2 → MAF[D,E] → DropPath
        Shortcut: identity  hoặc  PwR → LightSE  (khi stride>1 hoặc ch thay đổi)
        Output : main + shortcut
    """
    def __init__(self, in_ch, out_ch, stride, drop_path_rate=0.0):
        super().__init__()
        self.stride = stride
        self.in_ch  = in_ch
        self.out_ch = out_ch

        # Main branch
        self.Pw1       = conv1x1(in_ch, in_ch, use_bn=False, act=None)  # mixing, no norm
        self.TOP       = MS_TOP(ch=in_ch, stride=stride)                 # [A][B][C]
        self.Pw2       = conv1x1(in_ch, out_ch)                          # channel projection
        self.attention = SE(out_ch,reduction=16)                               # [D][E]
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        # Shortcut branch
        self.need_proj = (stride != 1 or in_ch != out_ch)
        if self.need_proj:
            self.PwR         = conv1x1(in_ch, out_ch, stride=stride)
            self.shortcut_se = SE(out_ch, reduction=16)   # attention trên shortcut

    def forward(self, x):
        # Main
        out = self.Pw1(x)
        out = self.TOP(out)        # [A][B][C]
        out = self.Pw2(out)
        out = self.attention(out)  # [D][E]
        out = self.drop_path(out)

        # Shortcut
        if self.need_proj:
            shortcut = self.shortcut_se(self.PwR(x))
        else:
            shortcut = x

        return out + shortcut


# =========================================================
# TickNetv6 — Backbone Tick-shape
# =========================================================
class TickNetv8se(nn.Module):
    """TickNetv6: Tick-shape backbone với FR-PDP block (5 cải tiến).

    Tick-shape channels: thu hẹp → mở rộng → thu hẹp → mở rộng
    (khác NetTOP backbone tuyến tính 64→512)
    """
    def __init__(self, num_classes, init_conv_ch, init_conv_stride,
                 channels, strides, in_ch=3, in_size=(224, 224),
                 use_data_bn=True, drop_path_max=0.05, dropout=0.10):
        super().__init__()
        self.in_size = in_size

        # Linear DropPath schedule: block 0 ≈ 0, block cuối = drop_path_max
        total_blocks = sum(len(s) for s in channels)
        dpr = [i / max(total_blocks - 1, 1) * drop_path_max
               for i in range(total_blocks)]

        self.backbone = nn.Sequential()
        if use_data_bn:
            self.backbone.add_module("data_bn", nn.BatchNorm2d(in_ch))
        self.backbone.add_module("init_conv",
            conv3x3(in_ch, init_conv_ch, stride=init_conv_stride))

        cur_ch  = init_conv_ch
        blk_idx = 0
        for sid, stage_ch in enumerate(channels):
            stage = nn.Sequential()
            for uid, uch in enumerate(stage_ch):
                s = strides[sid] if uid == 0 else 1
                stage.add_module(f"unit{uid + 1}",
                    FR_PDP_block(cur_ch, uch, s,
                                 drop_path_rate=dpr[blk_idx]))
                cur_ch   = uch
                blk_idx += 1
            self.backbone.add_module(f"stage{sid + 1}", stage)

        self.final_ch = 1024
        self.backbone.add_module("final_conv",
            conv1x1(cur_ch, self.final_ch, act="relu"))
        self.backbone.add_module("global_pool", nn.AdaptiveAvgPool2d(1))

        self.dropout    = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = Classifier(self.final_ch, num_classes)
        self._init_params()

    def _init_params(self):
        for m in self.backbone.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.classifier.init_params()

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        return self.classifier(x)


# =========================================================
# Builder functions
# =========================================================
def build_TickNetv8(num_classes, typesize="small", cifar=False,
                    drop_path_max=0.05, dropout=0.10):
    """
    typesize: "basic" | "small" | "large"
    cifar   : True → in_size=(32,32), False → in_size=(224,224)

    Tick-shape channel configs (cùng với NetTOP gốc để so sánh công bằng):
      basic : [128] [64] [128] [256] [512]             — 5 blocks
      small : [128] [64,128] [256,512,128] [64,128,256] [512]  — 10 blocks
      large : [128] [64,128] [256,512,128,64,128,256] [512,128,64,128,256] [512]
    """
    init_ch = 32

    if typesize == "basic":
        channels = [[128], [64], [128], [256], [512]]
    elif typesize == "small":
        channels = [[128], [64, 128], [256, 512, 128], [64, 128, 256], [512]]
    elif typesize == "large":
        channels = [[128], [64, 128],
                    [256, 512, 128, 64, 128, 256],
                    [512, 128, 64, 128, 256],
                    [512]]
    else:
        raise ValueError(f"typesize không hợp lệ: '{typesize}'. "
                         f"Chọn: 'basic', 'small', 'large'")

    if cifar:
        in_size, init_s = (32, 32), 1
        strides = [1, 1, 2, 2, 2]
    else:
        in_size, init_s = (224, 224), 2
        if typesize == "basic":
            strides = [1, 2, 2, 2, 2]
        else:
            strides = [2, 1, 2, 2, 2]

    return TickNetv8se(
        num_classes      = num_classes,
        init_conv_ch     = init_ch,
        init_conv_stride = init_s,
        channels         = channels,
        strides          = strides,
        in_size          = in_size,
        drop_path_max    = drop_path_max,
        dropout          = dropout,
    )


# =========================================================
# Quick self-test
# =========================================================
if __name__ == "__main__":
    import sys

    configs = [
        ("basic",  False, (1, 3, 224, 224), 120,  "ImageNet/Dogs basic"),
        ("small",  False, (1, 3, 224, 224), 120,  "ImageNet/Dogs small"),
        ("large",  False, (1, 3, 224, 224), 1000, "ImageNet large"),
        ("small",  True,  (1, 3,  32,  32), 10,   "CIFAR-10 small"),
        ("large",  True,  (1, 3,  32,  32), 100,  "CIFAR-100 large"),
    ]

    all_ok = True
    print("=" * 65)
    print(f"{'Config':<30} {'Params':>10}  {'Output':<12}  Status")
    print("=" * 65)

    for typesize, cifar, shape, nc, label in configs:
        try:
            model = build_TickNetv8(nc, typesize=typesize, cifar=cifar)
            model.eval()
            with torch.no_grad():
                x   = torch.randn(*shape)
                out = model(x)
            params = sum(p.numel() for p in model.parameters()) / 1e6
            assert out.shape == (shape[0], nc), f"shape sai: {out.shape}"
            print(f"  {label:<28} {params:>8.3f}M  {str(out.shape):<12}  OK")
        except Exception as e:
            print(f"  {label:<28} {'':>10}  {'':12}  FAIL: {e}")
            all_ok = False

    print("=" * 65)
    if all_ok:
        print("Tất cả tests PASSED — TickNetv6 sẵn sàng train.")
    else:
        print("Có lỗi — kiểm tra lại.")
        sys.exit(1)
