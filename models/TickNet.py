"""TickNet_BEST.py — final architecture matching the PAA manuscript.

   [A]  Multi-scale 1D kernels (k=3 + k=5) on the XZ and YZ planes.
    [B]  Multi-scale 2D kernels (3x3 + 5x5) on the XY plane.
    [C]  Per-channel learnable additive fusion  sigmoid(alpha . f_xz + beta . f_yz),
         replacing the multiplicative sigmoid(f_xz * f_yz) of NetTOP.
    [D]  Pre-sigmoid BatchNorm on the fusion argument alpha . f_xz + beta . f_yz,
         which keeps the gate in its unsaturated regime and therefore preserves
         gradient flow (Proposition 3 of the manuscript).

Design rationale
----------------
TickNet (TickNet.py) was our winning baseline because of:
    (i)  A shallow 7-block wide-first tick-shape backbone with sharp channel
         jumps 32 -> 128 -> 64 -> 128 -> 256 -> 128 -> 64 -> 512.
    (ii) An extra high-resolution perceptron at 112x112 that preserves spatial
         detail before the first down-sampling.

Both features are kept verbatim here. The MAF attention module has been
dropped; we now use the standard SE gate on the main branch and a LightSE
gate on every projected shortcut (matching NetTOP/TickNets conventions and
the experimental protocol described in the paper).


Usage
-----
    from TickNet_BEST import build_TickNet_BEST
    model = build_TickNet_BEST(num_classes=120, cifar=False)

All 2^4 = 16 ablation configurations are reachable through the flags
use_ms_1d / use_ms_2d / use_add_fusion / use_bn_fusion.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. Basic building blocks
# =============================================================================
class DropPath(nn.Module):
    """Stochastic depth - drops the residual branch with probability p."""

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(mask_shape).bernoulli_(keep)
        return x * mask.div_(keep)


class ConvBlock(nn.Module):
    """Generic Conv -> BN -> Activation wrapper."""

    def __init__(self, in_ch, out_ch, k, s, p,
                 groups=1, bias=False, use_bn=True, act="relu"):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else None
        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


def conv1x1(in_ch, out_ch, stride=1, groups=1, use_bn=True, act="relu"):
    return ConvBlock(in_ch, out_ch, k=1, s=stride, p=0,
                     groups=groups, bias=False, use_bn=use_bn, act=act)


def conv3x3(in_ch, out_ch, stride=1, groups=1, use_bn=True, act="relu"):
    return ConvBlock(in_ch, out_ch, k=3, s=stride, p=1,
                     groups=groups, bias=False, use_bn=use_bn, act=act)


# =============================================================================
# 2. Squeeze-and-excitation (main branch + shortcut)
# =============================================================================
class SE(nn.Module):
    """Standard squeeze-and-excitation gate (Hu et al. 2018).

    Used on the main branch of every FR-PDP block. Reduction ratio r = 16.
    """

    def __init__(self, ch: int, r: int = 16) -> None:
        super().__init__()
        mid = max(ch // r, 4)
        self.fc1 = nn.Linear(ch, mid)
        self.fc2 = nn.Linear(mid, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        s = F.adaptive_avg_pool2d(x, 1).view(b, c)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s.view(b, c, 1, 1)


class LightSE(nn.Module):
    """Light squeeze-and-excitation gate for projected shortcuts."""

    def __init__(self, ch: int, r: int = 16) -> None:
        super().__init__()
        mid = max(ch // r, 4)
        self.fc1 = nn.Linear(ch, mid)
        self.fc2 = nn.Linear(mid, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        s = F.adaptive_avg_pool2d(x, 1).view(b, c)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s.view(b, c, 1, 1)


# =============================================================================
# 3. MS-TOP - the four improvements of the paper
# =============================================================================
class MS_TOP(nn.Module):
    """Multi-scale three-orthogonal-plane extractor with pre-fusion BatchNorm.

    Flag map
        use_ms_1d      - flag [A]  multi-scale Conv1d on XZ / YZ.
        use_ms_2d      - flag [B]  multi-scale DW Conv2d on XY.
        use_add_fusion - flag [C]  learnable additive fusion alpha/beta.
        use_bn_fusion  - flag [D]  BatchNorm on the pre-sigmoid fusion argument.
                                   Requires flag [C] to be active.

    Forward path:

        1. XY branch.   DW Conv2d k=3 (+ k=5 if [B]) -> BN -> ReLU
        2. XZ branch.   reshape -> DW Conv1d k=3 (+ k=5 if [A]) -> reshape
                        -> BN -> optional AvgPool -> ReLU
        3. YZ branch.   symmetric to XZ.
        4. Fusion.
               [C] off, [D] off:  g = sigmoid( f_xz * f_yz )          (NetTOP)
               [C] on,  [D] off:  g = sigmoid( alpha*f_xz + beta*f_yz )
               [C] on,  [D] on :  g = sigmoid( BN( alpha*f_xz + beta*f_yz ) )
        5. ReLU( f_xy * g ).

    Flag [D] without flag [C] is not meaningful (would BN a product) and is
    silently disabled.
    """

    def __init__(
        self,
        ch: int,
        stride: int = 1,
        use_ms_1d: bool = True,
        use_ms_2d: bool = True,
        use_add_fusion: bool = True,
        use_bn_fusion: bool = True,
    ) -> None:
        super().__init__()
        self.ch = ch
        self.stride = stride
        self.use_ms_1d = use_ms_1d
        self.use_ms_2d = use_ms_2d
        self.use_add_fusion = use_add_fusion
        # Flag [D] requires [C].
        self.use_bn_fusion = use_bn_fusion and use_add_fusion

        # ---- [B] XY branch: DW Conv2d k=3, optional + k=5 ----
        self.dw_xy_3 = nn.Conv2d(ch, ch, 3, stride, 1, groups=ch, bias=False)
        if use_ms_2d:
            self.dw_xy_5 = nn.Conv2d(ch, ch, 5, stride, 2, groups=ch, bias=False)
        else:
            self.dw_xy_5 = None
        self.bn_xy = nn.BatchNorm2d(ch)

        # ---- [A] XZ branch: DW Conv1d k=3 (+ k=5) along width axis ----
        self.dw_xz_3 = nn.Conv1d(ch, ch, 3, 1, 1, groups=ch, bias=False)
        if use_ms_1d:
            self.dw_xz_5 = nn.Conv1d(ch, ch, 5, 1, 2, groups=ch, bias=False)
        else:
            self.dw_xz_5 = None
        self.bn_xz = nn.BatchNorm2d(ch)

        # ---- [A] YZ branch: DW Conv1d k=3 (+ k=5) along height axis ----
        self.dw_yz_3 = nn.Conv1d(ch, ch, 3, 1, 1, groups=ch, bias=False)
        if use_ms_1d:
            self.dw_yz_5 = nn.Conv1d(ch, ch, 5, 1, 2, groups=ch, bias=False)
        else:
            self.dw_yz_5 = None
        self.bn_yz = nn.BatchNorm2d(ch)

        # Stride-alignment pooling for XZ / YZ when the block down-samples.
        self.pool = (
            nn.AvgPool2d(kernel_size=stride, stride=stride)
            if stride > 1 else nn.Identity()
        )

        # ---- [C] Learnable per-channel fusion weights ----
        if self.use_add_fusion:
            self.alpha = nn.Parameter(torch.full((1, ch, 1, 1), 0.5))
            self.beta = nn.Parameter(torch.full((1, ch, 1, 1), 0.5))
        else:
            self.alpha = None
            self.beta = None

        # ---- [D] Pre-sigmoid BatchNorm on the fusion argument ----
        if self.use_bn_fusion:
            self.bn_fusion = nn.BatchNorm2d(ch)
        else:
            self.bn_fusion = None

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()

        # ----- [B] XY branch -----
        xy = self.dw_xy_3(x)
        if self.dw_xy_5 is not None:
            xy = xy + self.dw_xy_5(x)
        f_xy = F.relu(self.bn_xy(xy), inplace=True)

        # ----- [A] XZ branch -----
        # (B, C, H, W) -> (B, H, C, W) -> (B*H, C, W) for Conv1d along W.
        t_xz = x.permute(0, 2, 1, 3).contiguous().view(b * h, c, w)
        xz = self.dw_xz_3(t_xz)
        if self.dw_xz_5 is not None:
            xz = xz + self.dw_xz_5(t_xz)
        # back to (B, C, H, W) for BN2d and eventual pooling.
        xz = xz.view(b, h, c, w).permute(0, 2, 1, 3).contiguous()
        f_xz = F.relu(self.pool(self.bn_xz(xz)), inplace=True)

        # ----- [A] YZ branch -----
        t_yz = x.permute(0, 3, 1, 2).contiguous().view(b * w, c, h)
        yz = self.dw_yz_3(t_yz)
        if self.dw_yz_5 is not None:
            yz = yz + self.dw_yz_5(t_yz)
        yz = yz.view(b, w, c, h).permute(0, 2, 3, 1).contiguous()
        f_yz = F.relu(self.pool(self.bn_yz(yz)), inplace=True)

        # ----- Fusion (flags [C] and [D]) -----
        if self.use_add_fusion:
            # g = sigmoid(  alpha * f_xz  +  beta * f_yz )
            arg = self.alpha * f_xz + self.beta * f_yz
            # Flag [D]: standardise the argument before the sigmoid.
            if self.bn_fusion is not None:
                arg = self.bn_fusion(arg)
            gate = torch.sigmoid(arg)
        else:
            # NetTOP-style multiplicative gate.
            gate = torch.sigmoid(f_xz * f_yz)

        return F.relu(f_xy * gate, inplace=True)


# =============================================================================
# 4. FR-PDP block (point-depth-point with full-residual shortcut)
# =============================================================================
class FR_PDP_block(nn.Module):
    """Full-residual point-depth-point block.

    Main branch:    Pw_1 (no BN, no act) -> MS-TOP -> Pw_2 (BN, ReLU) -> SE -> DropPath
    Shortcut:       identity                if (stride == 1 and C_in == C_out),
                    LightSE(Pw_R)           otherwise.
    Output:         main + shortcut.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        drop_path_rate: float = 0.0,
        use_ms_1d: bool = True,
        use_ms_2d: bool = True,
        use_add_fusion: bool = True,
        use_bn_fusion: bool = True,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.in_ch = in_ch
        self.out_ch = out_ch

        # Main path
        self.Pw1 = conv1x1(in_ch, in_ch, use_bn=False, act=None)
        self.TOP = MS_TOP(
            ch=in_ch,
            stride=stride,
            use_ms_1d=use_ms_1d,
            use_ms_2d=use_ms_2d,
            use_add_fusion=use_add_fusion,
            use_bn_fusion=use_bn_fusion,
        )
        self.Pw2 = conv1x1(in_ch, out_ch)
        self.SE = SE(out_ch, r=16)
        self.drop_path = (
            DropPath(drop_path_rate)
            if drop_path_rate > 0 else nn.Identity()
        )

        # Shortcut path
        self.need_proj = (stride != 1 or in_ch != out_ch)
        if self.need_proj:
            self.PwR = conv1x1(in_ch, out_ch, stride=stride)
            self.shortcut_se = SE(out_ch, r=16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Main
        out = self.Pw1(x)
        out = self.TOP(out)
        out = self.Pw2(out)
        out = self.SE(out)
        out = self.drop_path(out)

        # Shortcut
        if self.need_proj:
            shortcut = self.shortcut_se(self.PwR(x))
        else:
            shortcut = x

        return out + shortcut


# =============================================================================
# 5. TickNet-BEST backbone
# =============================================================================
class TickNet_BEST(nn.Module):
    """Full-residual point-depth-point network with MS-TOP + pre-fusion BN.

    Architecture (for a 3x224x224 input, matches Table 1 of the manuscript):

        Stem 3x3 stride 2      : 3   -> 32        at 112x112
        Extra perceptron       : 32  -> 32        at 112x112
        Backbone B1 stride 2   : 32  -> 128       at 56x56
        Backbone B2 stride 1   : 128 -> 64        at 56x56
        Backbone B3 stride 1   : 64  -> 128       at 56x56
        Backbone B4 stride 2   : 128 -> 256       at 28x28
        Backbone B5 stride 1   : 256 -> 128       at 28x28
        Backbone B6 stride 1   : 128 -> 64        at 28x28
        Backbone B7 stride 2   : 64  -> 512       at 14x14
        Head 1x1               : 512 -> 1024      at 14x14
        Global avg pool + FC.

    The flags A / B / C / D can be individually disabled for ablation.
    """

    def __init__(
        self,
        num_classes: int,
        cifar: bool = False,
        drop_path_max: float = 0.05,
        dropout: float = 0.2,
        use_ms_1d: bool = True,
        use_ms_2d: bool = True,
        use_add_fusion: bool = True,
        use_bn_fusion: bool = True,
    ) -> None:
        super().__init__()
        init_stride = 1 if cifar else 2

        self._flag_kwargs = dict(
            use_ms_1d=use_ms_1d,
            use_ms_2d=use_ms_2d,
            use_add_fusion=use_add_fusion,
            use_bn_fusion=use_bn_fusion,
        )

        # Stem
        self.initial = conv3x3(3, 32, stride=init_stride)

        # Extra high-resolution perceptron (identity shortcut, 32 -> 32).
        self.extra_perceptron = FR_PDP_block(
            32, 32, stride=1, drop_path_rate=0.0, **self._flag_kwargs
        )

        # Seven-block tick-shape backbone.
        backbone_specs = [
            # (in_ch, out_ch, stride)
            (32, 128, 2),   # B1 - expand
            (128, 64, 1),   # B2 - compress
            (64, 128, 1),   # B3 - restore
            (128, 256, 2),  # B4 - expand
            (256, 128, 1),  # B5 - compress
            (128, 64, 1),   # B6 - compress further
            (64, 512, 2),   # B7 - final expand
        ]
        total = len(backbone_specs)
        dpr = [i / max(total - 1, 1) * drop_path_max for i in range(total)]

        blocks = []
        for (in_c, out_c, s), dp in zip(backbone_specs, dpr):
            blocks.append(
                FR_PDP_block(in_c, out_c, s, drop_path_rate=dp, **self._flag_kwargs)
            )
        self.backbone = nn.Sequential(*blocks)

        # Head
        self.final_conv = conv1x1(512, 1024, act="relu")
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(1024, num_classes)

        self._init_params()

    def _init_params(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        x = self.extra_perceptron(x)
        x = self.backbone(x)
        x = self.final_conv(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


# =============================================================================
# 6. Builder
# =============================================================================
def build_TickNet_BEST(
    num_classes: int,
    cifar: bool = False,
    drop_path_max: float = 0.05,
    dropout: float = 0.2,
    use_ms_1d: bool = True,
    use_ms_2d: bool = True,
    use_add_fusion: bool = True,
    use_bn_fusion: bool = True,
) -> TickNet_BEST:
    """Build TickNet-BEST with configurable ablation flags.

    Args:
        num_classes    : number of output classes.
        cifar          : if True, stem stride = 1 (32x32 input);
                         if False, stem stride = 2 (224x224 input).
        drop_path_max  : maximum stochastic-depth rate at the deepest block;
                         set to 0 to disable DropPath entirely.
        dropout        : dropout on the penultimate feature vector.
        use_ms_1d      : flag [A]  multi-scale on XZ / YZ branches.
        use_ms_2d      : flag [B]  multi-scale on XY branch.
        use_add_fusion : flag [C]  learnable additive fusion.
        use_bn_fusion  : flag [D]  pre-sigmoid BatchNorm (requires [C]).

    Returns:
        A TickNet_BEST nn.Module.
    """
    return TickNet_BEST(
        num_classes=num_classes,
        cifar=cifar,
        drop_path_max=drop_path_max,
        dropout=dropout,
        use_ms_1d=use_ms_1d,
        use_ms_2d=use_ms_2d,
        use_add_fusion=use_add_fusion,
        use_bn_fusion=use_bn_fusion,
    )


# =============================================================================
# 7. Self-test + ablation helper
# =============================================================================
if __name__ == "__main__":
    # Forward-shape smoke test on four configurations.
    configs = [
        # (name,                  cifar, input_shape,        num_classes)
        ("Stanford Dogs 224x224", False, (1, 3, 224, 224),   120),
        ("CIFAR-100   32x32",     True,  (1, 3,  32,  32),   100),
        ("CIFAR-10    32x32",     True,  (1, 3,  32,  32),    10),
        ("ImageNet    224x224",   False, (1, 3, 224, 224),  1000),
    ]

    print("=" * 74)
    print(f"{'Config':<28} {'# params':>10}  {'Output':<18}  Status")
    print("=" * 74)
    for name, cifar, shape, num_classes in configs:
        try:
            model = build_TickNet_BEST(num_classes=num_classes, cifar=cifar)
            model.eval()
            with torch.no_grad():
                x = torch.randn(*shape)
                y = model(x)
            params = sum(p.numel() for p in model.parameters()) / 1e6
            assert y.shape == (shape[0], num_classes)
            print(f"  {name:<26} {params:>8.3f} M  {str(y.shape):<18}  OK")
        except Exception as err:
            print(f"  {name:<26} {'':>10}  {'':18}  FAIL: {err}")
    print("=" * 74)

    # Ablation helper: parameter count for every flag combination.
    print("\nAblation - parameter count on Stanford Dogs (120 classes):\n")
    print(f"  {'flags':<10}  {'# params':>10}  {'diff vs baseline':>18}")
    print("  " + "-" * 44)
    flag_combos = [
        ("----", False, False, False, False),  # baseline
        ("A---", True,  False, False, False),
        ("-B--", False, True,  False, False),
        ("--C-", False, False, True,  False),
        ("--CD", False, False, True,  True),
        ("AB--", True,  True,  False, False),
        ("ABC-", True,  True,  True,  False),
        ("ABCD", True,  True,  True,  True),   # full
    ]
    baseline_params = None
    for label, a, b, c, d in flag_combos:
        m = build_TickNet_BEST(
            num_classes=120, cifar=False,
            use_ms_1d=a, use_ms_2d=b, use_add_fusion=c, use_bn_fusion=d,
        )
        p = sum(x.numel() for x in m.parameters()) / 1e6
        if baseline_params is None:
            baseline_params = p
            diff = ""
        else:
            diff = f"+{(p - baseline_params) * 1000:.2f} K"
        print(f"  {label:<10}  {p:>8.3f} M  {diff:>18}")
    print()

    # Component breakdown of the full model on Dogs.
    model = build_TickNet_BEST(120, cifar=False)
    total = sum(x.numel() for x in model.parameters())
    categories = {
        "Final 1x1 conv (head)":     0,
        "Pointwise 1x1 (Pw_2)":      0,
        "Pointwise 1x1 (Pw_R)":      0,
        "Pointwise 1x1 (Pw_1)":      0,
        "Classifier":                0,
        "SE main branch":            0,
        "LightSE shortcut":          0,
        "MS-TOP depthwise (A+B)":    0,
        "Fusion gains (alpha,beta)": 0,
        "Fusion BN (gamma,delta)":   0,
        "Stem 3x3":                  0,
        "Other BN / bias":           0,
    }
    for name, p in model.named_parameters():
        n = p.numel()
        low = name.lower()
        if "top.alpha" in low or "top.beta" in low:
            categories["Fusion gains (alpha,beta)"] += n
        elif "bn_fusion" in low:
            categories["Fusion BN (gamma,delta)"] += n
        elif "top" in low:
            categories["MS-TOP depthwise (A+B)"] += n
        elif ".se." in low:
            categories["SE main branch"] += n
        elif "shortcut_se" in low:
            categories["LightSE shortcut"] += n
        elif "pw2" in low:
            categories["Pointwise 1x1 (Pw_2)"] += n
        elif "pwr" in low:
            categories["Pointwise 1x1 (Pw_R)"] += n
        elif "pw1" in low:
            categories["Pointwise 1x1 (Pw_1)"] += n
        elif "initial" in low:
            categories["Stem 3x3"] += n
        elif "final_conv" in low:
            categories["Final 1x1 conv (head)"] += n
        elif "classifier" in low:
            categories["Classifier"] += n
        else:
            categories["Other BN / bias"] += n

    print(f"Parameter breakdown for Dogs model ({total/1e6:.3f} M total):\n")
    print(f"  {'Component':<30}  {'# params':>10}   share")
    print("  " + "-" * 54)
    for k, v in sorted(categories.items(), key=lambda kv: -kv[1]):
        if v > 0:
            print(f"  {k:<30}  {v:>10,}   {100 * v / total:>5.2f} %")
    print("=" * 74)
