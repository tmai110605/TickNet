"""TickNet_BESTv.py — TickNet-BEST with plane-operator ablation + group rate g.

Plane operators:
  'xy'         : F_xy only   — DW Conv2d frontal plane
  'xz'         : F_xz only   — DW Conv1d lateral W axis
  'yz'         : F_yz only   — DW Conv1d lateral H axis
  'top_prime'  : F'_TOP      — ReLU(F_xy * sigmoid(F_xz))
  'top_dprime' : F''_TOP     — ReLU(F_xy * sigmoid(F_yz))
  'top'        : MS-TOP full (flags [A][B][C][D])

Group rate g  (mirrors NetTOP Table 3):
  g=1  — all pointwise Pw2/PwR use groups=1  (default)
  g=2  — Pw2 and PwR use groups=2
  g=4  — Pw2 and PwR use groups=4
  NOTE: g must divide all channel sizes in the backbone.

MS-TOP flags (active only when operator='top'):
  use_ms_1d    — [A] multi-scale Conv1d k=3+5 on XZ/YZ
  use_ms_2d    — [B] multi-scale DW Conv2d k=3+5 on XY
  use_add_fusion — [C] learnable per-channel additive fusion alpha/beta
  use_bn_fusion  — [D] pre-sigmoid BN (requires [C])

Usage:
    from TickNet_BESTv import build_TickNet_BESTv
    model = build_TickNet_BESTv(num_classes=120, cifar=False, operator='top', g=2)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

OPERATORS = ['xy', 'xz', 'yz', 'top_prime', 'top_dprime', 'top']

# =============================================================================
# 1. Basic building blocks
# =============================================================================
class DropPath(nn.Module):
    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask.div_(keep)


class ConvBlock(nn.Module):
    """Conv -> BN -> Activation."""
    def __init__(self, in_ch, out_ch, k, s, p,
                 groups=1, bias=False, use_bn=True, act='relu'):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=bias)
        self.bn   = nn.BatchNorm2d(out_ch) if use_bn else None
        self.act  = nn.ReLU(inplace=True) if act == 'relu' else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn  is not None: x = self.bn(x)
        if self.act is not None: x = self.act(x)
        return x


def conv1x1(in_ch, out_ch, stride=1, groups=1, use_bn=True, act='relu'):
    return ConvBlock(in_ch, out_ch, k=1, s=stride, p=0,
                     groups=groups, bias=False, use_bn=use_bn, act=act)

def conv3x3(in_ch, out_ch, stride=1, groups=1, use_bn=True, act='relu'):
    return ConvBlock(in_ch, out_ch, k=3, s=stride, p=1,
                     groups=groups, bias=False, use_bn=use_bn, act=act)


# =============================================================================
# 2. Squeeze-and-Excitation
# =============================================================================
class SE(nn.Module):
    def __init__(self, ch: int, r: int = 16) -> None:
        super().__init__()
        mid = max(ch // r, 4)
        self.fc1 = nn.Linear(ch, mid)
        self.fc2 = nn.Linear(mid, ch)

    def forward(self, x):
        b, c, _, _ = x.size()
        s = F.adaptive_avg_pool2d(x, 1).view(b, c)
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s), inplace=True)))
        return x * s.view(b, c, 1, 1)


# =============================================================================
# 3. Plane feature extractors
# =============================================================================
class _XY(nn.Module):
    """Frontal plane: DW Conv2d k=3 (+ k=5 if [B]) -> BN -> ReLU."""
    def __init__(self, ch, stride, use_ms_2d):
        super().__init__()
        self.dw3 = nn.Conv2d(ch, ch, 3, stride, 1, groups=ch, bias=False)
        self.dw5 = nn.Conv2d(ch, ch, 5, stride, 2, groups=ch, bias=False) if use_ms_2d else None
        self.bn  = nn.BatchNorm2d(ch)

    def forward(self, x):
        out = self.dw3(x)
        if self.dw5 is not None:
            out = out + self.dw5(x)
        return F.relu(self.bn(out), inplace=True)


class _XZ(nn.Module):
    """Lateral XZ plane: DW Conv1d along W axis, k=3 (+ k=5 if [A])."""
    def __init__(self, ch, stride, use_ms_1d):
        super().__init__()
        self.dw3  = nn.Conv1d(ch, ch, 3, 1, 1, groups=ch, bias=False)
        self.dw5  = nn.Conv1d(ch, ch, 5, 1, 2, groups=ch, bias=False) if use_ms_1d else None
        self.bn   = nn.BatchNorm2d(ch)
        self.pool = nn.AvgPool2d(stride, stride) if stride > 1 else nn.Identity()

    def forward(self, x):
        b, c, h, w = x.size()
        t = x.permute(0, 2, 1, 3).contiguous().view(b * h, c, w)
        out = self.dw3(t)
        if self.dw5 is not None:
            out = out + self.dw5(t)
        out = out.view(b, h, c, w).permute(0, 2, 1, 3).contiguous()
        return F.relu(self.pool(self.bn(out)), inplace=True)


class _YZ(nn.Module):
    """Lateral YZ plane: DW Conv1d along H axis, k=3 (+ k=5 if [A])."""
    def __init__(self, ch, stride, use_ms_1d):
        super().__init__()
        self.dw3  = nn.Conv1d(ch, ch, 3, 1, 1, groups=ch, bias=False)
        self.dw5  = nn.Conv1d(ch, ch, 5, 1, 2, groups=ch, bias=False) if use_ms_1d else None
        self.bn   = nn.BatchNorm2d(ch)
        self.pool = nn.AvgPool2d(stride, stride) if stride > 1 else nn.Identity()

    def forward(self, x):
        b, c, h, w = x.size()
        t = x.permute(0, 3, 1, 2).contiguous().view(b * w, c, h)
        out = self.dw3(t)
        if self.dw5 is not None:
            out = out + self.dw5(t)
        out = out.view(b, w, c, h).permute(0, 2, 3, 1).contiguous()
        return F.relu(self.pool(self.bn(out)), inplace=True)


# =============================================================================
# 4. MS-TOP — generalised operator (6 plane modes)
# =============================================================================
class MS_TOP(nn.Module):
    """Multi-scale Three-Orthogonal-Plane operator.

    Fusion for operator='top':
      [C] off: g = sigmoid(F_xz * F_yz)
      [C] on : g = sigmoid(alpha*F_xz + beta*F_yz)
      [C]+[D]: g = sigmoid(BN(alpha*F_xz + beta*F_yz))
    """

    def __init__(self, ch, stride=1, operator='top',
                 use_ms_1d=True, use_ms_2d=True,
                 use_add_fusion=True, use_bn_fusion=True):
        super().__init__()
        assert operator in OPERATORS
        self.operator = operator

        need_xy = operator in ('xy', 'top_prime', 'top_dprime', 'top')
        need_xz = operator in ('xz', 'top_prime', 'top')
        need_yz = operator in ('yz', 'top_dprime', 'top')

        self.xy = _XY(ch, stride, use_ms_2d) if need_xy else None
        self.xz = _XZ(ch, stride, use_ms_1d) if need_xz else None
        self.yz = _YZ(ch, stride, use_ms_1d) if need_yz else None

        self.use_add_fusion = use_add_fusion and (operator == 'top')
        self.use_bn_fusion  = use_bn_fusion  and self.use_add_fusion

        if self.use_add_fusion:
            self.alpha = nn.Parameter(torch.full((1, ch, 1, 1), 0.5))
            self.beta  = nn.Parameter(torch.full((1, ch, 1, 1), 0.5))
        else:
            self.alpha = self.beta = None

        self.bn_fusion = nn.BatchNorm2d(ch) if self.use_bn_fusion else None

    def forward(self, x):
        op = self.operator

        if op == 'xy':  return self.xy(x)
        if op == 'xz':  return self.xz(x)
        if op == 'yz':  return self.yz(x)

        f_xy = self.xy(x)
        if op == 'top_prime':
            return F.relu(f_xy * torch.sigmoid(self.xz(x)), inplace=True)
        if op == 'top_dprime':
            return F.relu(f_xy * torch.sigmoid(self.yz(x)), inplace=True)

        # full TOP / MS-TOP
        f_xz = self.xz(x)
        f_yz = self.yz(x)
        if self.use_add_fusion:
            arg = self.alpha * f_xz + self.beta * f_yz
            if self.bn_fusion is not None:
                arg = self.bn_fusion(arg)
            gate = torch.sigmoid(arg)
        else:
            gate = torch.sigmoid(f_xz * f_yz)
        return F.relu(f_xy * gate, inplace=True)


# =============================================================================
# 5. FR-PDP block — group rate g applied to Pw2 and PwR
# =============================================================================
class FR_PDP_block(nn.Module):
    """Full-residual Point-Depth-Point block.

    Main: Pw1(g=1, no BN/act) → MS-TOP → Pw2(g=g) → SE → DropPath
    Skip: identity  OR  LightSE(PwR(g=g, stride))
    """

    def __init__(self, in_ch, out_ch, stride,
                 drop_path_rate=0.0, g=1,
                 operator='top',
                 use_ms_1d=True, use_ms_2d=True,
                 use_add_fusion=True, use_bn_fusion=True):
        super().__init__()

        # Pw1: always groups=1 (channel-mixing before depthwise)
        self.Pw1 = conv1x1(in_ch, in_ch, groups=1, use_bn=False, act=None)

        self.TOP = MS_TOP(ch=in_ch, stride=stride, operator=operator,
                          use_ms_1d=use_ms_1d, use_ms_2d=use_ms_2d,
                          use_add_fusion=use_add_fusion, use_bn_fusion=use_bn_fusion)

        # Pw2: grouped by g  (reduces params like NetTOP)
        self.Pw2 = conv1x1(in_ch, out_ch, groups=g)
        self.SE  = SE(out_ch, r=16)
        self.dp  = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        # Shortcut
        self.need_proj = (stride != 1 or in_ch != out_ch)
        if self.need_proj:
            # PwR: also grouped by g
            self.PwR          = conv1x1(in_ch, out_ch, stride=stride, groups=g)
            self.shortcut_se  = SE(out_ch, r=16)

    def forward(self, x):
        out = self.Pw1(x)
        out = self.TOP(out)
        out = self.Pw2(out)
        out = self.SE(out)
        out = self.dp(out)

        shortcut = self.shortcut_se(self.PwR(x)) if self.need_proj else x
        return out + shortcut


# =============================================================================
# 6. TickNet-BESTv backbone
# =============================================================================
class TickNet_BESTvg(nn.Module):
    """TickNet-BEST with group rate g and plane-operator ablation.

    Tick-shape channel schedule: 3→32→128→64→128→256→128→64→512→1024
    g=1/2/4 controls Pw2 & PwR grouped convolutions (mirrors NetTOP Table 3).
    """

    def __init__(self, num_classes, cifar=False, g=1,
                 operator='top', drop_path_max=0.05, dropout=0.2,
                 use_ms_1d=True, use_ms_2d=True,
                 use_add_fusion=True, use_bn_fusion=True):
        super().__init__()

        assert g in (1, 2, 4), f"g must be 1, 2 or 4, got {g}"
        flag_kw = dict(g=g, operator=operator,
                       use_ms_1d=use_ms_1d, use_ms_2d=use_ms_2d,
                       use_add_fusion=use_add_fusion, use_bn_fusion=use_bn_fusion)

        init_stride = 1 if cifar else 2
        self.initial = conv3x3(3, 32, stride=init_stride)
        self.extra_perceptron = FR_PDP_block(32, 32, stride=1,
                                             drop_path_rate=0.0, **flag_kw)

        specs = [
            (32, 128, 2),   # B1
            (128, 64, 1),   # B2
            (64, 128, 1),   # B3
            (128, 256, 2),  # B4
            (256, 128, 1),  # B5
            (128, 64, 1),   # B6
            (64, 512, 2),   # B7
        ]
        n   = len(specs)
        dpr = [i / max(n - 1, 1) * drop_path_max for i in range(n)]
        self.backbone = nn.Sequential(*[
            FR_PDP_block(ic, oc, s, drop_path_rate=dp, **flag_kw)
            for (ic, oc, s), dp in zip(specs, dpr)
        ])

        self.final_conv = conv1x1(512, 1024)
        self.avgpool    = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten    = nn.Flatten()
        self.dropout    = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(1024, num_classes)
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.initial(x)
        x = self.extra_perceptron(x)
        x = self.backbone(x)
        x = self.final_conv(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.classifier(x)


# =============================================================================
# 7. Builder
# =============================================================================
def build_TickNet_BESTvg(
    num_classes: int,
    cifar: bool = False,
    g: int = 1,
    operator: str = 'top',
    drop_path_max: float = 0.05,
    dropout: float = 0.2,
    use_ms_1d: bool = True,
    use_ms_2d: bool = True,
    use_add_fusion: bool = True,
    use_bn_fusion: bool = True,
) -> TickNet_BESTvg:
    """Build TickNet-BESTv.

    Args:
        num_classes    : output classes.
        cifar          : True → stem stride=1 (32×32 input).
        g              : group rate for Pw2 & PwR  (1 | 2 | 4).
        operator       : plane operator — see OPERATORS list.
        drop_path_max  : max stochastic-depth rate (0 = disabled).
        dropout        : dropout before classifier.
        use_ms_1d      : [A] multi-scale Conv1d on XZ/YZ.
        use_ms_2d      : [B] multi-scale DW Conv2d on XY.
        use_add_fusion : [C] learnable additive fusion alpha/beta.
        use_bn_fusion  : [D] pre-sigmoid BN (requires [C]).
    """
    return TickNet_BESTvg(
        num_classes=num_classes, cifar=cifar, g=g,
        operator=operator, drop_path_max=drop_path_max, dropout=dropout,
        use_ms_1d=use_ms_1d, use_ms_2d=use_ms_2d,
        use_add_fusion=use_add_fusion, use_bn_fusion=use_bn_fusion,
    )


# =============================================================================
# 8. Self-test
# =============================================================================
if __name__ == '__main__':
    import sys

    # --- shape smoke-test ---
    print('=' * 72)
    print(f"  {'g':<4} {'Operator':<14} {'Dataset':<14} {'# params':>10}  Status")
    print('=' * 72)
    for g in [1, 2, 4]:
        for op in OPERATORS:
            try:
                m = build_TickNet_BESTvg(120, cifar=False, g=g, operator=op)
                m.eval()
                import torch
                with torch.no_grad():
                    y = m(torch.randn(1, 3, 224, 224))
                p = sum(x.numel() for x in m.parameters()) / 1e6
                assert y.shape == (1, 120)
                print(f"  g={g}  {op:<14} {'Dogs 224':<14} {p:>8.3f} M  OK")
            except Exception as e:
                print(f"  g={g}  {op:<14} {'Dogs 224':<14} {'':>10}  FAIL: {e}")
    print('=' * 72)

    # --- param count table g=1/2/4 vs operator ---
    print('\nParams (M) per g × operator on Stanford Dogs:\n')
    print(f"  {'operator':<14}", end='')
    for g in [1, 2, 4]:
        print(f"  {'g='+str(g):>10}", end='')
    print()
    print('  ' + '-' * 46)
    import torch
    for op in OPERATORS:
        row = f"  {op:<14}"
        for g in [1, 2, 4]:
            m = build_TickNet_BESTvg(120, cifar=False, g=g, operator=op)
            p = sum(x.numel() for x in m.parameters()) / 1e6
            row += f"  {p:>10.3f}"
        print(row)
    print()
