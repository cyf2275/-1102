from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvINReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)), inplace=True)
        y = self.norm2(self.conv2(y))
        return F.relu(y + self.skip(x), inplace=True)


class ResUNetFPP(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=48, dropout_rate=0.0):
        super().__init__()
        c = int(base_channels)
        self.enc1 = ResidualBlock(in_channels, c)
        self.enc2 = ResidualBlock(c, c * 2)
        self.enc3 = ResidualBlock(c * 2, c * 4)
        self.enc4 = ResidualBlock(c * 4, c * 8)
        self.mid = ResidualBlock(c * 8, c * 16)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = ResidualBlock(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = ResidualBlock(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = ResidualBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = ResidualBlock(c * 2, c)
        self.out = nn.Conv2d(c, out_channels, 1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        m = self.dropout(self.mid(F.max_pool2d(e4, 2)))
        d4 = self.up4(m)
        if d4.shape[-2:] != e4.shape[-2:]:
            d4 = F.interpolate(d4, size=e4.shape[-2:], mode="bilinear", align_corners=False)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        if d3.shape[-2:] != e3.shape[-2:]:
            d3 = F.interpolate(d3, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(self.dec1(torch.cat([d1, e1], dim=1)))


class AttentionGate(nn.Module):
    def __init__(self, skip_channels: int, gate_channels: int, inter_channels: int):
        super().__init__()
        self.theta = nn.Conv2d(skip_channels, inter_channels, 1, bias=False)
        self.phi = nn.Conv2d(gate_channels, inter_channels, 1, bias=False)
        self.psi = nn.Conv2d(inter_channels, 1, 1)

    def forward(self, skip, gate):
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        att = torch.sigmoid(self.psi(F.relu(self.theta(skip) + self.phi(gate), inplace=True)))
        return skip * att


class AttentionUNetFPP(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=48, dropout_rate=0.0):
        super().__init__()
        c = int(base_channels)
        self.enc1 = ConvINReLU(in_channels, c)
        self.enc2 = ConvINReLU(c, c * 2)
        self.enc3 = ConvINReLU(c * 2, c * 4)
        self.enc4 = ConvINReLU(c * 4, c * 8)
        self.mid = ConvINReLU(c * 8, c * 16)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.att4 = AttentionGate(c * 8, c * 8, c * 4)
        self.dec4 = ConvINReLU(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.att3 = AttentionGate(c * 4, c * 4, c * 2)
        self.dec3 = ConvINReLU(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.att2 = AttentionGate(c * 2, c * 2, c)
        self.dec2 = ConvINReLU(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.att1 = AttentionGate(c, c, max(8, c // 2))
        self.dec1 = ConvINReLU(c * 2, c)
        self.out = nn.Conv2d(c, out_channels, 1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        m = self.dropout(self.mid(F.max_pool2d(e4, 2)))
        d4 = self.up4(m)
        s4 = self.att4(e4, d4)
        d4 = F.interpolate(d4, size=s4.shape[-2:], mode="bilinear", align_corners=False) if d4.shape[-2:] != s4.shape[-2:] else d4
        d4 = self.dec4(torch.cat([d4, s4], dim=1))
        d3 = self.up3(d4)
        s3 = self.att3(e3, d3)
        d3 = F.interpolate(d3, size=s3.shape[-2:], mode="bilinear", align_corners=False) if d3.shape[-2:] != s3.shape[-2:] else d3
        d3 = self.dec3(torch.cat([d3, s3], dim=1))
        d2 = self.up2(d3)
        s2 = self.att2(e2, d2)
        d2 = F.interpolate(d2, size=s2.shape[-2:], mode="bilinear", align_corners=False) if d2.shape[-2:] != s2.shape[-2:] else d2
        d2 = self.dec2(torch.cat([d2, s2], dim=1))
        d1 = self.up1(d2)
        s1 = self.att1(e1, d1)
        d1 = F.interpolate(d1, size=s1.shape[-2:], mode="bilinear", align_corners=False) if d1.shape[-2:] != s1.shape[-2:] else d1
        return self.out(self.dec1(torch.cat([d1, s1], dim=1)))


class NestedUNetFPP(nn.Module):
    """Compact UNet++ style baseline with deep nested skip refinement."""

    def __init__(self, in_channels=1, out_channels=1, base_channels=32, dropout_rate=0.0):
        super().__init__()
        c = int(base_channels)
        nb = [c, c * 2, c * 4, c * 8, c * 16]
        self.conv00 = ConvINReLU(in_channels, nb[0])
        self.conv10 = ConvINReLU(nb[0], nb[1])
        self.conv20 = ConvINReLU(nb[1], nb[2])
        self.conv30 = ConvINReLU(nb[2], nb[3])
        self.conv40 = ConvINReLU(nb[3], nb[4])
        self.dropout = nn.Dropout2d(dropout_rate)
        self.conv01 = ConvINReLU(nb[0] + nb[1], nb[0])
        self.conv11 = ConvINReLU(nb[1] + nb[2], nb[1])
        self.conv21 = ConvINReLU(nb[2] + nb[3], nb[2])
        self.conv31 = ConvINReLU(nb[3] + nb[4], nb[3])
        self.conv02 = ConvINReLU(nb[0] * 2 + nb[1], nb[0])
        self.conv12 = ConvINReLU(nb[1] * 2 + nb[2], nb[1])
        self.conv22 = ConvINReLU(nb[2] * 2 + nb[3], nb[2])
        self.conv03 = ConvINReLU(nb[0] * 3 + nb[1], nb[0])
        self.conv13 = ConvINReLU(nb[1] * 3 + nb[2], nb[1])
        self.conv04 = ConvINReLU(nb[0] * 4 + nb[1], nb[0])
        self.out = nn.Conv2d(nb[0], out_channels, 1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        x00 = self.conv00(x)
        x10 = self.conv10(F.max_pool2d(x00, 2))
        x20 = self.conv20(F.max_pool2d(x10, 2))
        x30 = self.conv30(F.max_pool2d(x20, 2))
        x40 = self.dropout(self.conv40(F.max_pool2d(x30, 2)))

        x01 = self.conv01(torch.cat([x00, self.up(x10, x00)], dim=1))
        x11 = self.conv11(torch.cat([x10, self.up(x20, x10)], dim=1))
        x21 = self.conv21(torch.cat([x20, self.up(x30, x20)], dim=1))
        x31 = self.conv31(torch.cat([x30, self.up(x40, x30)], dim=1))

        x02 = self.conv02(torch.cat([x00, x01, self.up(x11, x00)], dim=1))
        x12 = self.conv12(torch.cat([x10, x11, self.up(x21, x10)], dim=1))
        x22 = self.conv22(torch.cat([x20, x21, self.up(x31, x20)], dim=1))

        x03 = self.conv03(torch.cat([x00, x01, x02, self.up(x12, x00)], dim=1))
        x13 = self.conv13(torch.cat([x10, x11, x12, self.up(x22, x10)], dim=1))

        x04 = self.conv04(torch.cat([x00, x01, x02, x03, self.up(x13, x00)], dim=1))
        return self.out(x04)


class MPSBranch0(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.net(x)


class MPSBranchN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depth: int):
        super().__init__()
        layers = []
        ch = in_channels
        for i in range(depth):
            layers.append(nn.Conv2d(ch, out_channels, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            if i + 1 < depth:
                layers.append(nn.LeakyReLU(inplace=True))
            ch = out_channels
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MPSResidualInceptionBlock(nn.Module):
    """Residual-inception block used by MPS_XNet, scaled for 960px training."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        q = max(1, out_channels // 4)
        self.branch0 = MPSBranch0(in_channels, out_channels)
        self.branch1 = MPSBranchN(in_channels, q, 1)
        self.branch2 = MPSBranchN(in_channels, q, 2)
        self.branch3 = MPSBranchN(in_channels, q, 3)
        self.branch4 = MPSBranchN(in_channels, out_channels - 3 * q, 4)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        shortcut = self.branch0(x)
        multi = torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)],
            dim=1,
        )
        return self.act(shortcut + multi)


class MPSDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.res = MPSResidualInceptionBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.res(x)
        return self.pool(skip), skip


class MPSUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.res = MPSResidualInceptionBlock(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.res(torch.cat([x, skip], dim=1))


class MPSMiniUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=8):
        super().__init__()
        c = int(base_channels)
        self.down1 = MPSDownBlock(in_channels, c)
        self.down2 = MPSDownBlock(c, c * 2)
        self.down3 = MPSDownBlock(c * 2, c * 4)
        self.down4 = MPSDownBlock(c * 4, c * 8)
        self.mid = MPSResidualInceptionBlock(c * 8, c * 16)
        self.up4 = MPSUpBlock(c * 16, c * 8)
        self.up3 = MPSUpBlock(c * 8, c * 4)
        self.up2 = MPSUpBlock(c * 4, c * 2)
        self.up1 = MPSUpBlock(c * 2, c)
        self.out = nn.Conv2d(c, out_channels, 1)

    def forward(self, x):
        x1, s1 = self.down1(x)
        x2, s2 = self.down2(x1)
        x3, s3 = self.down3(x2)
        x4, s4 = self.down4(x3)
        x5 = self.mid(x4)
        x = self.up4(x5, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        return self.out(x)


class MPSXNetDepthFPP(nn.Module):
    """MPS_XNet-style physical multi-task network for single-frame depth.

    The original MPS_XNet predicts numerator/denominator, wrapped phase, and
    unwrapped phase. For FPP-ML-Bench we keep the same staged physical design,
    but train the final branch to output normalized depth so it can be compared
    under the same metric as all other single-frame depth methods.
    """

    def __init__(self, in_channels=1, out_channels=1, base_channels=8, dropout_rate=0.0):
        super().__init__()
        self.fenzi = MPSMiniUNet(in_channels, 1, base_channels)
        self.fenmu = MPSMiniUNet(in_channels, 1, base_channels)
        self.wrapped = MPSMiniUNet(in_channels + 2, 1, base_channels)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.depth = MPSMiniUNet(in_channels + 3, out_channels, base_channels)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        fenzi = self.fenzi(x)
        fenmu = self.fenmu(x)
        wrapped = self.wrapped(torch.cat([x, fenzi, fenmu], dim=1))
        depth = self.depth(self.dropout(torch.cat([x, fenzi, fenmu, wrapped], dim=1)))
        return {"fenzi": fenzi, "fenmu": fenmu, "wrapped": wrapped, "depth": depth}


class MPSXNetPhaseFPP(nn.Module):
    """MPS_XNet-style phase-first baseline for FPP-ML-Bench.

    This variant keeps the original MPS_XNet task structure closer than the
    depth-output adaptation: numerator, denominator, wrapped phase, and
    unwrapped phase are all predicted from a single fringe. The caller can map
    the unwrapped phase to depth with a fixed phase-to-depth proxy for fair
    benchmark evaluation.
    """

    def __init__(self, in_channels=1, out_channels=1, base_channels=8, dropout_rate=0.0):
        super().__init__()
        self.fenzi = MPSMiniUNet(in_channels, 1, base_channels)
        self.fenmu = MPSMiniUNet(in_channels, 1, base_channels)
        self.wrapped = MPSMiniUNet(in_channels + 2, 1, base_channels)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.unwrapped = MPSMiniUNet(in_channels + 3, out_channels, base_channels)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        fenzi = torch.tanh(self.fenzi(x))
        fenmu = torch.tanh(self.fenmu(x))
        wrapped = torch.sigmoid(self.wrapped(torch.cat([x, fenzi, fenmu], dim=1)))
        unwrapped = torch.sigmoid(
            self.unwrapped(self.dropout(torch.cat([x, fenzi, fenmu, wrapped], dim=1)))
        )
        return {"fenzi": fenzi, "fenmu": fenmu, "wrapped": wrapped, "unwrapped": unwrapped}


class Pix2PixDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=not normalize)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Pix2PixUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.net(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)


class Pix2PixGeneratorFPP(nn.Module):
    """U-Net generator used by pix2pix-style conditional GAN baselines."""

    def __init__(self, in_channels=1, out_channels=1, base_channels=64, dropout_rate=0.0):
        super().__init__()
        c = int(base_channels)
        self.d1 = Pix2PixDown(in_channels, c, normalize=False)
        self.d2 = Pix2PixDown(c, c * 2)
        self.d3 = Pix2PixDown(c * 2, c * 4)
        self.d4 = Pix2PixDown(c * 4, c * 8)
        self.d5 = Pix2PixDown(c * 8, c * 8)
        self.bottleneck = Pix2PixDown(c * 8, c * 8, normalize=False)
        self.u5 = Pix2PixUp(c * 8, c * 8, dropout_rate)
        self.u4 = Pix2PixUp(c * 16, c * 8, dropout_rate)
        self.u3 = Pix2PixUp(c * 16, c * 4)
        self.u2 = Pix2PixUp(c * 8, c * 2)
        self.u1 = Pix2PixUp(c * 4, c)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(c * 2, out_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        b = self.bottleneck(d5)
        x = self.u5(b, d5)
        x = self.u4(x, d4)
        x = self.u3(x, d3)
        x = self.u2(x, d2)
        x = self.u1(x, d1)
        return self.final(x)


class PatchGANDiscriminatorFPP(nn.Module):
    def __init__(self, in_channels=2, base_channels=64):
        super().__init__()
        c = int(base_channels)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c * 2, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(c * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 2, c * 4, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(c * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 4, c * 8, 4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(c * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 8, 1, 4, stride=1, padding=1),
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, fringe, depth):
        return self.net(torch.cat([fringe, depth], dim=1))


def build_single_frame_baseline(arch: str, in_channels=1, out_channels=1, base_channels=None, dropout_rate=0.0):
    arch = str(arch).lower()
    if arch == "resunet":
        return ResUNetFPP(in_channels, out_channels, base_channels or 48, dropout_rate)
    if arch in {"attention_unet", "attunet"}:
        return AttentionUNetFPP(in_channels, out_channels, base_channels or 48, dropout_rate)
    if arch in {"nested_unet", "unetpp", "unet++"}:
        return NestedUNetFPP(in_channels, out_channels, base_channels or 32, dropout_rate)
    if arch in {"mps_xnet", "mpsxnet", "mps_xnet_depth"}:
        return MPSXNetDepthFPP(in_channels, out_channels, base_channels or 8, dropout_rate)
    if arch in {"mps_xnet_phase", "mpsxnet_phase", "mps_xnet_phase_proxy"}:
        return MPSXNetPhaseFPP(in_channels, out_channels, base_channels or 8, dropout_rate)
    if arch in {"pix2pix", "pix2pix_generator"}:
        return Pix2PixGeneratorFPP(in_channels, out_channels, base_channels or 64, dropout_rate)
    raise ValueError(f"unknown single-frame baseline arch: {arch}")
