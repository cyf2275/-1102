import torch
import torch.nn as nn


class OfficialDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class OfficialDownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = OfficialDoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        return skip, self.pool(skip)


class OfficialUpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = OfficialDoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class OfficialUNetFPP(nn.Module):
    """UNet architecture matching the public FPP-ML-Bench UNet implementation."""

    def __init__(self, in_channels=1, out_channels=1, dropout_rate=0.0):
        super().__init__()
        self.down1 = OfficialDownSample(in_channels, 64)
        self.down2 = OfficialDownSample(64, 128)
        self.down3 = OfficialDownSample(128, 256)
        self.down4 = OfficialDownSample(256, 512)
        self.bottleneck = OfficialDoubleConv(512, 1024)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.up1 = OfficialUpSample(1024, 512)
        self.up2 = OfficialUpSample(512, 256)
        self.up3 = OfficialUpSample(256, 128)
        self.up4 = OfficialUpSample(128, 64)
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        skip3, x = self.down3(x)
        skip4, x = self.down4(x)
        x = self.dropout(self.bottleneck(x))
        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)
        return self.out(x)


class ZeroCondAdapter(nn.Module):
    """Lightweight zero-initialized 1x1 adapter for physics conditions."""

    def __init__(self, cond_channels, out_channels, hidden_channels=32):
        super().__init__()
        hidden_channels = max(8, min(int(hidden_channels), max(8, out_channels // 4)))
        self.net = nn.Sequential(
            nn.Conv2d(cond_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, cond, size):
        if cond.shape[-2:] != size:
            cond = torch.nn.functional.interpolate(cond, size=size, mode="bilinear", align_corners=False)
        return self.net(cond)


class OfficialUNetFPPAdapter(nn.Module):
    """Official UNet backbone with zero-initialized physics adapters.

    The raw fringe still enters the normal UNet path. Physics features are added
    as residual feature biases at each scale, so the model can start exactly
    from a fringe-only checkpoint when the adapters are zero.
    """

    def __init__(self, cond_channels, out_channels=1, dropout_rate=0.0, adapter_hidden=32):
        super().__init__()
        self.backbone = OfficialUNetFPP(in_channels=1, out_channels=out_channels, dropout_rate=dropout_rate)
        self.adapter1 = ZeroCondAdapter(cond_channels, 64, adapter_hidden)
        self.adapter2 = ZeroCondAdapter(cond_channels, 128, adapter_hidden)
        self.adapter3 = ZeroCondAdapter(cond_channels, 256, adapter_hidden)
        self.adapter4 = ZeroCondAdapter(cond_channels, 512, adapter_hidden)
        self.adapter_mid = ZeroCondAdapter(cond_channels, 1024, adapter_hidden)

    def load_backbone_state_dict(self, state_dict, strict=True):
        return self.backbone.load_state_dict(state_dict, strict=strict)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad_(False)

    def forward(self, fringe, cond):
        skip1 = self.backbone.down1.conv(fringe)
        skip1 = skip1 + self.adapter1(cond, skip1.shape[-2:])
        x = self.backbone.down1.pool(skip1)

        skip2 = self.backbone.down2.conv(x)
        skip2 = skip2 + self.adapter2(cond, skip2.shape[-2:])
        x = self.backbone.down2.pool(skip2)

        skip3 = self.backbone.down3.conv(x)
        skip3 = skip3 + self.adapter3(cond, skip3.shape[-2:])
        x = self.backbone.down3.pool(skip3)

        skip4 = self.backbone.down4.conv(x)
        skip4 = skip4 + self.adapter4(cond, skip4.shape[-2:])
        x = self.backbone.down4.pool(skip4)

        x = self.backbone.dropout(self.backbone.bottleneck(x))
        x = x + self.adapter_mid(cond, x.shape[-2:])
        x = self.backbone.up1(x, skip4)
        x = self.backbone.up2(x, skip3)
        x = self.backbone.up3(x, skip2)
        x = self.backbone.up4(x, skip1)
        return self.backbone.out(x)
