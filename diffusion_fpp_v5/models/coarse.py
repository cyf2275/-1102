import torch
import torch.nn as nn


class CoarsePredictor(nn.Module):
    """Small U-Net used only as a fringe-derived condition and sampler start."""

    def __init__(self, in_channels=7, out_channels=1, base_ch=32):
        super().__init__()
        self.enc1 = self._block(in_channels, base_ch)
        self.down1 = nn.Conv2d(base_ch, base_ch, 3, stride=2, padding=1)
        self.enc2 = self._block(base_ch, base_ch * 2)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 2, 3, stride=2, padding=1)
        self.enc3 = self._block(base_ch * 2, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)
        self.dec2 = self._block(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1)
        self.dec1 = self._block(base_ch * 2, base_ch)
        self.out = nn.Conv2d(base_ch, out_channels, 3, padding=1)

    @staticmethod
    def _block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        d2 = self.up2(e3)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = torch.nn.functional.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = torch.nn.functional.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return torch.tanh(self.out(d1))

