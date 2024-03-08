import torch
import torch.nn as nn
from tool import padding,unpadding
import torch.nn.functional as F
from fusion import Block_fusion

class TwinVitFusion(nn.Module):
    def __init__(self, encoder_rgb, encoder_tir, decoder_pixel):
        super(TwinVitFusion, self).__init__()
        self.encoder_rgb = encoder_rgb
        self.encoder_tir = encoder_tir
        self.decoder = decoder_pixel
        self.n_cls = encoder_rgb.n_cls
        self.patch_size = encoder_rgb.patch_size
        self.fusion = Block_fusion(encoder_rgb.d_model)

    def forward(self, im, im_tir):
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        im_tir = padding(im_tir, self.patch_size)
        H, W = im.size(2), im.size(3)

        x_rgb = self.encoder_rgb(im, return_features=True)
        x_tir = self.encoder_tir(im_tir, return_features=True)
        x = self.fusion(x_rgb, x_tir)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder_rgb.distilled
        #         x_rgb = x_rgb[:, num_extra_tokens:]
        #         class_rgb_decoder = x_rgb[:, -(self.n_cls + 2): -2]
        #         x_rgb = x_rgb[:, :-(self.n_cls + 2)]

        #         x_tir = x_tir[:, num_extra_tokens:]
        #         class_tir_decoder = x_tir[:, -2: ]
        #         x_tir = x_tir[:, :-(self.n_cls + 2)]

        #         masks_binary = self.decoder_binary(x_tir, (H, W), class_tir_decoder)
        #         masks_binary = F.interpolate(masks_binary, size=(H, W), mode="bilinear")
        #         masks_binary= unpadding(masks_binary, (H_ori, W_ori))
        #         masks_rgb = self.decoder(x_rgb, (H, W), class_rgb_decoder)
        #         masks_rgb = F.interpolate(masks_rgb, size=(H, W), mode="bilinear")
        #         masks_rgb= unpadding(masks_rgb, (H_ori, W_ori))
        x = x[:, num_extra_tokens:]
        class_for_decoder = x[:, -(self.n_cls + 2):-2]
        # class_for_binary = x[:,-(self.n_cls + 2):-(self.n_cls)]
        x = x[:, :-(self.n_cls + 2)]

        masks = self.decoder(x, (H, W), class_for_decoder)
        masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        masks = unpadding(masks, (H_ori, W_ori))

        return masks
