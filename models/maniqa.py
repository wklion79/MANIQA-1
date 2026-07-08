import os
import sys

import torch
import torch.nn as nn
import timm

from timm.models.vision_transformer import Block
from torch import nn
from einops import rearrange

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.swin import SwinTransformer
else:
    from .swin import SwinTransformer


class TABlock(nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        self.norm_fact = dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x, debug=False, debug_prefix="TAB"):
        # 논문 Figure 2 / TAB: channel dimension attention으로 채널 사이의 전역 관계를 계산한다.
        _x = x
        B, C, N = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)

        if debug:
            print(f"[{debug_prefix}] input feature: {tuple(x.shape)}")
            print(f"[{debug_prefix}] q/k/v for channel attention: {tuple(q.shape)} / {tuple(k.shape)} / {tuple(v.shape)}")

        attn = q @ k.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        if debug:
            print(f"[{debug_prefix}] channel attention weight: {tuple(attn.shape)}")
        x = (attn @ v).transpose(1, 2).reshape(B, C, N)
        x = self.proj_drop(x)
        x = x + _x
        if debug:
            print(f"[{debug_prefix}] output feature after channel weighting + residual: {tuple(x.shape)}")
        return x


class SaveOutput:
    def __init__(self):
        self.outputs = []
    
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
    
    def clear(self):
        self.outputs = []


class MANIQA(nn.Module):
    def __init__(self, embed_dim=72, num_outputs=1, patch_size=8, drop=0.1, 
                    depths=[2, 2], window_size=4, dim_mlp=768, num_heads=[4, 4],
                    img_size=224, num_tab=2, scale=0.8, vit_pretrained=True, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.input_size = img_size // patch_size
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)
        
        # 논문 Figure 2 / ViT feature extraction: 입력 이미지를 patch embedding 후 ViT feature로 변환한다.
        self.vit = timm.create_model('vit_base_patch8_224', pretrained=vit_pretrained)
        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)

        self.tablock1 = nn.ModuleList()
        for i in range(num_tab):
            tab = TABlock(self.input_size ** 2)
            self.tablock1.append(tab)

        self.conv1 = nn.Conv2d(embed_dim * 4, embed_dim, 1, 1, 0)
        self.swintransformer1 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale
        )

        self.tablock2 = nn.ModuleList()
        for i in range(num_tab):
            tab = TABlock(self.input_size ** 2)
            self.tablock2.append(tab)

        self.conv2 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)
        self.swintransformer2 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim // 2,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale
        )
        
        self.fc_score = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.ReLU()
        )
        self.fc_weight = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.Sigmoid()
        )
    
    def _debug_shape(self, name, tensor):
        print(f"[MANIQA debug] {name}: {tuple(tensor.shape)}")

    def extract_feature(self, save_output, debug=False):
        # 논문 Figure 2 / ViT의 여러 layer feature 추출: ViT block 6, 7, 8, 9의 patch token을 사용한다.
        x6 = save_output.outputs[6][:, 1:]
        x7 = save_output.outputs[7][:, 1:]
        x8 = save_output.outputs[8][:, 1:]
        x9 = save_output.outputs[9][:, 1:]
        if debug:
            self._debug_shape("silected ViT layer 6 feature without cls token", x6)
            self._debug_shape("selected ViT layer 7 feature without cls token", x7)
            self._debug_shape("selected ViT layer 8 feature without cls token", x8)
            self._debug_shape("selected ViT layer 9 feature without cls token", x9)
        # 논문 Figure 2 / feature concatenate: 선택된 여러 layer feature를 channel 방향으로 붙인다.
        x = torch.cat((x6, x7, x8, x9), dim=2)
        if debug:
            self._debug_shape("concatenated ViT feature", x)
        return x

    def forward(self, x, debug=False):
        if debug:
            # 논문 Figure 2 / 입력 이미지 및 224x224 crop 또는 resize 결과.
            self._debug_shape("input image tensor", x)
            if hasattr(self.vit, "patch_embed"):
                patch_embed = self.vit.patch_embed(x)
                self._debug_shape("patch embedding output", patch_embed)

        # 논문 Figure 2 / ViT feature extraction: ViT forward output보다 hook으로 잡은 중간 layer feature를 사용한다.
        _x = self.vit(x)
        if debug:
            self._debug_shape("raw ViT model output (not used for quality score)", _x)
            print(f"[MANIQA debug] saved ViT block outputs: {len(self.save_output.outputs)}")
        x = self.extract_feature(self.save_output, debug=debug)
        self.save_output.outputs.clear()

        # stage 1
        # 논문 Figure 2 / TAB: feature를 (B, C, patch_count)로 바꿔 channel attention을 적용한다.
        x = rearrange(x, 'b (h w) c -> b c (h w)', h=self.input_size, w=self.input_size)
        if debug:
            self._debug_shape("stage1 TAB input", x)
        for i, tab in enumerate(self.tablock1):
            x = tab(x, debug=debug, debug_prefix=f"TAB stage1 block{i}")
        if debug:
            self._debug_shape("TAB output after stage1 blocks", x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv1(x)
        if debug:
            self._debug_shape("stage1 conv output before SSTB", x)
        # 논문 Figure 2 / SSTB: Swin window attention과 shifted window attention으로 patch 간 local interaction을 만든다.
        x = self.swintransformer1(x)
        if debug:
            self._debug_shape("SSTB output after stage1", x)

        # stage2
        # 논문 Figure 2 / TAB: 두 번째 channel attention stage.
        x = rearrange(x, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        if debug:
            self._debug_shape("stage2 TAB input", x)
        for i, tab in enumerate(self.tablock2):
            x = tab(x, debug=debug, debug_prefix=f"TAB stage2 block{i}")
        if debug:
            self._debug_shape("TAB output after stage2 blocks", x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv2(x)
        if debug:
            self._debug_shape("stage2 conv output before SSTB", x)
        # 논문 Figure 2 / SSTB: 두 번째 Swin 기반 spatial attention stage.
        x = self.swintransformer2(x)
        if debug:
            self._debug_shape("SSTB output after stage2", x)

        # 논문 Figure 2 / Dual Branch: 각 patch feature마다 score branch와 weight branch를 통과시킨다.
        x = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
        if debug:
            self._debug_shape("patch features before dual branch", x)
        score = torch.tensor([], device=x.device, dtype=x.dtype)
        for i in range(x.shape[0]):
            f = self.fc_score(x[i])
            w = self.fc_weight(x[i])
            if debug:
                self._debug_shape(f"score branch output for batch {i}", f)
                self._debug_shape(f"weight branch output for batch {i}", w)
            # 논문 Figure 2 / patch-weighted quality prediction: patch score를 patch weight로 가중 평균한다.
            _s = torch.sum(f * w) / torch.sum(w)
            if debug:
                self._debug_shape(f"final predicted score for batch {i}", _s.unsqueeze(0))
            score = torch.cat((score, _s.unsqueeze(0)), 0)
        return score
