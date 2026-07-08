# MANIQA `forward()` 논문 Figure 2 기준 해설

이 문서는 `models/maniqa.py`를 읽으면서 MANIQA 논문 Figure 2의 모듈이 실제 코드에서 어디에 해당하는지 이해하기 위한 설명입니다.

## 1. 모델 정의 위치

핵심 파일은 다음입니다.

```text
models/maniqa.py
```

이 파일에는 세 가지 핵심 클래스가 있습니다.

```text
TABlock    : 논문의 TAB, channel dimension attention
SaveOutput : ViT 중간 layer feature를 저장하는 hook helper
MANIQA     : 전체 모델
```

SSTB 관련 Swin Transformer 구현은 다음 파일에 있습니다.

```text
models/swin.py
```

## 2. `MANIQA.__init__()`에서 만들어지는 모듈

`MANIQA.__init__()`를 논문 Figure 2 순서로 보면 다음과 같습니다.

| 논문 모듈 | 코드 |
| --- | --- |
| ViT backbone | `self.vit = timm.create_model('vit_base_patch8_224', ...)` |
| ViT 중간 feature 저장 | `layer.register_forward_hook(self.save_output)` |
| TAB stage1 | `self.tablock1` |
| channel 축소 | `self.conv1` |
| SSTB stage1 | `self.swintransformer1` |
| TAB stage2 | `self.tablock2` |
| channel 축소 | `self.conv2` |
| SSTB stage2 | `self.swintransformer2` |
| score branch | `self.fc_score` |
| weight branch | `self.fc_weight` |

## 3. 입력 shape

기본 설정은 다음입니다.

```python
img_size = 224
patch_size = 8
input_size = img_size // patch_size
```

따라서:

```text
input_size = 28
patch count = 28 * 28 = 784
```

모델 입력은 다음 shape이어야 합니다.

```text
x: (B, 3, 224, 224)
```

## 4. ViT가 하는 일

ViT는 이미지를 patch token으로 바꿉니다.

MANIQA에서 사용하는 ViT는 patch size가 8입니다.

```text
224x224 image
-> 8x8 patch
-> 28x28 patches
-> 784 patch tokens
```

각 patch token은 768차원 feature입니다.

```text
ViT patch feature: (B, 784, 768)
```

## 5. Hook으로 ViT 중간 layer feature 저장

MANIQA는 ViT의 마지막 결과만 쓰지 않습니다.

ViT block마다 hook을 걸어 중간 출력을 저장합니다.

```python
for layer in self.vit.modules():
    if isinstance(layer, Block):
        handle = layer.register_forward_hook(self.save_output)
```

이후 `self.vit(x)`를 실행하면 `self.save_output.outputs`에 각 block 출력이 쌓입니다.

## 6. `extract_feature()`

`extract_feature()`는 저장된 ViT block 출력 중 네 개를 선택합니다.

```python
x6 = save_output.outputs[6][:, 1:]
x7 = save_output.outputs[7][:, 1:]
x8 = save_output.outputs[8][:, 1:]
x9 = save_output.outputs[9][:, 1:]
```

여기서 `[:, 1:]`는 cls token을 제거한다는 뜻입니다.

선택된 feature는 모두 다음 shape입니다.

```text
x6: (B, 784, 768)
x7: (B, 784, 768)
x8: (B, 784, 768)
x9: (B, 784, 768)
```

그다음 feature dimension 방향으로 concatenate합니다.

```python
x = torch.cat((x6, x7, x8, x9), dim=2)
```

결과:

```text
x: (B, 784, 3072)
```

## 7. Stage 1 TAB

TAB는 Transposed Attention Block입니다.

핵심 아이디어는 attention을 patch token 축이 아니라 channel 축에서 수행하는 것입니다.

먼저 shape을 바꿉니다.

```python
x = rearrange(x, 'b (h w) c -> b c (h w)')
```

결과:

```text
(B, 784, 3072)
-> (B, 3072, 784)
```

이제 TAB 안에서 q, k, v가 만들어집니다.

```python
q = self.c_q(x)
k = self.c_k(x)
v = self.c_v(x)
```

그리고 attention weight가 계산됩니다.

```python
attn = q @ k.transpose(-2, -1)
```

stage1의 attention weight shape:

```text
(B, 3072, 3072)
```

이것은 channel끼리의 관계 행렬입니다.

논문에서 TAB가 global interaction을 강화한다고 설명하는 부분이 여기에 해당합니다.

## 8. Stage 1 SSTB

TAB 결과는 다시 2D feature map으로 돌아갑니다.

```python
x = rearrange(x, 'b c (h w) -> b c h w')
```

결과:

```text
(B, 3072, 784)
-> (B, 3072, 28, 28)
```

이후 `conv1`으로 channel을 줄입니다.

```python
x = self.conv1(x)
```

결과:

```text
(B, 3072, 28, 28)
-> (B, 768, 28, 28)
```

그 다음 SSTB에 해당하는 Swin Transformer가 적용됩니다.

```python
x = self.swintransformer1(x)
```

Swin Transformer는 window attention을 사용합니다.

```text
작은 window 안에서 patch끼리 attention
다음 block에서는 window를 shift
shifted window 덕분에 window 경계 밖 patch와도 정보 교환
```

이 부분이 논문에서 local interaction을 강화한다고 설명하는 SSTB입니다.

## 9. Stage 2 TAB + SSTB

Stage 2는 stage 1과 같은 구조를 한 번 더 적용합니다.

먼저 TAB를 위해 shape을 바꿉니다.

```python
x = rearrange(x, 'b c h w -> b c (h w)')
```

결과:

```text
(B, 768, 28, 28)
-> (B, 768, 784)
```

stage2 TAB attention weight:

```text
(B, 768, 768)
```

그 다음 `conv2`로 channel을 줄입니다.

```text
(B, 768, 28, 28)
-> (B, 384, 28, 28)
```

두 번째 SSTB를 통과한 결과도 다음 shape을 유지합니다.

```text
(B, 384, 28, 28)
```

## 10. Dual Branch

마지막 feature map을 patch별 feature로 바꿉니다.

```python
x = rearrange(x, 'b c h w -> b (h w) c')
```

결과:

```text
(B, 384, 28, 28)
-> (B, 784, 384)
```

이제 각 patch마다 384차원 feature가 있습니다.

MANIQA는 두 branch를 사용합니다.

```python
f = self.fc_score(x[i])
w = self.fc_weight(x[i])
```

각 branch 의미:

```text
f: patch별 quality score
w: patch별 importance weight
```

shape:

```text
f: (784, 1)
w: (784, 1)
```

## 11. Patch-weighted quality prediction

최종 score는 단순 평균이 아닙니다.

patch score를 patch weight로 가중 평균합니다.

```python
_s = torch.sum(f * w) / torch.sum(w)
```

직관적으로는 다음과 같습니다.

```text
이미지 전체 patch를 모두 같은 비중으로 보지 않는다.
품질 판단에 더 중요한 patch는 weight가 커진다.
중요도가 낮은 patch는 weight가 작아진다.
최종 score는 중요한 patch의 품질을 더 많이 반영한다.
```

이 부분이 논문 Figure 2의 dual branch structure for patch-weighted quality prediction입니다.

## 12. 전체 shape 요약

```text
input image
(B, 3, 224, 224)

ViT patch embedding
(B, 784, 768)

selected ViT layer features
4 x (B, 784, 768)

concatenate
(B, 784, 3072)

stage1 TAB input
(B, 3072, 784)

stage1 TAB attention
(B, 3072, 3072)

stage1 SSTB input/output
(B, 768, 28, 28)

stage2 TAB input
(B, 768, 784)

stage2 TAB attention
(B, 768, 768)

stage2 SSTB output
(B, 384, 28, 28)

patch features before dual branch
(B, 784, 384)

score branch
(784, 1)

weight branch
(784, 1)

final predicted score
(B,)
```

## 13. 한 문장으로 이해하기

MANIQA는 ViT로 이미지 patch의 표현을 뽑고, 여러 ViT layer를 합친 뒤, TAB로 channel 관점의 전역 관계를 강화하고, SSTB로 patch 위치 관점의 지역 관계를 강화한 다음, 각 patch의 품질 점수와 중요도를 따로 예측해서 최종 이미지 품질 점수를 만드는 모델입니다.

