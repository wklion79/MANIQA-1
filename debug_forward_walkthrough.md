# MANIQA 한 장 이미지 Forward 따라가기

이 문서는 `debug_forward.py`를 실행하면서 논문 Figure 2의 전체 흐름을 코드 기준으로 이해하기 위한 설명입니다.

지금 목표는 학습 성능 재현이 아니라, 한 장의 이미지가 MANIQA 모델 안에서 어떤 tensor shape으로 바뀌는지 확인하는 것입니다.

## 0. 먼저 큰 그림

MANIQA는 no-reference image quality assessment 모델입니다.

즉, 원본 이미지 없이 distorted image 한 장만 보고 품질 점수를 예측합니다.

논문 Figure 2의 흐름을 코드 기준으로 쓰면 다음과 같습니다.

```text
image
-> 224x224 input
-> ViT patch embedding
-> ViT block feature extraction
-> select layer 6, 7, 8, 9
-> concatenate features
-> TAB: channel-wise global interaction
-> SSTB: spatial local interaction
-> TAB again
-> SSTB again
-> score branch + weight branch
-> weighted average quality score
```

코드에서는 이 전체 과정이 `models/maniqa.py`의 `MANIQA.forward()`에 들어 있습니다.

## 1. 입력 이미지

`debug_forward.py`에서는 이미지를 읽은 뒤 `224x224`로 resize합니다.

```python
image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)
```

논문에서는 crop 또는 resize된 image patch를 모델에 넣습니다.

MANIQA의 기본 설정은 다음입니다.

```text
image shape: (B, 3, 224, 224)
patch size: 8
patch grid: 28 x 28
number of patches: 784
```

여기서 `B`는 batch size입니다. 한 장만 보면 보통 `B=1`입니다.

## 2. Patch embedding

ViT는 이미지를 바로 CNN처럼 처리하지 않고, 작은 patch들로 나눕니다.

MANIQA는 `vit_base_patch8_224`를 사용합니다.

```python
self.vit = timm.create_model('vit_base_patch8_224', pretrained=vit_pretrained)
```

`224x224` 이미지를 `8x8` patch로 나누면 한 변에 28개 patch가 생깁니다.

```text
224 / 8 = 28
28 x 28 = 784
```

그래서 patch embedding 후 shape은 다음처럼 나옵니다.

```text
(1, 784, 768)
```

뜻은 다음과 같습니다.

```text
1: 이미지 한 장
784: patch token 개수
768: 각 patch를 표현하는 feature dimension
```

## 3. ViT feature extraction

ViT는 여러 Transformer block을 지나면서 patch feature를 계속 업데이트합니다.

MANIQA는 ViT의 최종 출력 하나만 쓰지 않습니다. 대신 중간 layer feature를 여러 개 가져옵니다.

`models/maniqa.py`에서는 ViT 내부 `Block`마다 forward hook을 등록합니다.

```python
for layer in self.vit.modules():
    if isinstance(layer, Block):
        handle = layer.register_forward_hook(self.save_output)
```

이 hook 덕분에 `self.vit(x)`를 실행하면 각 ViT block의 출력이 `self.save_output.outputs`에 저장됩니다.

## 4. 선택된 layer feature

MANIQA는 ViT block 6, 7, 8, 9의 feature를 사용합니다.

```python
x6 = save_output.outputs[6][:, 1:]
x7 = save_output.outputs[7][:, 1:]
x8 = save_output.outputs[8][:, 1:]
x9 = save_output.outputs[9][:, 1:]
```

`[:, 1:]`는 첫 번째 token을 제외한다는 뜻입니다.

ViT에는 보통 cls token이 앞에 붙습니다. MANIQA는 이미지 품질을 patch별로 계산해야 하므로 cls token보다 patch token들이 중요합니다.

각 layer feature shape은 다음입니다.

```text
x6: (1, 784, 768)
x7: (1, 784, 768)
x8: (1, 784, 768)
x9: (1, 784, 768)
```

## 5. Feature concatenate

논문 Figure 2에서는 여러 ViT layer feature를 모아서 씁니다.

코드에서는 channel 방향, 즉 마지막 dimension으로 붙입니다.

```python
x = torch.cat((x6, x7, x8, x9), dim=2)
```

shape은 이렇게 바뀝니다.

```text
4개의 (1, 784, 768)
-> 1개의 (1, 784, 3072)
```

여기서 `3072 = 768 x 4`입니다.

이 단계의 의미는 간단합니다.

ViT의 한 layer만 보면 특정 깊이의 표현만 보게 됩니다. 여러 layer를 같이 쓰면 더 다양한 수준의 품질 단서를 모을 수 있습니다.

## 6. TAB: Transposed Attention Block

TAB는 논문에서 channel dimension attention을 담당합니다.

일반적인 Transformer attention은 token 간 관계를 봅니다. 그런데 MANIQA의 TAB는 feature channel 사이의 관계를 봅니다.

그래서 먼저 tensor를 바꿉니다.

```python
x = rearrange(x, 'b (h w) c -> b c (h w)')
```

shape 변화는 다음입니다.

```text
(1, 784, 3072)
-> (1, 3072, 784)
```

이제 `3072`개의 channel이 attention의 주인공이 됩니다.

TAB 내부에서는 q, k, v를 만들고 channel attention weight를 계산합니다.

```python
q = self.c_q(x)
k = self.c_k(x)
v = self.c_v(x)
attn = q @ k.transpose(-2, -1)
```

stage1 TAB attention weight shape은 다음입니다.

```text
(1, 3072, 3072)
```

뜻은 “3072개의 channel이 서로 얼마나 관련 있는지”를 보는 행렬입니다.

논문식으로 말하면 TAB는 이미지의 서로 다른 영역과 distortion 정보를 channel 축에서 전역적으로 섞어주는 역할을 합니다.

## 7. SSTB: Scale Swin Transformer Block

TAB가 channel 관계를 봤다면, SSTB는 patch의 공간적 관계를 봅니다.

TAB 이후 feature는 다시 2D grid로 바뀝니다.

```python
x = rearrange(x, 'b c (h w) -> b c h w', h=28, w=28)
```

그리고 `conv1`로 channel 수를 줄입니다.

```text
(1, 3072, 28, 28)
-> (1, 768, 28, 28)
```

그 다음 Swin Transformer가 적용됩니다.

Swin Transformer는 전체 patch를 한 번에 보지 않고 작은 window 단위로 attention을 합니다.

MANIQA 설정에서는 window size가 4입니다.

```text
window size: 4 x 4
one window contains: 16 patches
```

Swin block은 두 종류의 attention을 번갈아 씁니다.

```text
W-MSA: window 안에서 attention
SW-MSA: window를 shift해서 attention
```

이렇게 하면 가까운 patch끼리의 local interaction을 효율적으로 만들 수 있습니다.

논문에서 SSTB가 spatial dimension attention을 담당한다고 설명하는 부분이 이 단계입니다.

## 8. Stage 2

MANIQA는 TAB와 SSTB를 한 번만 쓰지 않고 두 stage로 사용합니다.

stage2 흐름은 다음입니다.

```text
stage1 output: (1, 768, 28, 28)
-> TAB input: (1, 768, 784)
-> TAB attention weight: (1, 768, 768)
-> conv2 output: (1, 384, 28, 28)
-> SSTB output: (1, 384, 28, 28)
```

stage1이 여러 ViT layer를 합친 큰 feature를 정리했다면, stage2는 줄어든 feature에서 다시 channel과 spatial interaction을 다듬는 과정으로 이해하면 됩니다.

## 9. Dual Branch

마지막 feature는 다시 patch별 feature로 바뀝니다.

```python
x = rearrange(x, 'b c h w -> b (h w) c')
```

shape은 다음입니다.

```text
(1, 384, 28, 28)
-> (1, 784, 384)
```

이제 784개 patch마다 384차원 feature가 있습니다.

MANIQA는 여기서 두 branch를 사용합니다.

```python
f = self.fc_score(x[i])
w = self.fc_weight(x[i])
```

각 branch의 의미는 다음입니다.

```text
score branch: 각 patch의 품질 점수
weight branch: 각 patch가 최종 품질에 얼마나 중요한지
```

shape은 다음입니다.

```text
score branch output: (784, 1)
weight branch output: (784, 1)
```

즉 patch 784개마다 score 하나, weight 하나가 나옵니다.

## 10. 최종 quality score

마지막으로 patch별 score를 weight로 가중 평균합니다.

```python
_s = torch.sum(f * w) / torch.sum(w)
```

직관적으로 말하면 다음과 같습니다.

```text
모든 patch를 똑같이 평균하지 않는다.
품질 판단에 중요한 patch는 더 크게 반영한다.
덜 중요한 patch는 작게 반영한다.
```

그래서 최종 출력은 이미지 한 장당 scalar score입니다.

```text
final predicted score: (1,)
```

## 11. 실행 명령어

MANIQA 폴더에서 실행하세요.

```powershell
cd C:\Users\BTREEE\work\MANIQA
.venv\Scripts\python.exe debug_forward.py --device cpu
```

이미지 없이 구조만 확인하려면 다음처럼 실행합니다.

```powershell
.venv\Scripts\python.exe debug_forward.py --random --device cpu
```

checkpoint가 있다면 다음처럼 넣을 수 있습니다.

```powershell
.venv\Scripts\python.exe debug_forward.py --device cpu --checkpoint ckpt_koniq10k.pt
```

## 12. 읽는 순서 추천

처음 볼 때는 아래 순서대로 보면 덜 헷갈립니다.

1. `debug_forward.py` 실행 결과에서 shape만 훑기
2. `models/maniqa.py`의 `forward()`를 위에서 아래로 읽기
3. `extract_feature()`에서 왜 6, 7, 8, 9번 layer를 쓰는지 확인하기
4. `TABlock.forward()`에서 channel attention weight shape 보기
5. `models/swin.py`의 `SwinBlock.forward()`에서 window attention 흐름 보기
6. 마지막 `fc_score`, `fc_weight`가 patch별로 어떻게 score를 만드는지 보기

