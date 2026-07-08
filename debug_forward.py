import argparse
import os
import random

import numpy as np
import torch

from models.maniqa import MANIQA


def setup_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_image_tensor(image_path, img_size, use_random=False):
    if use_random:
        # 논문 Figure 2 / 입력 이미지: 실제 이미지가 없을 때도 forward 구조 확인을 위해 random tensor를 사용한다.
        tensor = torch.rand(1, 3, img_size, img_size)
        print(f"[debug_forward] random input image tensor: {tuple(tensor.shape)}")
        return tensor

    import cv2

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    print(f"[debug_forward] original image HWC: {(h, w, 3)}")

    # 논문 Figure 2 / 224x224 crop 또는 resize: 디버그에서는 한 장을 안정적으로 통과시키기 위해 resize를 사용한다.
    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)
    image = image.astype("float32") / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = (image - 0.5) / 0.5
    tensor = torch.from_numpy(image).unsqueeze(0).float()
    print(f"[debug_forward] resized + normalized image tensor: {tuple(tensor.shape)}")
    return tensor


def load_checkpoint_if_requested(model, checkpoint_path, device):
    if not checkpoint_path:
        print("[debug_forward] checkpoint 없음: random initialized MANIQA head로 shape만 확인합니다.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint, strict=False)
    print(f"[debug_forward] checkpoint 로드 완료: {checkpoint_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="MANIQA Figure 2 forward shape debugger")
    parser.add_argument("--image", default="image/kunkun.png", help="디버그에 사용할 이미지 경로")
    parser.add_argument("--random", action="store_true", help="이미지 대신 random tensor로 forward 실행")
    parser.add_argument("--checkpoint", default="", help="선택: MANIQA checkpoint 경로")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="실행 장치")
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--pretrained-vit", action="store_true", help="ViT ImageNet pretrained weight 사용")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=768)
    parser.add_argument("--dim-mlp", type=int, default=768)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--scale", type=float, default=0.8)
    return parser.parse_args()


def main():
    args = parse_args()
    setup_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[debug_forward] device: {device}")

    x = load_image_tensor(args.image, args.img_size, use_random=args.random).to(device)

    # 논문 Figure 2 / 전체 MANIQA: ViT -> feature concatenate -> TAB -> SSTB -> Dual Branch 순서로 실행한다.
    model = MANIQA(
        embed_dim=args.embed_dim,
        num_outputs=1,
        dim_mlp=args.dim_mlp,
        patch_size=args.patch_size,
        img_size=args.img_size,
        window_size=args.window_size,
        depths=[2, 2],
        num_heads=[4, 4],
        num_tab=2,
        scale=args.scale,
        vit_pretrained=args.pretrained_vit,
    ).to(device)

    load_checkpoint_if_requested(model, args.checkpoint, device)
    model.eval()

    with torch.no_grad():
        score = model(x, debug=True)

    print(f"[debug_forward] final predicted score tensor: {score.detach().cpu().numpy()}")


if __name__ == "__main__":
    main()
