import argparse
import csv
import os
import random

import cv2
import numpy as np
import torch
from tqdm import tqdm

from models.maniqa import MANIQA


def setup_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="Test MANIQA checkpoints with base_random or global_fixed5 crops.")
    parser.add_argument("--image_dir", required=True, help="Folder containing test images.")
    parser.add_argument("--ckpt_path", required=True, help="Path to a MANIQA state_dict checkpoint.")
    parser.add_argument("--output_csv", required=True, help="CSV path to save scores.")
    parser.add_argument("--crop_mode", choices=["base_random", "global_fixed5"], required=True)
    parser.add_argument("--crop_fusion", choices=["mean", "min"], default="mean")
    parser.add_argument("--local_weight", type=float, default=0.5)
    parser.add_argument("--num_repeats", type=int, default=30, help="Repeats for base_random stochastic crops.")
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20)
    return parser.parse_args()


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_to_tensor(image):
    image = image.astype("float32") / 255
    image = np.transpose(image, (2, 0, 1))
    image = (image - 0.5) / 0.5
    return torch.from_numpy(image).type(torch.FloatTensor)


def read_rgb_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def ensure_min_size(image, crop_size):
    h, w = image.shape[:2]
    if h >= crop_size and w >= crop_size:
        return image

    scale = crop_size / min(h, w)
    new_w = int(np.ceil(w * scale))
    new_h = int(np.ceil(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def random_crop(image, crop_size):
    image = ensure_min_size(image, crop_size)
    h, w = image.shape[:2]
    top = np.random.randint(0, h - crop_size + 1)
    left = np.random.randint(0, w - crop_size + 1)
    return image[top:top + crop_size, left:left + crop_size]


def fixed_local_crops(image, crop_size):
    image = ensure_min_size(image, crop_size)
    h, w = image.shape[:2]
    positions = [
        (0, 0),
        (0, w - crop_size),
        ((h - crop_size) // 2, (w - crop_size) // 2),
        (h - crop_size, 0),
        (h - crop_size, w - crop_size),
    ]
    return [image[top:top + crop_size, left:left + crop_size] for top, left in positions]


def list_images(image_dir):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = []
    for name in sorted(os.listdir(image_dir)):
        path = os.path.join(image_dir, name)
        if os.path.isfile(path) and os.path.splitext(name.lower())[1] in extensions:
            image_paths.append(path)
    return image_paths


def build_model(device):
    net = MANIQA(
        embed_dim=768,
        num_outputs=1,
        dim_mlp=768,
        patch_size=8,
        img_size=224,
        window_size=4,
        depths=[2, 2],
        num_heads=[4, 4],
        num_tab=2,
        scale=0.8,
    )
    return net.to(device)


def load_checkpoint(net, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if isinstance(state, dict):
        state = {key.replace("module.", "", 1): value for key, value in state.items()}
    net.load_state_dict(state, strict=False)


def predict_base_random(net, image, crop_size, num_repeats, device):
    scores = []
    for _ in range(num_repeats):
        crop = random_crop(image, crop_size)
        x = normalize_to_tensor(crop).unsqueeze(0).to(device)
        score = net(x).squeeze().item()
        scores.append(score)
    return scores


def predict_global_fixed5(net, image, crop_size, crop_fusion, local_weight, device):
    global_img = cv2.resize(image, (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)
    local_imgs = fixed_local_crops(image, crop_size)

    x_global = normalize_to_tensor(global_img).unsqueeze(0).to(device)
    x_local = torch.stack([normalize_to_tensor(img) for img in local_imgs]).to(device)

    pred_global = net(x_global).squeeze()
    pred_local = net(x_local).view(-1)

    if crop_fusion == "min":
        local_score = pred_local.min()
    else:
        local_score = pred_local.mean()

    final_score = (1 - local_weight) * pred_global + local_weight * local_score
    return [final_score.item()], pred_global.item(), [score.item() for score in pred_local]


def main():
    args = parse_args()
    setup_seed(args.seed)

    cpu_num = 1
    os.environ["OMP_NUM_THREADS"] = str(cpu_num)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
    os.environ["MKL_NUM_THREADS"] = str(cpu_num)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    image_paths = list_images(args.image_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {args.image_dir}")

    device = get_device()
    net = build_model(device)
    load_checkpoint(net, args.ckpt_path, device)
    net.eval()

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_name",
                "crop_mode",
                "score_mean",
                "score_std",
                "num_scores",
                "global_score",
                "local_scores",
            ],
        )
        writer.writeheader()

        with torch.no_grad():
            for image_path in tqdm(image_paths):
                image = read_rgb_image(image_path)
                image_name = os.path.basename(image_path)

                if args.crop_mode == "global_fixed5":
                    scores, global_score, local_scores = predict_global_fixed5(
                        net,
                        image,
                        args.crop_size,
                        args.crop_fusion,
                        args.local_weight,
                        device,
                    )
                    writer.writerow({
                        "image_name": image_name,
                        "crop_mode": args.crop_mode,
                        "score_mean": float(np.mean(scores)),
                        "score_std": float(np.std(scores)),
                        "num_scores": len(scores),
                        "global_score": global_score,
                        "local_scores": ";".join(str(score) for score in local_scores),
                    })
                else:
                    scores = predict_base_random(
                        net,
                        image,
                        args.crop_size,
                        args.num_repeats,
                        device,
                    )
                    writer.writerow({
                        "image_name": image_name,
                        "crop_mode": args.crop_mode,
                        "score_mean": float(np.mean(scores)),
                        "score_std": float(np.std(scores)),
                        "num_scores": len(scores),
                        "global_score": "",
                        "local_scores": "",
                    })

    print(f"Saved scores to: {args.output_csv}")


if __name__ == "__main__":
    main()
