import os
import torch
import numpy as np
import logging
import time
import torch.nn as nn
import random
import argparse
import json
import csv

from torchvision import transforms
from torch.utils.data import DataLoader
from models.maniqa import MANIQA
from config import Config
from utils.process import RandCrop, ToTensor, Normalize, five_point_crop
from utils.process import split_dataset_kadid10k, split_dataset_koniq10k
from utils.process import RandRotation, RandHorizontalFlip
from scipy.stats import spearmanr, pearsonr
from torch.utils.tensorboard import SummaryWriter 
from tqdm import tqdm


# Leave CUDA selection to the runtime unless a specific GPU is explicitly needed.
# On CPU-only systems, the training code will fall back automatically.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_logging(config):
    if not os.path.exists(config.log_path): 
        os.makedirs(config.log_path)
    filename = os.path.join(config.log_path, config.log_file)
    logging.basicConfig(
        level=logging.INFO,
        filename=filename,
        filemode='w',
        format='[%(asctime)s %(levelname)-8s] %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(description="Train MANIQA with selectable crop/view strategy.")
    parser.add_argument(
        "--crop_mode",
        choices=["base_random", "global_fixed5"],
        default=None,
        help="Input view strategy. base_random uses one random crop; global_fixed5 uses global resize plus five fixed local crops."
    )
    parser.add_argument(
        "--crop_fusion",
        choices=["mean", "min"],
        default=None,
        help="How to fuse the five local crop scores in global_fixed5 mode."
    )
    parser.add_argument(
        "--local_weight",
        type=float,
        default=None,
        help="Weight for the local score in global_fixed5 mode. Final score = (1-w)*global + w*local."
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--n_epoch", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--val_freq", type=int, default=None, help="Override validation frequency in epochs.")
    parser.add_argument("--t_max", type=int, default=None, help="Override CosineAnnealingLR T_max.")
    parser.add_argument("--train_keep_ratio", type=float, default=None, help="Override train subset ratio.")
    parser.add_argument("--val_keep_ratio", type=float, default=None, help="Override validation subset ratio.")
    parser.add_argument("--split_seed", type=int, default=None, help="Seed used for the fixed 80/20 split.")
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to last_training_state.pt for resuming an interrupted run."
    )
    parser.add_argument(
        "--eval_crop_repeats",
        type=int,
        default=None,
        help="Number of deterministic random-crop passes averaged for base_random evaluation."
    )
    parser.add_argument(
        "--eval_protocol",
        choices=["validation", "test"],
        default=None,
        help=(
            "validation: evaluate periodically and save best checkpoint by eval score. "
            "test: treat the held-out split as test, evaluate only at the final epoch, "
            "and do not select checkpoints by test score."
        )
    )
    return parser.parse_args()


def predict_batch(config, net, data, device):
    if config.crop_mode == "global_fixed5":
        x_global = data['d_img_global'].to(device)
        x_local = data['d_img_local'].to(device)
        b, n, c, h, w = x_local.shape

        pred_global = net(x_global)
        pred_local = net(x_local.view(b * n, c, h, w)).view(b, n)

        if config.crop_fusion == "min":
            pred_local = pred_local.min(dim=1).values
        else:
            pred_local = pred_local.mean(dim=1)

        return (1 - config.local_weight) * pred_global + config.local_weight * pred_local

    x_d = data['d_img_org'].to(device)
    return net(x_d)


def train_epoch(config, epoch, net, criterion, optimizer, scheduler, train_loader):
    losses = []
    net.train()
    # save data for one epoch
    pred_epoch = []
    labels_epoch = []
    
    device = get_device()
    net.to(device)

    for data in tqdm(train_loader):
        labels = data['score']

        labels = torch.squeeze(labels.type(torch.FloatTensor)).to(device)
        pred_d = predict_batch(config, net, data, device)

        optimizer.zero_grad()
        loss = criterion(torch.squeeze(pred_d), labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()

        # save results in one epoch
        pred_batch_numpy = pred_d.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)
    
    # compute correlation coefficient
    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

    ret_loss = np.mean(losses)
    logging.info('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}'.format(epoch + 1, ret_loss, rho_s, rho_p))

    return ret_loss, rho_s, rho_p


def eval_epoch(config, epoch, net, criterion, test_loader):
    with torch.no_grad():
        net.eval()
        device = get_device()
        net.to(device)

        repeats = config.eval_crop_repeats if config.crop_mode == "base_random" else 1
        all_predictions = []
        labels_epoch = None
        image_names = None
        numpy_state = np.random.get_state()

        try:
            for repeat_idx in range(repeats):
                # Fixed evaluation seeds make repeated runs directly comparable.
                np.random.seed(config.eval_crop_seed + repeat_idx)
                pred_repeat = []
                labels_repeat = []
                names_repeat = []
                for data in tqdm(test_loader):
                    labels = data['score'].type(torch.FloatTensor).to(device)
                    pred = predict_batch(config, net, data, device)
                    pred_repeat = np.append(pred_repeat, pred.detach().cpu().numpy())
                    labels_repeat = np.append(labels_repeat, labels.detach().cpu().numpy())
                    names_repeat.extend(data.get('image_name', []))
                all_predictions.append(pred_repeat)
                if labels_epoch is None:
                    labels_epoch = labels_repeat
                    image_names = names_repeat
        finally:
            # Evaluation must not alter the random sequence used by later training epochs.
            np.random.set_state(numpy_state)

        pred_epoch = np.mean(np.stack(all_predictions, axis=0), axis=0)
        loss = float(np.mean((pred_epoch - labels_epoch) ** 2))
        
        # compute correlation coefficient
        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

        if image_names:
            prediction_name = (
                "test_predictions.csv" if config.eval_protocol == "test"
                else "eval_predictions_epoch{}.csv".format(epoch + 1)
            )
            with open(os.path.join(config.ckpt_path, prediction_name), "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["image_name", "target", "prediction", "crop_repeats"])
                for image_name, target, prediction in zip(image_names, labels_epoch, pred_epoch):
                    writer.writerow([image_name, float(target), float(prediction), repeats])

        logging.info(
            'Epoch:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4} ===== eval repeats:{}'.format(
                epoch + 1, loss, rho_s, rho_p, repeats
            )
        )
        return loss, rho_s, rho_p


def get_score_range(txt_file_name, train_names):
    train_names = set(train_names)
    scores = []
    with open(txt_file_name, 'r') as label_file:
        for line in label_file:
            image_name, score = line.split()
            if image_name in train_names:
                scores.append(float(score))
    if not scores:
        raise ValueError("No training scores were found for normalization.")
    return min(scores), max(scores)


def save_experiment_metadata(config, train_dataset, eval_dataset):
    os.makedirs(config.ckpt_path, exist_ok=True)
    metadata_path = os.path.join(config.ckpt_path, "experiment_config.json")
    with open(metadata_path, "w", encoding="utf-8") as config_file:
        json.dump(dict(config), config_file, indent=2, ensure_ascii=True)

    manifests = {
        "train_files.txt": train_dataset.data_dict['d_img_list'],
        "{}_files.txt".format(config.eval_protocol): eval_dataset.data_dict['d_img_list']
    }
    for file_name, image_names in manifests.items():
        with open(os.path.join(config.ckpt_path, file_name), "w", encoding="utf-8") as manifest_file:
            manifest_file.write("\n".join(image_names) + "\n")


def save_training_state(path, epoch, net, optimizer, scheduler, config):
    model_to_save = net.module if isinstance(net, nn.DataParallel) else net
    state = {
        "epoch": epoch,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": dict(config),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state()
    }
    if torch.cuda.is_available():
        state["cuda_random_state"] = torch.cuda.get_rng_state_all()
    torch.save(state, path)


def load_training_state(path, net, optimizer, scheduler, device, config):
    # Recovery checkpoints are created locally by this script and include RNG states,
    # so they require full (non-weights-only) deserialization on recent PyTorch.
    state = torch.load(path, map_location=device, weights_only=False)
    saved_config = state.get("config", {})
    resume_keys = [
        "crop_mode", "crop_fusion", "local_weight", "batch_size", "n_epoch",
        "split_seed", "train_keep_ratio", "val_keep_ratio", "T_max"
    ]
    mismatches = [
        key for key in resume_keys
        if key in saved_config and saved_config[key] != config[key]
    ]
    if mismatches:
        raise ValueError(
            "Resume configuration does not match the current run: {}".format(", ".join(mismatches))
        )
    model_to_load = net.module if isinstance(net, nn.DataParallel) else net
    model_to_load.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_random_state"])
    if torch.cuda.is_available() and "cuda_random_state" in state:
        torch.cuda.set_rng_state_all(state["cuda_random_state"])
    return int(state["epoch"])


if __name__ == '__main__':
    args = parse_args()

    cpu_num = 1
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    setup_seed(20)

    # config file
    config = Config({
        # dataset path
        "dataset_name": "koniq10k",

        # PIPAL
        "train_dis_path": "C:\\Users\\BTREEE\\work\\MANIQA\\data\\PIPAL22\\Train\\Train_dis",
        "val_dis_path": "C:\\Users\\BTREEE\\work\\MANIQA\\datasets\\PIPAL22\\Val_dis",
        "pipal22_train_label": "C:\\Users\\BTREEE\\work\\MANIQA\\data\\PIPAL22\\Train_Label",
        "pipal22_val_txt_label": "C:\\Users\\BTREEE\\work\\MANIQA\\data\\PIPAL22\\pipal21_val.txt",

        # KADID-10K
        "kadid10k_path": "C:\\Users\\BTREEE\\work\\MANIQA\\datasets\\kadid10k",
        "kadid10k_label": "C:\\Users\\BTREEE\\work\\MANIQA\\data\\kadid10k\\kadid10k_label.txt",

        # KONIQ-10K
        "koniq10k_path": "C:\\Users\\BTREEE\\work\\MANIQA\\datasets\\koniq10k\\1024x768",
        "koniq10k_label": "C:\\Users\\BTREEE\\work\\MANIQA\\data\\koniq10k\\koniq10k_label.txt",
        
        # optimization
        "batch_size": 2,
        "learning_rate": 1e-5,
        "weight_decay": 1e-5,
        "n_epoch": 4,
        "val_freq": 2,
        "T_max": 160,
        "eta_min": 0,
        "num_avg_val": 1, # if training koniq10k, num_avg_val is set to 1
        "num_workers": 0,
        
        # data
        "split_seed": 20,
        "eval_crop_seed": 2026,
        "eval_crop_repeats": 5,
        "train_keep_ratio": 0.01,
        "val_keep_ratio": 0.02,
        "eval_protocol": "test",
        "crop_size": 224,
        "crop_mode": "base_random", # "base_random" or "global_fixed5"
        "crop_fusion": "mean", # "mean" or "min" for global_fixed5 local scores
        "local_weight": 0.5,
        "prob_aug": 0.7,

        # model
        "patch_size": 8,
        "img_size": 224,
        "embed_dim": 768,
        "dim_mlp": 768,
        "num_heads": [4, 4],
        "window_size": 4,
        "depths": [2, 2],
        "num_outputs": 1,
        "num_tab": 2,
        "scale": 0.8,
        
        # load & save checkpoint
        "model_name": "koniq10k-quick_s20",
        "type_name": "Koniq10k",
        "ckpt_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "models"),
        "log_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "log"),
        "log_file": ".log",
        "tensorboard_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "tensorboard")
    })

    if args.crop_mode is not None:
        config.crop_mode = args.crop_mode
    if args.crop_fusion is not None:
        config.crop_fusion = args.crop_fusion
    if args.local_weight is not None:
        config.local_weight = args.local_weight
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.n_epoch is not None:
        config.n_epoch = args.n_epoch
    if args.val_freq is not None:
        config.val_freq = args.val_freq
    if args.t_max is not None:
        config.T_max = args.t_max
    if args.train_keep_ratio is not None:
        config.train_keep_ratio = args.train_keep_ratio
    if args.val_keep_ratio is not None:
        config.val_keep_ratio = args.val_keep_ratio
    if args.split_seed is not None:
        config.split_seed = args.split_seed
    if args.eval_crop_repeats is not None:
        config.eval_crop_repeats = args.eval_crop_repeats
    if args.eval_protocol is not None:
        config.eval_protocol = args.eval_protocol

    if not 0 < config.train_keep_ratio <= 1 or not 0 < config.val_keep_ratio <= 1:
        raise ValueError("train_keep_ratio and val_keep_ratio must be in (0, 1].")
    if not 0 <= config.local_weight <= 1:
        raise ValueError("local_weight must be in [0, 1].")
    if config.eval_crop_repeats < 1:
        raise ValueError("eval_crop_repeats must be at least 1.")
    
    if config.dataset_name == 'koniq10k':
        ratio_tag = "tr{}_ev{}".format(
            str(config.train_keep_ratio).replace(".", "p"),
            str(config.val_keep_ratio).replace(".", "p")
        )
        config.model_name = "{}_{}_{}_{}_split{}_ep{}_bs{}".format(
            config.model_name,
            config.eval_protocol,
            config.crop_mode,
            ratio_tag,
            config.split_seed,
            config.n_epoch,
            config.batch_size
        )
        if config.crop_mode == "global_fixed5":
            config.model_name = "{}_{}_lw{}".format(
                config.model_name,
                config.crop_fusion,
                str(config.local_weight).replace(".", "p")
            )
        else:
            config.model_name = "{}_r{}".format(config.model_name, config.eval_crop_repeats)

    config.log_file = config.model_name + ".log"
    config.tensorboard_path = os.path.join(config.tensorboard_path, config.type_name)
    config.tensorboard_path = os.path.join(config.tensorboard_path, config.model_name)

    config.ckpt_path = os.path.join(config.ckpt_path, config.type_name)
    config.ckpt_path = os.path.join(config.ckpt_path, config.model_name)

    config.log_path = os.path.join(config.log_path, config.type_name)

    if not os.path.exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)
    
    if not os.path.exists(config.tensorboard_path):
        os.makedirs(config.tensorboard_path)

    if config.dataset_name == 'koniq10k' and not os.path.exists(config.koniq10k_path):
        raise FileNotFoundError(
            f"KONIQ dataset path does not exist: {config.koniq10k_path}. "
            "Please update 'koniq10k_path' in train_maniqa.py to the folder that contains the KONIQ images."
        )

    set_logging(config)
    logging.info(config)

    writer = SummaryWriter(config.tensorboard_path)

    if config.dataset_name == 'kadid10k':
        from data.kadid10k.kadid10k import Kadid10k
        train_name, val_name = split_dataset_kadid10k(
            txt_file_name=config.kadid10k_label,
            split_seed=config.split_seed
        )
        dis_train_path = config.kadid10k_path
        dis_val_path = config.kadid10k_path
        label_train_path = config.kadid10k_label
        label_val_path = config.kadid10k_label
        Dataset = Kadid10k
    elif config.dataset_name == 'pipal':
        from data.PIPAL22.pipal import PIPAL
        dis_train_path = config.train_dis_path
        dis_val_path = config.val_dis_path
        label_train_path = config.pipal22_train_label
        label_val_path = config.pipal22_val_txt_label
        Dataset = PIPAL
    elif config.dataset_name == 'koniq10k':
        from data.koniq10k.koniq10k import Koniq10k
        train_name, val_name = split_dataset_koniq10k(
            txt_file_name=config.koniq10k_label,
            split_seed=config.split_seed
        )
        dis_train_path = config.koniq10k_path
        dis_val_path = config.koniq10k_path
        label_train_path = config.koniq10k_label
        label_val_path = config.koniq10k_label
        Dataset = Koniq10k
    else:
        pass
    
    # data load
    dataset_kwargs = {}
    train_dataset_kwargs = {}
    eval_dataset_kwargs = {}
    if config.dataset_name == 'koniq10k':
        dataset_kwargs = {
            "crop_mode": config.crop_mode,
            "crop_size": config.crop_size,
            "score_range": get_score_range(config.koniq10k_label, train_name)
        }
        train_dataset_kwargs = {"horizontal_flip_prob": config.prob_aug}
        eval_dataset_kwargs = {"horizontal_flip_prob": 0.0}

    train_transform = transforms.Compose([
        RandCrop(patch_size=config.crop_size),
        Normalize(0.5, 0.5),
        RandHorizontalFlip(prob_aug=config.prob_aug),
        ToTensor()
    ])
    val_transform = transforms.Compose([
        RandCrop(patch_size=config.crop_size),
        Normalize(0.5, 0.5),
        ToTensor()
    ])

    if config.crop_mode == "global_fixed5":
        train_transform = None
        val_transform = None

    train_dataset = Dataset(
        dis_path=dis_train_path,
        txt_file_name=label_train_path,
        list_name=train_name,
        transform=train_transform,
        keep_ratio=config.train_keep_ratio,
        **dataset_kwargs,
        **train_dataset_kwargs
    )
    val_dataset = Dataset(
        dis_path=dis_val_path,
        txt_file_name=label_val_path,
        list_name=val_name,
        transform=val_transform,
        keep_ratio=config.val_keep_ratio,
        **dataset_kwargs,
        **eval_dataset_kwargs
    )

    logging.info('number of train scenes: {}'.format(len(train_dataset)))
    logging.info('number of val scenes: {}'.format(len(val_dataset)))

    # load the data
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size,
        num_workers=config.num_workers, drop_last=True, shuffle=True)

    val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size,
        num_workers=config.num_workers, drop_last=False, shuffle=False)

    if len(train_loader) == 0:
        raise ValueError("Training subset is smaller than batch_size while drop_last=True.")
    if args.t_max is None:
        config.T_max = len(train_loader) * config.n_epoch

    save_experiment_metadata(config, train_dataset, val_dataset)
    logging.info('Final scheduler T_max: {}'.format(config.T_max))
    logging.info('Experiment metadata saved to: {}'.format(config.ckpt_path))

    # model defination
    net = MANIQA(embed_dim=config.embed_dim, num_outputs=config.num_outputs, dim_mlp=config.dim_mlp,
        patch_size=config.patch_size, img_size=config.img_size, window_size=config.window_size,
        depths=config.depths, num_heads=config.num_heads, num_tab=config.num_tab, scale=config.scale)

    logging.info('{} : {} [M]'.format('#Params', sum(map(lambda x: x.numel(), net.parameters())) / 10 ** 6))

    device = get_device()
    if torch.cuda.is_available():
        net = nn.DataParallel(net)
    net = net.to(device)

    # loss function
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)

    start_epoch = 0
    if args.resume is not None:
        start_epoch = load_training_state(args.resume, net, optimizer, scheduler, device, config)
        logging.info('Resumed training after epoch {} from {}'.format(start_epoch, args.resume))

    # train & validation/test
    losses, scores = [], []
    best_srocc = 0
    best_plcc = 0
    main_score = -np.inf
    for epoch in range(start_epoch, config.n_epoch):
        start_time = time.time()
        logging.info('Running training epoch {}'.format(epoch + 1))
        loss_val, rho_s, rho_p = train_epoch(config, epoch, net, criterion, optimizer, scheduler, train_loader)

        writer.add_scalar("Train_loss", loss_val, epoch)
        writer.add_scalar("SRCC", rho_s, epoch)
        writer.add_scalar("PLCC", rho_p, epoch)

        should_eval = (epoch + 1) % config.val_freq == 0
        if config.eval_protocol == "test":
            should_eval = (epoch + 1) == config.n_epoch

        if should_eval:
            eval_name = "test" if config.eval_protocol == "test" else "eval"
            logging.info('Starting {}...'.format(eval_name))
            logging.info('Running {} in epoch {}'.format(eval_name, epoch + 1))
            loss, rho_s, rho_p = eval_epoch(config, epoch, net, criterion, val_loader)
            logging.info('{} done...'.format(eval_name.capitalize()))

            writer.add_scalar("{}_loss".format(eval_name), loss, epoch)
            writer.add_scalar("{}_SRCC".format(eval_name), rho_s, epoch)
            writer.add_scalar("{}_PLCC".format(eval_name), rho_p, epoch)

            model_to_save = net.module if isinstance(net, nn.DataParallel) else net

            if config.eval_protocol == "test":
                model_name = "final_epoch{}.pt".format(epoch + 1)
                model_save_path = os.path.join(config.ckpt_path, model_name)
                torch.save(model_to_save.state_dict(), model_save_path)
                with open(os.path.join(config.ckpt_path, "final_metrics.json"), "w", encoding="utf-8") as metrics_file:
                    json.dump({
                        "epoch": epoch + 1,
                        "loss": loss,
                        "srcc": float(rho_s),
                        "plcc": float(rho_p),
                        "eval_crop_repeats": config.eval_crop_repeats if config.crop_mode == "base_random" else 1
                    }, metrics_file, indent=2)
                logging.info(
                    'Final fixed-epoch test result. epoch:{}, loss:{}, SRCC:{}, PLCC:{}'.format(
                        epoch + 1, loss, rho_s, rho_p
                    )
                )
                logging.info('Saving final fixed-epoch model: {}'.format(model_save_path))
            elif rho_s + rho_p > main_score:
                main_score = rho_s + rho_p
                best_srocc = rho_s
                best_plcc = rho_p

                logging.info('======================================================================================')
                logging.info('============================== best main score is {} ================================='.format(main_score))
                logging.info('======================================================================================')

                # save weights
                model_name = "epoch{}.pt".format(epoch + 1)
                model_save_path = os.path.join(config.ckpt_path, model_name)
                torch.save(model_to_save.state_dict(), model_save_path)
                with open(os.path.join(config.ckpt_path, "best_metrics.json"), "w", encoding="utf-8") as metrics_file:
                    json.dump({
                        "epoch": epoch + 1,
                        "loss": loss,
                        "srcc": float(best_srocc),
                        "plcc": float(best_plcc),
                        "selection_score": float(main_score)
                    }, metrics_file, indent=2)
                logging.info('Saving weights and model of epoch{}, SRCC:{}, PLCC:{}'.format(epoch + 1, best_srocc, best_plcc))
        
        recovery_path = os.path.join(config.ckpt_path, "last_training_state.pt")
        save_training_state(recovery_path, epoch + 1, net, optimizer, scheduler, config)
        logging.info('Recovery checkpoint saved: {}'.format(recovery_path))
        logging.info('Epoch {} done. Time: {:.2}min'.format(epoch + 1, (time.time() - start_time) / 60))

    writer.close()
