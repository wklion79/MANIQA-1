# Work Directory Experiment Notes

Last updated: 2026-07-10

This note is meant to make the current `C:\Users\BTREEE\work` workspace understandable from another PC, even when the large local dataset folders are not copied.

## Workspace Overview

```text
C:\Users\BTREEE\work
├─ MANIQA/          Main modified MANIQA project currently used for training
├─ MANIQA_PR/       Mostly environment/editor folder only in this workspace
├─ MANIQR_ver2/     Separate cleaner/restructured experiment project
├─ output/          Training outputs created from running MANIQA from work root
├─ .venv/           Root virtual environment, usually do not move via Git
└─ work/            Extra local folder
```

Large dataset folders are local-only and should generally not be pushed to GitHub:

```text
MANIQA/datasets/
MANIQA/data/        contains labels and may also contain large dataset material
```

## Main Current Project: MANIQA

Main files changed or important:

```text
MANIQA/train_maniqa.py
MANIQA/data/koniq10k/koniq10k.py
MANIQA/models/maniqa.py
MANIQA/models/maniqa_forward_walkthrough.md
MANIQA/models/maniqa_forward_walkthrough.ipynb
MANIQA/debug_forward.py
MANIQA/debug_forward_walkthrough.md
MANIQA/debug_forward_walkthrough.ipynb
MANIQA/requirements.txt
```

### Current Research Question

The working research idea is to compare two input crop/view strategies for MANIQA under the same backbone and training pipeline:

1. `base_random`
   - Uses one random `224x224` crop from the original image.
   - Represents the baseline random crop strategy.

2. `global_fixed5`
   - Uses one global image resized to `224x224`.
   - Uses five local fixed crops: top-left, top-right, center, bottom-left, bottom-right.
   - Runs the same MANIQA network on global and local views.
   - Fuses scores as:

```text
final_score = (1 - local_weight) * global_score + local_weight * local_score
```

where `local_score` is either mean or min over the five local crop scores.

The motivation is that random crop can miss localized artifacts/distortions. Example limitation:

```text
artifact appears only in top-left region
↓
random crop may sample center/right region
↓
artifact-free crop is evaluated
↓
image quality may be predicted too high
```

The proposed comparison checks whether global + fixed local crops react more consistently to localized distortions.

### Current Training Settings

In `MANIQA/train_maniqa.py`, the quick experiment settings are currently:

```python
"batch_size": 4,
"learning_rate": 1e-5,
"weight_decay": 1e-5,
"n_epoch": 10,
"val_freq": 2,
"T_max": 600,
"num_workers": 0,
"train_keep_ratio": 0.03,
"val_keep_ratio": 0.05,
"crop_size": 224,
"crop_mode": "base_random",
"crop_fusion": "mean",
"local_weight": 0.5,
"model_name": "koniq10k-quick_s20",
```

Reason for these settings:

- The PC only exposes `Intel(R) UHD Graphics 630`; PyTorch CUDA is not available.
- Training runs on CPU and is slow.
- Full KONIQ training is too large for the available time.
- The current setup is a pilot/quick ablation, not a full reproduction of MANIQA paper performance.

Approximate quick subset:

```text
KONIQ train full split: 8058 images
KONIQ val full split:   2015 images
train_keep_ratio=0.03:  about 241 images -> 60 batches with batch_size=4
val_keep_ratio=0.05:    about 100 images -> 25 batches with batch_size=4
```

### How To Run Experiments

Run baseline random crop:

```powershell
& c:\Users\BTREEE\work\MANIQA\.venv\Scripts\python.exe c:/Users/BTREEE/work/MANIQA/train_maniqa.py --crop_mode base_random
```

Run proposed global + fixed five crops:

```powershell
& c:\Users\BTREEE\work\MANIQA\.venv\Scripts\python.exe c:/Users/BTREEE/work/MANIQA/train_maniqa.py --crop_mode global_fixed5
```

Run proposed mode with min local fusion:

```powershell
& c:\Users\BTREEE\work\MANIQA\.venv\Scripts\python.exe c:/Users/BTREEE/work/MANIQA/train_maniqa.py --crop_mode global_fixed5 --crop_fusion min
```

Useful help command:

```powershell
& c:\Users\BTREEE\work\MANIQA\.venv\Scripts\python.exe c:/Users/BTREEE/work/MANIQA/train_maniqa.py --help
```

### Output Naming

The script automatically appends crop strategy info to `model_name`.

Expected output names:

```text
koniq10k-quick_s20_base_random
koniq10k-quick_s20_global_fixed5_mean_lw0p5
koniq10k-quick_s20_global_fixed5_min_lw0p5
```

Outputs may appear in either of these locations depending on the current working directory used when running:

```text
C:\Users\BTREEE\work\output\...
C:\Users\BTREEE\work\MANIQA\output\...
```

Known current root output examples:

```text
output/log/Koniq10k/koniq10k-base_s20.log
output/log/Koniq10k/koniq10k-quick_s20.log
output/models/Koniq10k/koniq10k-base_s20/epoch1.pt
output/models/Koniq10k/koniq10k-quick_s20/epoch2.pt
```

For clean future runs, prefer running from `C:\Users\BTREEE\work` and keep outputs in root `output/`.

### Important Code Changes Already Made

`MANIQA/train_maniqa.py`

- Added CLI args:

```text
--crop_mode {base_random,global_fixed5}
--crop_fusion {mean,min}
--local_weight FLOAT
```

- Added `predict_batch(config, net, data, device)` to handle both crop modes.
- Changed `train_epoch` to receive `config`.
- Changed eval to use `predict_batch`.
- Added automatic output naming by crop strategy.
- Fixed CPU checkpoint saving:

```python
model_to_save = net.module if isinstance(net, nn.DataParallel) else net
torch.save(model_to_save.state_dict(), model_save_path)
```

This prevents CPU runs from crashing on `net.module`.

`MANIQA/data/koniq10k/koniq10k.py`

- Fixed `keep_ratio`; originally it was accepted but not applied.
- Added `crop_mode` and `crop_size`.
- `base_random` returns:

```text
d_img_org: Tensor-like image after transform, shape [3, 224, 224]
score
```

- `global_fixed5` returns:

```text
d_img_global: Tensor, shape [3, 224, 224]
d_img_local:  Tensor, shape [5, 3, 224, 224]
score
```

### Validation Already Done

The following checks were run successfully:

```powershell
python -m py_compile MANIQA\train_maniqa.py MANIQA\data\koniq10k\koniq10k.py
```

Shape checks:

```text
base keys/shapes:
['d_img_org', 'score'] (3, 224, 224) (1,)

fixed keys/shapes:
['d_img_global', 'd_img_local', 'score'] (3, 224, 224) (5, 3, 224, 224) (1,)
```

Dummy `predict_batch` check:

```text
base pred:  (4,)
fixed pred: (4,)
```

### Warning Seen During Training

PyTorch warning:

```text
torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.
```

This is not a crash. It comes from model/timm internals and does not block current training.

### Hardware Note

GPU check showed:

```text
Intel(R) UHD Graphics 630
```

This is not a CUDA GPU. Current MANIQA training is effectively CPU-only. That is why each 60-batch epoch can take about 30+ minutes.

### Recommended Wording For Results

Because the experiment uses small subsets and CPU-limited quick training, describe it as:

```text
pilot ablation
quick experiment
localized distortion sensitivity test
```

Avoid claiming:

```text
full MANIQA reproduction
final benchmark performance
GAN artifact detection improvement
```

Better claim:

```text
Under the same MANIQA backbone and quick KONIQ subset setting, we compare whether global-fixed local views respond more consistently to localized real-world distortions than a random crop baseline.
```

## Controlled Distortion Test Idea

The planned test image is a dog photo with a natural-looking light flare / glare distortion.

Suggested experiment:

1. Use clean image and distorted versions.
2. Add localized real-world distortions rather than GAN-specific artifacts, because the current trained model uses KONIQ.
3. Recommended distortions:

```text
localized glare / overexposure
localized blur
localized haze / low contrast
localized sensor noise
localized JPEG/block artifact
```

For KONIQ justification, use phrasing like:

```text
PIPAL is more suitable for restoration/GAN-specific artifacts, but this experiment studies localized real-world distortion sensitivity, so a KONIQ-trained quick model is acceptable for a pilot crop-strategy test.
```

For `base_random`, run repeated inference on the same image because random crop is stochastic:

```text
base_random: 10-30 repeated scores per image, report mean and std
global_fixed5: deterministic fixed views, report score
score_drop = clean_score - distorted_score
```

Minimum recommended test set:

```text
1 image:
clean
top-left flare
center flare
bottom-right flare
```

Better pilot set:

```text
3 images x 4 versions = 12 images
```

## MANIQA Forward Flow Analysis Project

Files:

```text
MANIQA/models/maniqa.py
MANIQA/models/maniqa_forward_walkthrough.md
MANIQA/models/maniqa_forward_walkthrough.ipynb
MANIQA/debug_forward.py
MANIQA/debug_forward_walkthrough.md
MANIQA/debug_forward_walkthrough.ipynb
```

High-level MANIQA forward flow:

```text
input image [B, 3, 224, 224]
↓
ViT base patch8/224 from timm
↓
forward hooks collect ViT block outputs
↓
select block outputs 6, 7, 8, 9 without cls token
↓
concatenate selected features by channel dimension
↓
TAB stage 1
↓
1x1 conv
↓
SwinTransformer stage 1
↓
TAB stage 2
↓
1x1 conv
↓
SwinTransformer stage 2
↓
patch-level score branch and weight branch
↓
weighted average patch score
↓
image quality score
```

Notes:

- `MANIQA.forward(x)` still accepts a single tensor `[B, 3, 224, 224]`.
- The new `global_fixed5` strategy does not change the MANIQA architecture.
- It applies the same MANIQA network repeatedly to multiple views and fuses predicted scores outside the model.
- This keeps the comparison focused on input/crop strategy rather than changing the backbone.

## MANIQR_ver2 Project

`MANIQR_ver2` is a separate, cleaner experimental project structure. It appears to contain:

```text
MANIQR_ver2/configs/
MANIQR_ver2/data/
MANIQR_ver2/docs/
MANIQR_ver2/outputs/
MANIQR_ver2/scripts/
MANIQR_ver2/src/
MANIQR_ver2/tests/
```

Notable files:

```text
MANIQR_ver2/README.md
MANIQR_ver2/docs/MANIQA_DATASET_WORKFLOW.md
MANIQR_ver2/configs/baseline_random_crop.yaml
MANIQR_ver2/configs/fixed_multicrop.yaml
MANIQR_ver2/configs/global_local.yaml
MANIQR_ver2/configs/smoke_baseline_random.yaml
MANIQR_ver2/configs/smoke_global_local.yaml
MANIQR_ver2/scripts/train.py
MANIQR_ver2/scripts/run_ablation.py
MANIQR_ver2/scripts/compare_models.py
MANIQR_ver2/scripts/evaluate.py
MANIQR_ver2/scripts/generate_report.py
MANIQR_ver2/src/models/maniqa_baseline.py
MANIQR_ver2/src/models/global_local_maniqa.py
MANIQR_ver2/src/models/crop_aggregator.py
MANIQR_ver2/src/datasets/iqa_dataset.py
MANIQR_ver2/src/datasets/transforms.py
MANIQR_ver2/tests/test_crops.py
MANIQR_ver2/tests/test_aggregation.py
MANIQR_ver2/tests/test_metrics.py
MANIQR_ver2/tests/test_shapes.py
```

Existing small outputs:

```text
MANIQR_ver2/outputs/smoke_baseline_random/
MANIQR_ver2/outputs/smoke_global_local/
MANIQR_ver2/outputs/final_report.md
MANIQR_ver2/outputs/eda/
```

This project is useful if you want a cleaner, config-driven implementation later. The current active training discussion, however, has been happening in `MANIQA/train_maniqa.py`.

## MANIQA_PR

`MANIQA_PR` currently appears to contain only environment/editor folders:

```text
MANIQA_PR/.venv/
MANIQA_PR/.vscode/
```

No important project files were found in the quick scan.

## What To Move To Another PC

For code and notes:

```text
EXPERIMENT_NOTES.md
MANIQA/
MANIQR_ver2/
```

But exclude heavy/unnecessary folders:

```text
MANIQA/.venv/
MANIQA/datasets/
MANIQA/data/large image folders if present
MANIQA/__pycache__/
MANIQA/**/__pycache__/
MANIQA/output/tensorboard/ if not needed
MANIQA/output/models/ large .pt files if not needed
MANIQR_ver2/**/__pycache__/
```

If continuing training/inference, also move selected checkpoints:

```text
output/models/Koniq10k/.../*.pt
MANIQA/output/models/Koniq10k/.../*.pt
```

If reproducing logs/results:

```text
output/log/
MANIQA/output/log/
MANIQR_ver2/outputs/
```

Dataset requirement on another PC:

```text
KONIQ image folder expected by current train_maniqa.py:
C:\Users\BTREEE\work\MANIQA\datasets\koniq10k\1024x768

KONIQ label file expected:
C:\Users\BTREEE\work\MANIQA\data\koniq10k\koniq10k_label.txt
```

If paths differ on another PC, update these in `MANIQA/train_maniqa.py`:

```python
"koniq10k_path": "...",
"koniq10k_label": "...",
```

## Suggested GitHub Strategy

Do push:

```text
source code
configs
notes
small CSVs
small logs
README/docs
```

Do not push:

```text
datasets
.venv folders
large checkpoints
tensorboard event files if large
__pycache__
```

Use `.gitignore` patterns such as:

```gitignore
.venv/
**/.venv/
__pycache__/
**/__pycache__/
datasets/
**/datasets/
*.pt
*.pth
events.out.tfevents.*
```

If checkpoints must be shared, use Google Drive, OneDrive, external SSD, or Git LFS rather than normal GitHub.

## Next Recommended Steps

1. Finish or stop the current `base_random` quick run.
2. Record best SRCC/PLCC from:

```text
output/log/Koniq10k/...
```

3. Run `global_fixed5` with the same quick settings.
4. Compare best SRCC/PLCC and validation loss.
5. Create controlled distortion images:

```text
clean
top-left glare
center glare
bottom-right glare
```

6. Build or run an inference script that reports:

```text
image_name
crop_mode
score
score_drop
```

7. For `base_random`, repeat inference 10-30 times and report mean/std.
8. Write conclusions as a pilot localized-distortion sensitivity test, not as full benchmark proof.

