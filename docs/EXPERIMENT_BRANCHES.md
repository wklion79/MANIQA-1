# Experiment Branches

This repository keeps the crop-strategy experiment stages in separate branches.
Large datasets, virtual environments, checkpoints, TensorBoard files, and local
diagnostic images are intentionally excluded from Git.

## Branch Map

| Branch | Purpose | Status |
| --- | --- | --- |
| `experiment/01-original-baseline` | Repository state before the crop-strategy experiment | Historical commit |
| `experiment/02-validation-crop-ablation` | Base-random vs global-fixed5 validation workflow | Reconstructed from local experiment notes |
| `main` | Final fixed train/test protocol and paired statistical comparison | Current final implementation |

The validation branch is explicitly marked as reconstructed because its exact
intermediate source state was not committed while the experiment was running.

## Final Train/Test Protocol

The final pilot uses a deterministic 80/20 split. With the quick-run ratios, it
uses 80 training images and 40 held-out test images. The test split is evaluated
only after the final fixed epoch and is never used for checkpoint selection.

Base random-crop run:

```powershell
& .\.venv\Scripts\python.exe .\train_maniqa.py --eval_protocol test --crop_mode base_random --batch_size 2 --n_epoch 4 --train_keep_ratio 0.01 --val_keep_ratio 0.02 --split_seed 20 --eval_crop_repeats 5
```

Global plus five fixed local crops:

```powershell
& .\.venv\Scripts\python.exe .\train_maniqa.py --eval_protocol test --crop_mode global_fixed5 --crop_fusion mean --local_weight 0.5 --batch_size 2 --n_epoch 4 --train_keep_ratio 0.01 --val_keep_ratio 0.02 --split_seed 20
```

Each run writes its configuration, exact train/test manifests, per-image test
predictions, final metrics, final model weights, and a resumable training state
under `output/models/Koniq10k/<experiment-name>/`.

Paired comparison:

```powershell
& .\.venv\Scripts\python.exe .\compare_crop_results.py --base_csv <base-test_predictions.csv> --global_csv <global-test_predictions.csv> --bootstrap 2000 --seed 20 --output .\output\crop_comparison.json
```

## Interpretation Boundary

This is a constrained pilot comparison of two training pipelines. The global
pipeline evaluates six views per training image, while the baseline evaluates one
random crop. Results are therefore not a compute-matched architecture ablation.
The small subset also does not establish general improvement on all KONIQ images
or GAN artifacts; confidence intervals and the localized-distortion diagnostic
experiment should be reported separately.
