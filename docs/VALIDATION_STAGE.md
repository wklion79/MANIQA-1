# Reconstructed Validation Experiment

This branch reconstructs the second-stage validation comparison from local notes.
Its exact intermediate source snapshot was not committed during the original run,
so it must not be presented as an untouched historical commit.

Both crop pipelines use the same seeded split and quick-run ratios. Validation is
performed every two epochs and the checkpoint with the largest `SRCC + PLCC` is
saved. These validation values are model-selection results, not final independent
test results.

Base random crop:

```powershell
& .\.venv\Scripts\python.exe .\train_maniqa.py --eval_protocol validation --crop_mode base_random --batch_size 2 --n_epoch 4 --val_freq 2 --train_keep_ratio 0.01 --val_keep_ratio 0.02 --split_seed 20 --eval_crop_repeats 5
```

Global plus five fixed crops:

```powershell
& .\.venv\Scripts\python.exe .\train_maniqa.py --eval_protocol validation --crop_mode global_fixed5 --crop_fusion mean --local_weight 0.5 --batch_size 2 --n_epoch 4 --val_freq 2 --train_keep_ratio 0.01 --val_keep_ratio 0.02 --split_seed 20
```

Use the `main` branch for the final fixed train/test protocol.
