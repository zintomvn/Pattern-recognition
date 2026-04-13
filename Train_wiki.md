# Training Wiki — ANRL Face Anti-Spoofing Project

> This document is the collective knowledge base from the training setup session.
> All decisions, code changes, and usage patterns are recorded here for future agents and users.

---

## 1. Project Overview

**Task**: Adaptive Normalized Representation Learning (ANRL) for Generalizable Face Anti-Spoofing.
**Paper**: ACM MM 2021.
**Framework**: Meta-learning (MAML-style), Domain-Generalization (DG).

### Key Files

| File | Role |
|---|---|
| `src/train.py` | `ANRLTask` — task-specific training logic (meta-train / meta-test loop) |
| `src/common/task/base_task.py` | `BaseTask` — generic training scaffold (env, optimizer, checkpointing, logging) |
| `src/common/utils/parameters.py` | CLI argument parser + OmegaConf config merging |
| `src/datasets/DG_dataset.py` | `DG_Dataset` — domain-generalization dataset (CASIA, MSU-MFSD, NUAA, replayattack) |
| `src/common/task/fas/modules.py` | `test_module()` — validation/test inference + metric computation |
| `src/configs/CM2N.yaml` | Training config (epochs, batch_size, LR, losses, wandb, exam_dir) |

---

## 2. Config System

### OmegaConf Merge Order (Critical)

```python
# parameters.py — line 28–37
_C = OmegaConf.load(args.config)     # 1. Load YAML
_C.merge_with(vars(args))             # 2. CLI args override YAML values
```

**Important consequence**: CLI arguments (`--resume`, `--distributed`, `--test`, etc.) **always override** whatever is in the YAML. This is intentional — it lets you override config without editing files.

### Standard Config Keys (`CM2N.yaml`)

```yaml
seed: 1234
exam_dir: /content/drive/MyDrive/HCMUS/PatternRecog/Pattern-recognition/checkpoints  # Google Drive!

dataset:
  name: DG_Dataset
  DG_Dataset:
    img_mode: rgb_hsv
    depth_map: True
    depth_map_size: 32
    # list paths for train_pos/neg, val_pos/neg, test_pos/neg per domain

transform:
  image_size: 256
  mean: [0.5, 0.5, 0.5]
  std: [0.5, 0.5, 0.5]
  num_segments: 2

model:
  name: Framework
  ckpt_path:          # empty — no pretrained weights
  resume:             # set via --resume CLI arg instead (see §4)
  params:
    in_ch: 6           # RGB + HSV
    mid_ch: 384
    AdapNorm: True

loss:                  # Classification (CrossEntropy)
loss_2:                # Depth regression (MSELoss), weight=0.1
loss_3:                # Domain + Discrimination (EuclideanLoss), domain_weight=0.001, discri_weight=0.001

optimizer:
  name: Adam
  params:
    lr: 0.0001
    weight_decay: 0.00001

scheduler:
  name: StepLR
  params:
    step_size: 20
    gamma: 0.5

train:
  epochs: 2            # only 2 epochs — protect checkpoints!
  batch_size: 16
  print_interval: 10
  metasize: 1
  meta_step_size: 0.0001

val:
  batch_size: 16

test:
  batch_size: 16
  record_results: True

wandb:
  project: ANRL
  group: CMtoN
  mode: offline        # offline = no internet needed, logs saved locally
  save_code: False
```

---

## 3. Training Loop (ANRLTask)

### Flow in `fit()`

```
for epoch in 1..epochs:
    self.train(epoch)       # meta-train + meta-test, logs metrics to wandb
    self.validate(epoch)   # eval on val set, saves best checkpoint, logs Val_Loss
self.test()                # eval on test set using model_best.pth.tar
```

### `train()` — What Gets Logged (wandb)

Called once per epoch (after `validate()`). Logs 10 metrics at wandb `step=epoch`:

| Wandb Key | Source |
|---|---|
| `Meta_train_Acc` | Meta-train accuracy |
| `Meta_train_Class` | Classification loss (Loss 1) |
| `Meta_train_Depth` | Depth loss (Loss 2) |
| `Meta_train_Domain` | Domain loss (Loss 3) |
| `Meta_train_Discri` | Discrimination loss (Loss 3) |
| `Meta_test_Acc` | Meta-test accuracy |
| `Meta_test_Class` | Meta-test classification loss |
| `Meta_test_Depth` | Meta-test depth loss |
| `Meta_test_Domain` | Meta-test domain loss |
| `Meta_test_Discri` | Meta-test discrimination loss |

**Total combined loss**: `Loss_1 + Loss_2 * 0.1 + Loss_3_domain * 0.001 + Loss_3_discri * 0.001`

### `validate()` — What Gets Logged (wandb)

| Wandb Key | Source |
|---|---|
| `Val_Loss` | CrossEntropy loss on validation set (epoch-level average) |

Called after each `train()`. Also saves:
- **Best checkpoint** (by AUC) → `ckpts/model_best.pth.tar`
- See §4 for periodic checkpoint behavior.

---

## 4. Checkpoint Strategy (Colab Persistence)

### Problem

Colab local disk is ephemeral — cleared when session disconnects. All checkpoints must go to Google Drive.

**Solution**: `exam_dir` set to Google Drive path:
```yaml
exam_dir: /content/drive/MyDrive/HCMUS/PatternRecog/Pattern-recognition/checkpoints
```

### Checkpoint Types

| Type | Trigger | Path | Keep |
|---|---|---|---|
| **Best** | New best AUC in validation | `ckpts/model_best.pth.tar` | 1 (overwritten if better) |
| **Periodic** | After each `validate()` call | `ckpts/model_epoch_N.pth.tar` | Latest 5 (rolling window) |

### Periodic Checkpoint Implementation

Added to `src/common/task/base_task.py` — `_save_periodic_checkpoint()`:

```python
def _save_periodic_checkpoint(self, epoch, monitor_metric='ACC'):
    """Save a periodic checkpoint every epoch to Google Drive for Colab persistence.
    Keeps the latest 5 periodic checkpoints (rolling window).
    """
    if self.cfg.local_rank == 0 and hasattr(self, 'saver') and self.saver is not None:
        ckpt_dir = f'{self.cfg.exam_dir}/ckpts'
        os.makedirs(ckpt_dir, exist_ok=True)

        # Keep only latest 5 periodic checkpoints
        existing = sorted(glob.glob(f'{ckpt_dir}/model_epoch_*.pth.tar'))
        while len(existing) >= 5:
            oldest = existing.pop(0)
            if os.path.exists(oldest):
                os.remove(oldest)

        ckpt_name = f'{ckpt_dir}/model_epoch_{epoch}.pth.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
        }, ckpt_name)
        self.logger.info(f'Periodic checkpoint saved: {ckpt_name}')
```

**Called in `train.py` `fit()` loop**, after `self.validate(epoch)`:
```python
if self.cfg.local_rank == 0 and not getattr(self.cfg, 'test_only', False):
    self._save_periodic_checkpoint(epoch)
```

### Decision: Why Rolling 5 Instead of All?

With only **2 epochs**, saving every epoch is sufficient. The rolling 5 prevents disk bloat if epochs are later increased. It also means after 2 epochs you have both `model_epoch_1.pth.tar` and `model_epoch_2.pth.tar`.

---

## 5. Quick Reference

### Run Training
```bash
python train.py -c configs/CM2N.yaml
```

### Resume Training (after Colab disconnect)
```bash
python train.py -c configs/CM2N.yaml \
    --resume /content/drive/MyDrive/HCMUS/PatternRecog/Pattern-recognition/checkpoints/ckpts/model_epoch_1.pth.tar
```

### Run Test Only (no training, no checkpoint saving)
```bash
python train.py -c configs/CM2N.yaml --test
```

### Check Saved Checkpoints
```bash
ls /content/drive/MyDrive/HCMUS/PatternRecog/Pattern-recognition/checkpoints/ckpts/
```

### Expected wandb Metrics
```
Train (per epoch): Meta_train_Acc, Meta_train_Class, Meta_train_Depth,
                   Meta_train_Domain, Meta_train_Discri,
                   Meta_test_Acc, Meta_test_Class, Meta_test_Depth,
                   Meta_test_Domain, Meta_test_Discri
Val   (per epoch): Val_Loss
```

---

## 9. Key Design Decisions Summary

| # | Decision | Rationale |
|---|---|---|
| 1 | `--resume` as CLI arg, not YAML field | Avoids editing YAML on each resume; overrides cleanly |
| 2 | Periodic checkpoint every epoch (rolling 5) | 2-epoch training = high risk of losing progress; rolling keeps disk clean |
| 3 | Google Drive as checkpoint dir | Colab local disk is ephemeral; Drive is the only persistent option |
| 4 | `Val_Loss` via `compute_loss=True` flag in `test_module` | Non-breaking change; default behavior unchanged |
| 5 | Fail-fast in `_convert_to_depth` | Silent fallthrough masked the bug; descriptive `FileNotFoundError` helps debug |
| 6 | `wandb.mode: offline` | Colab may have no internet; offline logs sync later when connected |
| 7 | OmegaConf merge order: YAML → CLI | CLI always wins; enables overriding without editing files |
