# ANRL Face Anti-Spoofing — Codebase Knowledge Base

> For AI agents. Everything needed to understand, modify, and run this project.

---

## Project Overview

- **Task**: Adaptive Normalized Representation Learning (ANRL) — Domain-Generalization Face Anti-Spoofing
- **Paper**: ACM MM 2021
- **Framework**: MAML-style meta-learning + domain generalization
- **Domains**: CASIA, MSU-MFSD, NUAA (train), replayattack (test)
- **Input**: RGB + HSV image (6-channel), with depth map supervision

---

## Architecture

### Model (`src/models/framework.py`)

```
Framework
├── FeatExtractor(in_ch=6, mid_ch=384)   # custom CNN, see §Model Details
├── Classifier   (FeatEmbedder, 384→2)  # binary classification
└── DepthEstmator(384→1)                 # depth map regression
```

**Forward**: `x → FeatExtractor → (feat, dx4)` → `Classifier(feat)` + `DepthEstmator(feat)`

### Model Details — FeatExtractor (NOT ResNet/VGG)

Pure conv only (ignore BN, ReLU, pooling, skip connections for structural comparison):

```
Input (6ch: RGB+HSV)
 ├── inc:    conv 6→64
 ├── down1:  conv 64→128 → 128→196 → 196→128,  max_pool
 ├── down2:  conv 128→128 → 128→196 → 196→128, max_pool
 └── down3:  conv 128→128 → 128→196 → 196→128, max_pool

Output: dx2 + dx3 (adaptive_pool to 32×32) + dx4 → concat → 384ch

Total: 10 conv layers, no skip connections, bottleneck intermediate 196ch
```

**Not ResNet** — no residual/skip connections, no 64→128→256→512 doubling, only ~10 layers (ResNet18 has 20).

**Not VGG** — non-standard channel progression, custom intermediate bottleneck 196.

It's a **custom lightweight CNN** purpose-built for MAML meta-learning (note `params` argument for gradient-based weight updates).

---

## Data Flow

```
CM2N.yaml (OmegaConf)
    │
    ├── dataset.DG_Dataset: kwargs → DG_Dataset.__init__
    │   ├── train_pos_list1/2/3_path → items_1/2/3 (3 domains)
    │   ├── val_pos/neg_list_path   → items
    │   ├── test_pos/neg_list_path   → items
    │   └── augment_datasets: []    → moire gating (§Augmentation)
    │
    ├── transform: → create_dg_data_transforms(args.transform, split)
    │   └── train: HorizontalFlip + HueSaturationValue(p=0.1) + Resize + Normalize + ToTensor
    │       val/test: Resize + Normalize + ToTensor
    │
    ├── model: Framework(in_ch=6, mid_ch=384, AdapNorm=True, ...)
    │
    └── train/val/test: batch_size, epochs, etc.

↓
src/datasets/factory.py: create_dataloader(args, split, category)
    ├── extracts kwargs = args.dataset.DG_Dataset
    ├── builds dg_transform
    └── returns DataLoader(DG_Dataset(...))
```

---

## DG_Dataset (`src/datasets/DG_dataset.py`)

### Initialization

```python
def __init__(self, data_root=None, split='train', category=None,
             transform=None, img_mode='rgb', print_info=False, **kwargs):
```

All `**kwargs` keys from YAML become `self.<key>` via `setattr`. Every YAML key in `dataset.DG_Dataset` is an instance attribute.

### Key Attributes

| Attribute | Source | Purpose |
|---|---|---|
| `use_LMDB` | YAML | LMDB vs disk read |
| `augment_datasets` | YAML | moire augmentation gating |
| `img_mode` | YAML | `'rgb'` or `'rgb_hsv'` |
| `depth_map`, `depth_map_size` | YAML | depth preprocessing |
| `crop_face_from_5points` | YAML | optional face crop |
| `return_path` | YAML | adds `img_path` to `__getitem__` return |

### `__getitem_once` — Single-domain item retrieval

Used by both train (3 domains) and val/test (1 domain). Logic:

1. Read image (LMDB or `cv2.imread`) + depth map (path from `_convert_to_depth`)
2. **Moire augmentation** (if gated, see §Augmentation)
3. Zero depth if `label != 0` (spoof = no real depth)
4. Optional face crop
5. HSV conversion if `img_mode == 'rgb_hsv'`
6. Transform → Tensor
7. Depth resize to `depth_map_size` (default 32)
8. Return `(img_6ch, label, depth)` or `(img_6ch, label, depth, img_path)` if `return_path`

### `_convert_to_depth` — Domain-to-depth path mapping

```python
def _convert_to_depth(self, img_path):
    if 'replayattack' in img_path: return img_path.replace('replayattack', 'replayattack_depth')
    if 'CASIA'     in img_path: return img_path.replace('CASIA', 'CASIA_depth')
    if 'MSU-MFSD'  in img_path: return img_path.replace('MSU-MFSD', 'MSU-MFSD_depth')
    if 'NUAA'      in img_path: return img_path.replace('NUAA', 'NUAA_depth')
    # else: raises FileNotFoundError
```

### Domain Detection (for augmentation)

```python
for d in ['CASIA', 'MSU-MFSD', 'NUAA', 'replayattack']:
    if d in img_path: domain = d
```

---

## Augmentation System

### Moire + ColorJitter (physical spoof simulation)

Gated per-domain via `augment_datasets` in YAML. When triggered:
1. Apply `Moire` (polar-coordinate warp, 10% probability)
2. Apply `ColorJitter` (brightness/contrast/saturation/hue ±0.4)
3. Re-label from `0` (real) → `1` (spoof)

```python
# DG_dataset.py __init__
self.augment_datasets = getattr(self, 'augment_datasets', [])   # default empty
self.moire = Moire()                                            # src/extensions/simulate/moire.py
self.color_jitter = tv_transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)

# DG_dataset.py __getitem_once — after depth is None check
if self.split == 'train' and label == 0:
    domain = None
    for d in ['CASIA', 'MSU-MFSD', 'NUAA', 'replayattack']:
        if d in img_path: domain = d; break
    if domain and domain in self.augment_datasets and random.random() < 0.1:
        img = self.moire(img)
        img = cv2.cvtColor(np.array(Image.fromarray(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transform(
            self.color_jitter)), cv2.COLOR_RGB2BGR)
        label = 1
```

### Enabling Augmentation

```yaml
# CM2N.yaml — under dataset.DG_Dataset
augment_datasets: []        # empty = no augmentation (safe default)
augment_datasets: ['CASIA'] # only CASIA
augment_datasets: ['CASIA', 'MSU-MFSD', 'NUAA']  # all train domains
```

### Other Augmentations

`transform` in YAML controls train-time augmentations (built in `datasets/factory.py`):
- `HorizontalFlip` (train only)
- `HueSaturationValue(p=0.1)` (train only)
- `Resize`, `Normalize`, `ToTensor` (always)

---

## Training Loop

```
fit()
  for epoch in 1..epochs:
      self.train(epoch)         → logs 10 wandb metrics
      self.validate(epoch)     → logs Val_Loss, saves model_best.pth.tar
      _save_periodic_checkpoint(epoch)  → saves model_epoch_N.pth.tar
  self.test()                  → uses model_best.pth.tar
```

### Wandb Metrics

| Key | Meaning |
|---|---|
| `Meta_train_Acc/Class/Depth/Domain/Discri` | Meta-train (inner loop) |
| `Meta_test_Acc/Class/Depth/Domain/Discri` | Meta-test (outer loop) |
| `Val_Loss` | Validation CrossEntropy loss |

**Loss**: `Loss_1 + Loss_2*0.1 + Loss_3_domain*0.001 + Loss_3_discri*0.001`

---

## Config System

### OmegaConf Merge Order

```
1. Load YAML  → OmegaConf.load(args.config)
2. Override   → _C.merge_with(vars(args))   ← CLI always wins
```

CLI args (`--resume`, `--test`, `--distributed`, etc.) always override YAML. This is intentional — enables override without editing files.

### Key Config Keys (`CM2N.yaml`)

| Section | Key | Notes |
|---|---|---|
| `dataset.DG_Dataset` | `img_mode` | `'rgb'` or `'rgb_hsv'` — changes return shape |
| `dataset.DG_Dataset` | `depth_map`, `depth_map_size` | Depth preprocessing |
| `dataset.DG_Dataset` | `augment_datasets` | Moire gating list |
| `dataset.DG_Dataset` | `use_LMDB` | LMDB vs disk read |
| `dataset.DG_Dataset` | `crop_face_from_5points` | Optional face crop |
| `model.params` | `in_ch` | Default 6 (RGB+HSV) |
| `model.params` | `AdapNorm` | True = BN+IN blend, False = BN only |
| `model.params` | `AdapNorm_attention_flag` | `'1layer'` or `'2layer'` FC gating |
| `model.params` | `model_initial` | `'kaiming'` init method |
| `optimizer.params.lr` | `lr` | Default 0.0001 |
| `train.epochs` | `epochs` | Keep low to protect checkpoints |
| `train.metasize` | `metasize` | Meta-batch size |
| `train.meta_step_size` | `meta_step_size` | Inner loop LR |
| `wandb.mode` | `offline` | No internet needed; logs sync later |
| `exam_dir` | — | Checkpoint dir — must be Google Drive in Colab |

---

## Checkpoint System

| Type | Trigger | Path | Keep |
|---|---|---|---|
| **Best** | New best AUC in validate | `ckpts/model_best.pth.tar` | 1 |
| **Periodic** | After each validate | `ckpts/model_epoch_N.pth.tar` | 5 (rolling) |

**Colab warning**: local disk is ephemeral. `exam_dir` must point to Google Drive.

---

## Quick Commands

```bash
# Train
python train.py -c configs/CM2N.yaml

# Resume
python train.py -c configs/CM2N.yaml --resume <ckpt_path>

# Test only (no training, no checkpoint saving)
python train.py -c configs/CM2N.yaml --test

# Check checkpoints
ls $(python -c "from common.utils.parameters import get_parameters; print(get_parameters().exam_dir)")/ckpts/
```

---

## Key Design Decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | `--resume` as CLI arg | Override cleanly without editing YAML |
| 2 | Periodic checkpoint rolling 5 | 2-epoch training = high loss risk; rolling keeps disk clean |
| 3 | Google Drive as `exam_dir` | Colab local disk is ephemeral |
| 4 | `Val_Loss` via `compute_loss=True` flag | Non-breaking, default unchanged |
| 5 | `FileNotFoundError` fail-fast in `_convert_to_depth` | Silent fallthrough masked bugs |
| 6 | `augment_datasets` per-domain gating | Unlisted domains untouched; preserves DG goal |
| 7 | `wandb.mode: offline` | Colab may have no internet |
| 8 | OmegaConf: YAML → CLI | CLI always wins for overrides |

---

## File Inventory

| File | Role |
|---|---|
| `train.py` | `ANRLTask` — meta-train/meta-test loop |
| `src/common/task/base_task.py` | `BaseTask` — scaffold (opt, ckpt, logger) |
| `src/common/utils/parameters.py` | CLI parser + OmegaConf merge |
| `src/datasets/DG_dataset.py` | `DG_Dataset` — domain-generalization dataset |
| `src/datasets/factory.py` | `create_dataloader` — builds DataLoader + transform |
| `src/models/framework.py` | `Framework`, `FeatExtractor`, `FeatEmbedder`, `DepthEstmator` |
| `src/models/base_block.py` | `Conv_block` (with AdapNorm), `Basic_block` |
| `src/common/task/fas/modules.py` | `test_module()` — validation/test inference |
| `src/extensions/simulate/moire.py` | `Moire` class — physical moire augmentation |
| `src/extensions/simulate/CvprDataset_P1.py` | Alternative dataset (P1 task, different augmentation) |
| `src/configs/CM2N.yaml` | Main training config |
| `Train_wiki.md` | This file |