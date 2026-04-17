# ANRL — Quick Reference

> For AI agents. Short and to the point.

---

## Run Commands

```bash
python train.py -c configs/CM2N.yaml
python train.py -c configs/CM2N.yaml --resume <ckpt>
python train.py -c configs/CM2N.yaml --test
```

## Architecture

```
Framework
├── FeatExtractor(in_ch=6, mid_ch=384)   # custom 10-layer CNN (no ResNet/VGG)
├── Classifier   (FeatEmbedder, 384→2)   # binary classification
└── DepthEstmator(384→1)                 # depth map regression

Forward: x → FeatExtractor → (feat, dx4) → Classifier(feat) + DepthEstmator(feat)
```

## Data Flow

```
YAML → DG_Dataset
  ├── train_pos/neg_list1/2/3_path → items_1/2/3 (domains)
  ├── val/test_pos/neg_list_path   → single items
  └── augment_datasets: []        → moire gating

→ factory.create_dataloader → DataLoader → (img[6ch], label, depth)
```

`img_mode`: `'rgb'` (3ch) or `'rgb_hsv'` (6ch). `depth_map: True` required.

## DG_Dataset `__getitem_once` Logic

1. Read image + depth map (LMDB or `cv2.imread`)
2. Moire augmentation (10% prob, gated by `augment_datasets`)
3. Zero depth if `label != 0` (spoof)
4. HSV if `img_mode == 'rgb_hsv'`
5. Transform → Tensor, depth resize → `depth_map_size`

## Loss

```
Total = Loss_1 (CE) + Loss_2 (MSE) * 0.1 + Loss_3_domain * 0.001 + Loss_3_discri * 0.001
```

## Key Config Options

| Key | Purpose |
|---|---|
| `use_LMDB` | LMDB vs disk read |
| `img_mode: rgb_hsv` | 6-channel input (RGB+HSV) |
| `depth_map: True`, `depth_map_size: 32` | Depth supervision |
| `augment_datasets: []` | Moire gating (`[]` = off) |
| `model.params.AdapNorm` | BN+IN blend (True) or BN only (False) |
| `train.metasize` | Meta-batch size (# train domains) |
| `train.meta_step_size` | Inner loop LR |
| `wandb.mode: offline` | No internet needed |
| `exam_dir` | Checkpoint dir (Google Drive in Colab) |

## Checkpoints

- `ckpts/model_best.pth.tar` — best AUC
- `ckpts/model_epoch_N.pth.tar` — periodic (keeps last 5)

## File Inventory

| File | Role |
|---|---|
| `src/train.py` | `ANRLTask` — meta-train/meta-test loop |
| `src/datasets/DG_dataset.py` | `DG_Dataset` — domain-generalization dataset |
| `src/datasets/factory.py` | `create_dataloader` |
| `src/models/framework.py` | `Framework`, `FeatExtractor`, `Classifier`, `DepthEstmator` |
| `src/models/base_block.py` | `Conv_block`, `Basic_block` |
| `src/common/task/fas/modules.py` | `test_module()` — val/test inference |
| `src/common/task/base_task.py` | `BaseTask` — scaffold |
| `src/common/utils/parameters.py` | CLI parser + OmegaConf merge |
| `src/extensions/simulate/moire.py` | `Moire` augmentation |
| `src/configs/CM2N.yaml` | Main config |
