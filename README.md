# Pattern-recognition
Final project

## Install
```bash
pip install numpy opencv-python wandb easydict omegaconf timm albumentations
```

## Preprocess depth map

python preprocess_depth.py --config small.yml

## Run train

python train.py -c configs/small.yml --test