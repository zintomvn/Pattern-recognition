import os
import sys
import numpy as np
from tqdm import tqdm

import torch

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))
from common.utils import *


def test_module(model, test_data_loaders, forward_function, device='cuda', distributed=False, compute_loss=False):
    """Test module for Face Anti-spoofing

    Args:
        model (nn.module): fas model
        test_data_loaders (torch.dataloader): list of test data loaders
        forward_function (function): model forward function
        device (str, optional): Defaults to 'cuda'.
        distributed (bool, optional): whether to use distributed training. Defaults to False.
        compute_loss (bool, optional): if True, also compute CrossEntropy loss. Defaults to False.

    Returns:
        y_preds (list): predictions
        y_trues (list): ground truth labels
        val_loss (float, optional): average validation loss if compute_loss=True
    """
    prob_dict = {}
    label_dict = {}

    y_preds = []
    y_trues = []

    total_loss = 0.0
    num_samples = 0
    crit = torch.nn.CrossEntropyLoss() if compute_loss else None

    model.eval()
    num_iters = min(len(loaders) for loaders in test_data_loaders)
    for iter, all_datas in enumerate(zip(*test_data_loaders)):
        for datas in all_datas:
            with torch.no_grad():
                images = datas[0].to(device)
                targets = datas[1].to(device)
                map_GT = datas[2].to(device)
                img_path = datas[3]
                probs = forward_function(images)

                if compute_loss and crit is not None:
                    # Get logits by calling model directly (not via forward_function which returns softmax probs)
                    logits, _, _ = model(images)[:3]
                    loss = crit(logits, targets)
                    total_loss += loss.item() * targets.size(0)
                    num_samples += targets.size(0)

                if not distributed:
                    probs_np = probs.cpu().data.numpy()
                    label = targets.cpu().data.numpy()

                    for i in range(len(probs_np)):
                        video_path = img_path[i].rsplit('/', 1)[0]
                        if (video_path in prob_dict.keys()):
                            prob_dict[video_path].append(probs_np[i])
                            label_dict[video_path].append(label[i])
                        else:
                            prob_dict[video_path] = []
                            label_dict[video_path] = []
                            prob_dict[video_path].append(probs_np[i])
                            label_dict[video_path].append(label[i])
                else:
                    y_preds.extend(probs)
                    y_trues.extend(targets)

    if not distributed:
        y_preds = []
        y_trues = []
        for key in prob_dict.keys():
            # calculate the scores in video-level via averaging the scores of the images from the same videos
            avg_single_video_prob = sum(prob_dict[key]) / len(prob_dict[key])
            avg_single_video_label = sum(label_dict[key]) / len(label_dict[key])
            y_preds = np.append(y_preds, avg_single_video_prob)
            y_trues = np.append(y_trues, avg_single_video_label)

    val_loss = total_loss / num_samples if compute_loss and num_samples > 0 else None
    if compute_loss:
        return y_preds, y_trues, val_loss
    return y_preds, y_trues
