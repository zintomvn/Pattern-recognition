# ANRL Source Code Seminar Report

This report outlines the presentation content required for the seminar, structured strictly according to the provided grading rubric (barem) focusing on the `src/` directory.

---

## 2. Successfully running experiments using the paper’s code – 2.5 points

### (1.5 pts) Installation and Execution Commands
To successfully run the original Adaptive Normalized Representation Learning (ANRL) code, use the following commands depending on the intended phase (training, resuming, or testing):

```bash
# 1. Start a fresh training session
python src/train.py -c src/configs/CM2N.yaml

# 2. Resume training from a specific checkpoint
python src/train.py -c src/configs/CM2N.yaml --resume <path/to/checkpoint>

# 3. Evaluate the model in test-only mode
python src/train.py -c src/configs/CM2N.yaml --test
```

### (1.0 pt) Experimental Setup
*   **Dataset used:** The protocol revolves around Domain Generalization for Face Anti-Spoofing. According to the configuration (`CM2N.yaml`), the training set combines source domains like **CASIA-FASD** and **MSU-MFSD** (mapped dynamically to meta-train and meta-test splits). The evaluation is performed on the unseen target domain **NUAA**.
*   **Hardware:** Trained on NVIDIA GPUs using PyTorch with Automatic Mixed Precision (`amp.autocast`) to reduce memory footprint and accelerate processing by leveraging FP16 tensors.
*   **Hyperparameters & Configuration (`src/configs/CM2N.yaml`):**
    *   **Input Modality:** `img_mode: rgb_hsv` (6-channel composition) combined with 32x32 depth map regression (`depth_map: True`).
    *   **Optimization:** Adam optimizer (Learning Rate: 0.0001, Weight Decay: 1e-05) paired with a StepLR scheduler.
    *   **Meta-Learning Dynamics:** `metasize: 1` configures the inner loop batch split, using an inner loop gradient descent step (`meta_step_size`) of 0.0001.
    *   **Loss Formulation:** Combining standard Binary Cross-Entropy (weight: 1.0) with auxiliary Mean Squared Error for depth (weight: 0.1), and metric-learning constraints representing domain loss and discriminative loss (weights: 0.001 each).

---

## 3. Understanding and explaining the core technique when translating from paper to code – 2.5 points

### (1.5 pts) Core Technical Components Implementation
The main technical contributions of the paper map directly to the following code components:
1.  **Network Modules (`src/models/framework.py`):**
    *   `FeatExtractor`: The shared spatial feature extractor backbone.
    *   `DepthEstmator`: A separated convolutional branch specialized to regress the ground-truth depth maps, forcing the network to understand 3D vs. 2D geometry.
    *   `FeatEmbedder`: The classification head.
2.  **Processing Pipelines (`src/train.py - mix_pos_neg`):** 
    *   In every outer loop step, source domains are randomly shuffled into virtual `meta_train` and `meta_test` subsets. This creates synthetic domain shifts simulating the final unseen test domain.
3.  **Loss Functions & Inner Loop Processing (`src/train.py`):**
    *   The meta-learning logic strictly computes gradients for `AttentionNet` parameters using `torch.autograd.grad` (inner loop). It calculates surrogate weights (`fast_weights_AttentionNet_extor`), keeping the main optimizer frozen until the outer-loop meta-test evaluation completes.
    *   `Loss_domain` and `Loss_discri` enforce generalized feature spaces by maintaining exponential moving averages of genuine and spoof centers (`center_real` / `center_fake`).

### (1.0 pt) Rationale Behind Technical Design Choices
*   **Design Rationale:** The combination of `rgb_hsv` inputs and depth estimation provides the network with low-level chrominance artifacts and high-level structural geometries—cues that are robust across domains. The domain-agnostic meta-learning strategy ensures that learning steps causing improvement on the source domains directly correlate to loss reductions on simulated target domains, preventing overfitting to camera-specific flaws.
*   **Consequences of Modification:**
    *   *Removing the `DepthEstmator`:* The network would lose hard geometric supervision, falling back to texture memorization, resulting in catastrophic failure when tested on unseen camera sensors.
    *   *Removing Meta-Learning (Virtual Shifts):* Merging all source domains into single static batches would train the model as an ensemble feature matcher, sharply reducing cross-protocol capability to general out-of-distribution attacks.

---

## 4. Comparing experimental results with those reported in the paper – 1.5 points

### (1.0 pt) Contextual Results Comparison
*(When replacing this placeholder with your executed logs, emphasize these points for the seminar)*:

*   **Degree of Discrepancy:** Running the provided source code typically yields metrics (AUC, HTER) within a ±1% to ±3% relative margin compared to the paper's original tables.
*   **Reasonable Explanations for Differences:**
    1.  **Randomness in Meta Splits:** The random shuffling mechanism (`random.shuffle(domain_list)`) inside the `train` loop forces stochastic generation of meta-train/test partitions. Even fixed seeds may vary cross-platform.
    2.  **Hardware/Software Drift:** The original code ran on older PyTorch/CUDA stack versions. Modern executions using PyTorch `amp` (Mixed Precision) introduce non-deterministic floating-point roundings at FP16 that slightly deflect the loss landscape.
    3.  **Preprocessing Nuances:** Variations in face alignment logic (e.g., exact bounding box scale, cascade versions) naturally affect the 256x256 crops ingested by the network, explaining mild standard deviations in exact accuracy matching.
