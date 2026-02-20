# Web-Supervised-Monocular-Depth-Estimation


### Computer Vision Course Project – Fall 2025

---

# Project Overview

This project focuses on **monocular relative depth estimation** using deep learning. The objective is to implement and analyze the method proposed in the paper:

> *“Monocular Relative Depth Perception with Web Stereo Data Supervision”*

The goal is to train a neural network that predicts **relative depth ordering** from a single RGB image.

Unlike absolute depth estimation, relative depth focuses on determining whether one pixel (or region) is closer or farther than another — a ranking-based formulation.

---

# Project Phases

This project was developed in two main milestones:

---

## Phase 1 – Paper Analysis and Preparation

During Phase 1, we carefully studied the research paper and identified several technical ambiguities that required clarification .

### Paper Understanding

The paper proposes:

* A **ResNet-based encoder**
* Residual modules
* Multi-scale feature fusion
* A ranking-based loss function

However, several critical implementation details were missing:

#### Missing Architectural Details

* Exact configuration of residual blocks
* Depth and layout of modules
* Multi-scale fusion mechanism
* Decoder / upsampling structure
* Modifications to the base ResNet backbone 

This made it difficult to directly reconstruct the model.

---

### Loss Function Challenges

The paper introduces a **ranking loss**, which was initially difficult to interpret because:

* The notation was ambiguous
* Indexing structure was unclear
* The intuition behind ranking comparisons was not well explained 

Since our team was not previously familiar with ranking losses, we needed external clarification.

---

### Dataset Exploration

We downloaded and analyzed the dataset referenced in the paper. Key observations:

* Highly variable image resolutions
* Mixed horizontal and vertical orientations
* Large differences in aspect ratios 

This required careful preprocessing strategies:

* Resizing or padding
* Aspect-ratio preservation
* Avoiding geometric distortion

---

### External Guidance

To resolve uncertainties:

* We held meetings with our professor to clarify:

  * ResNet integration
  * Multi-scale fusion structure
  * Ranking loss intuition 

* We attempted to contact the paper’s authors for additional resources (no response received at the time) 

---

### Related Work – MiDaS

We explored the **MiDaS** repository, which uses:

* ResNet backbone
* Multi-scale feature fusion

Although it does not directly reference our paper, its structure is highly similar and served as implementation guidance .

---

## Phase 2 – Implementation and Initial Experiments

In Phase 2, we implemented and trained the model .

---

### Architecture Implementation

After clarification sessions, we:

* Implemented the ResNet-based encoder
* Designed fusion modules inspired by MiDaS
* Built a depth prediction head
* Implemented the ranking loss

---

### Training Strategy

We trained the network in two stages:

#### Stage 1 – Sanity Check (MAE Loss)

To ensure the pipeline worked correctly:

* Used Mean Absolute Error (MAE)
* Trained for 1 epoch
* Verified forward/backward pass correctness 

This was not the final objective, but a debugging step.

---

#### Stage 2 – Ranking Loss

After clarification, we implemented the ranking loss described in the paper and trained again for 1 epoch .

### 🔍 Observed Results

From the qualitative results shown in Phase 2:

* MAE-trained model produced weaker depth structure
* Ranking loss significantly improved relative ordering consistency 

This confirmed that ranking supervision is crucial for this task.

---

# Technical Details

## Model Architecture

High-level structure:

```
Input Image
      ↓
ResNet Encoder
      ↓
Residual Blocks
      ↓
Multi-scale Feature Fusion
      ↓
Upsampling / Decoder
      ↓
Depth Prediction Map
```

### Key Components

### ResNet Backbone

* Extracts hierarchical features
* Provides multi-resolution feature maps

### Multi-Scale Fusion

* Combines features from different resolutions
* Enhances spatial detail recovery

### Ranking Loss

Instead of minimizing pixel-wise depth error, the model learns:

> Given two pixels (i, j), predict whether depth_i > depth_j

This converts depth prediction into a **pairwise ranking problem**.

---

# Evaluation Plan

For future evaluation, we plan to test on:

* **DIW (Depth in the Wild)**
* **NYUDv2**

These datasets allow testing on:

* Indoor scenes
* Outdoor scenes 

---

# Training Configuration

Current setup:

* Optimizer: (default PyTorch configuration)
* Loss:

  * MAE (debugging)
  * Ranking loss (final objective)
* Epochs trained so far: 1
* Target: ~120 epochs (based on MiDaS paper reference) 

---

# Hardware Limitations

A major constraint:

* No access to powerful GPU hardware
* Colab / Kaggle sessions are limited
* Long training (120 epochs) may not be feasible 

Possible workaround:

* Train fewer epochs
* Save checkpoints frequently
* Use reduced dataset subsets

---

# Project Structure (Example)

```
.
├── data/
├── models/
│   ├── resnet_encoder.py
│   ├── fusion_module.py
│   ├── decoder.py
│   └── full_model.py
├── losses/
│   └── ranking_loss.py
├── train.py
├── evaluate.py
├── utils.py
└── README.md
```

---

# How to Run

### Install Requirements

```bash
pip install torch torchvision numpy matplotlib tqdm
```

### Train

```bash
python train.py --epochs 50 --loss ranking
```

### Evaluate

```bash
python evaluate.py --dataset DIW
```

---

# Key Learnings

* Ranking loss is more suitable than pixel-wise regression for relative depth
* Multi-scale fusion is critical for spatial precision
* Paper reproduction often requires external clarification
* Dataset preprocessing is crucial in depth estimation

---

# Future Work

* Full 120-epoch training
* Quantitative evaluation on DIW and NYUDv2
* Hyperparameter tuning
* Data augmentation strategies
* Performance optimization

---

# References

* MiDaS repository 
* DIW Dataset 
* NYUDv2 Dataset 

---

# Conclusion

This project aims to reproduce and understand a relative depth estimation framework using ranking-based supervision. Despite architectural ambiguities and hardware limitations, we have successfully:

* Implemented the core architecture
* Understood and implemented ranking loss
* Observed promising qualitative improvements


