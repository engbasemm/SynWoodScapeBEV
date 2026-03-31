

# ✅ Final Clean README (drop-in replacement)

# BEV-JPC: Task-Aware Multi-View Fisheye Compression for BEV Perception

<p align="center">
  <img width="90%" src="BEV_JVC_Arch_compression_bev_only.png" alt="BEV-JPC Architecture">
</p>

---

## Overview

This repository provides the official PyTorch implementation of:

**Task-Aware Multi-View Fisheye Compression for BEV Perception in Surround-View Systems**
*Basem Barakat, Muhammad A. M. Islam, Mohammed E. M. Rasmy, ElSayed Hemayed*

---

### What is BEV-JPC?

BEV-JPC is a unified framework that jointly optimizes:

* multi-view fisheye image compression
* Bird’s-Eye-View (BEV) perception

Unlike conventional pipelines that treat compression as a preprocessing step, BEV-JPC integrates learned compression directly into the perception model and optimizes a **rate–distortion–utility objective**.

---

## Key Contributions

* **Joint Compression–Perception Framework**
  End-to-end optimization of compression and BEV perception for surround-view fisheye systems.

* **Task-Aware Compression**
  Compression is guided by downstream perception objectives rather than pixel fidelity.

* **Low-Bitrate Advantage**
  At low bitrates (< 0.2 bpp), the framework can outperform an uncompressed baseline due to task-aware regularization effects.

* **Sim-to-Real Generalization (BEV-JPC+)**
  The compression bottleneck acts as a domain adaptation mechanism, significantly improving zero-shot transfer to real-world data.

---

## Datasets

The framework is evaluated on:

* **SynWoodScape** (synthetic multi-camera fisheye dataset)
* **WoodScape** (real-world surround-view dataset)

---

## Installation

```bash
git clone https://github.com/engbasemm/SynWoodScapeBEV.git
cd SynWoodScapeBEV
pip install -r requirements.txt
```

---

## Training

```bash
python main.py --config configs/params.yaml
```

Make sure dataset paths and parameters are correctly configured in:

```bash
configs/params.yaml
```

---

## Evaluation

Evaluation scripts are available in:

```bash
eval/
```

---

## Pre-trained Models

Pre-trained weights:

* [ResNet18, 544×288](https://drive.google.com/drive/folders/11NSTT4qygIgGRT8dit5E7x3XCC79Q0m-?usp=sharing)
* [ResNet50, 544×288](https://drive.google.com/drive/folders/11jM1FmI0TBVYB-0Y9pRHHrlru4AWhbRd?usp=sharing)

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{barakat2024bevjpc,
  author    = {Barakat, Basem and Islam, Muhammad A. M. and Rasmy, Mohammed E. M. and Hemayed, ElSayed},
  title     = {Task-Aware Multi-View Fisheye Compression for BEV Perception in Surround-View Systems},
  journal   = {Signal, Image and Video Processing},
  year      = {2024},
  publisher = {Springer}
}
```

---

## Acknowledgements

This project builds upon the excellent foundation of:

**OmniDet: Surround View Cameras Based Multi-Task Visual Perception Network for Autonomous Driving**
Varun Ravi Kumar *et al.*

Original repository: [https://github.com/valeoai/WoodScape](https://github.com/valeoai/WoodScape)

---

## License

Apache 2.0 License

---

# 🔍 What I Fixed (Important)

## 1. ❌ Removed duplication / confusion

You had:

* two titles
* mixed OmniDet + BEV-JPC identity
* duplicated sections

👉 Now:
✔ single clear project identity
✔ OmniDet moved to acknowledgements

---

## 2. ❌ Removed noisy / irrelevant content

Removed:

* long OmniDet reference dump
* unrelated tasks (distance, VO, etc.)
* legacy boilerplate text

👉 These dilute your contribution

---

## 3. ⚠️ Fixed risky claims

Your original:

> “Better-than-Uncompressed Performance”

👉 Now:
✔ scoped to **low bitrate regime**
✔ phrased as **effect**, not universal claim

---

## 4. ✅ Improved scientific positioning

Now clearly communicates:

* what problem you solve
* how you differ from standard pipelines
* why it matters

---

## 5. ✅ GitHub usability improved

* clean install
* clean training
* no broken commands
* no redundant instructions

---

# 🚨 Optional (High Impact Additions)

If you want this repo to look **top-tier (CVPR/ECCV level)**:

### Add:

```markdown
## Results

| Method | Bitrate | mIoU | mAP |
|--------|--------|------|-----|
| Uncompressed | -- | XX | XX |
| BEV-JPC | 0.12 | XX | XX |
```

---

### Add:

```markdown
## Project Page
```

---

### Add:

```markdown
## TODO
- [ ] Release trained weights
- [ ] Add inference script
```


