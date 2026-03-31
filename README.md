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

## Results

The framework's performance is evaluated on both synthetic (SynWoodScape) and real-world (WoodScape) datasets. The key results for Sim-to-Real transfer (training on synthetic, testing on real) are summarized below, highlighting the effectiveness of the compression bottleneck as a domain adaptation tool.

| Method | Bitrate (bpp) | Source mIoU (SynWoodScape) | Target mIoU (WoodScape) |
| :--- | :---: | :---: | :---: |
| Uncompressed BEV | -- | 65.4% | 22.1% |
| JPEG2000 + BEV | 0.15 | 58.2% | 19.5% |
| BEV-JPC | 0.12 | 66.1% | 28.4% |
| **BEV-JPC+ (Full)** | **0.12** | **63.5%** | **41.5%** |

