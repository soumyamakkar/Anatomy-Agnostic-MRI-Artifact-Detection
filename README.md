# Towards Anatomy-Agnostic MRI Artifact Detection Using Unsupervised Learning

## Project Information

**Author:** Soumya Makkar
**Roll Number:** 2210992401
**Project Type:** Research Paper
**Team:** Individual Research Project

---

## Conference Submission Status

**Status:** Submitted to Conference
**Conference:** 2026 IEEE 5th World Conference on Applied Intelligence and Computing (AIC2026)
**Paper ID:** 3595
**Submission Date:** May 16, 2026
**Primary Subject Area:** Machine Vision
**Secondary Subject Area:** Computational Intelligence

---

## Abstract

Artifacts in Magnetic Resonance Imaging (MRI) are distortions or anomalies that can arise from several factors including patient movement, hardware issues, physical tissue properties as well as acquisition related factors. These errors often render the scan useless and can sometimes be even confused with pathology. Recent advancements in medical imaging deep learning have led to significant progress in several medical imaging tasks, such as image classification, object detection, segmentation, registration, etc. Deep neural networks are known to be sensitive to image quality, leading to inaccurate results when fed with slices corrupted by artifacts.

Existing artifact detection methods are built primarily on supervised methods, that require an enormous amount of well-annotated data. Such labelled datasets are time-consuming, subjective, scarce, and can render a lot of unlabeled data unutilized. Additionally, a lot of methods focus purely on a single sequence or anatomy, which can make models unable to explicitly disentangle anatomical features from artifacts.

In accordance with these limitations, we propose a systematic evaluation of supervised, unsupervised, and self-supervised learning paradigms for MRI artifact detection under a unified experimental framework. We explore supervised baselines such as ResNet-18 and EfficientNet-B0, alongside unsupervised anomaly detection methods (e.g., autoencoders and PatchCore) and self-supervised representation learning approaches including SimCLR, DINO, and masked autoencoders. Our experiments are conducted on multi-cohort open MRI datasets to promote generalizability in acquisition settings and anatomy. Our findings suggest that learning anatomy-independent representations through label-free paradigms offers a scalable and practical pathway toward automated MRI quality control without requiring artifact-specific annotations.

---

## Project Overview

This research project presents a comprehensive comparison of **6 anomaly detection methods** for identifying motion artifacts in MRI scans across different anatomical regions (brain and knee). The key innovation lies in training models exclusively on **clean MRI slices** and detecting artifacts as out-of-distribution anomalies at test time—eliminating the need for labeled artifact data during training.

### Key Contributions

1. **Anatomy-Agnostic Framework**: Models trained on combined brain and knee data to learn anatomy-independent artifact representations
2. **Multi-Paradigm Evaluation**: Systematic comparison of supervised, unsupervised, and self-supervised approaches
3. **Cross-Anatomy Generalization**: Testing models trained on one anatomy and evaluated on another
4. **Reproducible Pipeline**: All experiments reproducible with SEED=42 and deterministic sampling
5. **Open Datasets**: Leveraging publicly available IXI, FastMRI, MR-ART, and KMAR datasets

---

## Methods Evaluated

| Method | Paradigm | Architecture | Anomaly Scoring Mechanism |
|--------|----------|--------------|---------------------------|
| **Supervised Baseline** | Supervised | ResNet-18 | Binary classification probability |
| **SimCLR** | Self-supervised | ResNet-50 + contrastive learning | k-NN distance in feature space (k=5) |
| **MAE (Masked Autoencoder)** | Self-supervised | ViT-Small/16 | Reconstruction error + k-NN |
| **DINO** | Self-supervised | ViT-Small/16 (student-teacher) | k-NN on teacher features |
| **DAE (Denoising Autoencoder)** | Unsupervised | Convolutional encoder-decoder | Reconstruction error + k-NN |
| **PatchCore** | Unsupervised | WideResNet-50-2 (frozen backbone) | Distance to memory bank |

---

## Datasets

### Training Data
- **Brain MRI**: IXI Dataset (T1 and T2 weighted sequences)
- **Knee MRI**: FastMRI Dataset (multi-coil knee scans)

### Test Data (Artifact Detection)
- **MR-ART**: Real motion artifacts in brain MRI
- **KMAR**: Synthetic motion artifacts in knee MRI

### Training Variants
1. **Knee-only**: Models trained exclusively on FastMRI knee data
2. **Brain-only**: Models trained exclusively on IXI brain data
3. **Multi-anatomy**: Models trained on combined brain + knee data

---

## Experimental Design

### Workflow

```
1. Data Preprocessing
   ├── Convert DICOM/h5 to 192×192 .npy slices
   ├── Normalize pixel intensities
   └── Split into train/validation/test sets

2. Model Training (18 total experiments)
   ├── 6 methods × 3 anatomies
   ├── Train only on clean slices
   └── Save model checkpoints

3. Evaluation
   ├── Generate anomaly scores on test sets
   ├── Compute ROC-AUC and PR-AUC
   ├── Generate t-SNE visualizations
   └── Create confusion matrices
```

---

## Repository Structure

```
MRI-Artifact-Anomaly-Detection/
│
├── source-code/
│   ├── Preprocessing FastMRI knee.ipynb
│   ├── Preprocessing ixi brain.ipynb
│   ├── Artifact_preprocessing.ipynb
│   ├── SupervisedArtifactClassifier.ipynb
│   ├── SimCLR_MultiAnatomy.ipynb
│   ├── SimCLR_brain_only_train.ipynb
│   ├── simclr-all-anatomies-new.ipynb
│   ├── DINO_multianatomy.ipynb
│   ├── dino-ablation-all-anatomies.ipynb
│   ├── vitmae-multianatomy.ipynb
│   ├── DAE_multianatomy.ipynb
│   ├── patchcore-all-anatomies.ipynb
│   └── evaluation-mri-artifact.ipynb
│
├── report-and-ppt/
│   └── [Project documentation and presentation]
│
├── IPR-submission-proof/
│   └── [Conference submission confirmation]
│
└── README.md
```

---

## Key Findings

The research demonstrates that:

1. **Self-supervised methods** (especially DINO and SimCLR) can match or exceed supervised baselines without requiring artifact labels
2. **Multi-anatomy training** improves generalization across different anatomical regions
3. **Feature-based anomaly detection** (k-NN in learned feature spaces) is more robust than reconstruction-based methods
4. **Label-free paradigms** offer a scalable pathway for automated MRI quality control

---

## Requirements

### Dependencies
- Python 3.8+
- PyTorch 1.12+
- torchvision
- NumPy
- scikit-learn
- matplotlib
- seaborn
- tqdm
- PIL (Pillow)

### Hardware
- GPU: NVIDIA T4 or better (16GB+ VRAM recommended)
- RAM: 16GB minimum
- Storage: ~50GB for preprocessed datasets

---

## Usage

### 1. Data Preprocessing
```bash
# Preprocess brain MRI data
jupyter notebook "source-code/Preprocessing ixi brain.ipynb"

# Preprocess knee MRI data
jupyter notebook "source-code/Preprocessing FastMRI knee.ipynb"

# Preprocess artifact test sets
jupyter notebook "source-code/Artifact_preprocessing.ipynb"
```

### 2. Model Training
```bash
# Train supervised baseline
jupyter notebook "source-code/SupervisedArtifactClassifier.ipynb"

# Train SimCLR (multi-anatomy)
jupyter notebook "source-code/simclr-all-anatomies-new.ipynb"

# Train DINO (multi-anatomy)
jupyter notebook "source-code/dino-ablation-all-anatomies.ipynb"

# Train MAE (multi-anatomy)
jupyter notebook "source-code/vitmae-multianatomy.ipynb"

# Train DAE (multi-anatomy)
jupyter notebook "source-code/DAE_multianatomy.ipynb"

# Train PatchCore (multi-anatomy)
jupyter notebook "source-code/patchcore-all-anatomies.ipynb"
```

### 3. Evaluation
```bash
# Run unified evaluation pipeline
jupyter notebook "source-code/evaluation-mri-artifact.ipynb"
```

---

## Reproducibility

All experiments use:
- **Random Seed:** 42
- **Deterministic Operations:** Enabled where possible
- **Fixed Train/Val/Test Splits:** Consistent across all methods
- **Same Preprocessing Pipeline:** Standardized to 192×192 resolution

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{makkar2026anatomy,
  title={Towards Anatomy-Agnostic MRI Artifact Detection Using Unsupervised Learning},
  author={Makkar, Soumya},
  booktitle={2026 IEEE 5th World Conference on Applied Intelligence and Computing (AIC2026)},
  year={2026},
  organization={IEEE}
}
```

---

## Contact

**Soumya Makkar**
Roll Number: 2210992401
Email: soumya2401.be22@chitkara.edu.in

---

## License

This project is intended for academic and research purposes.

---

## Acknowledgments

This research leverages the following open datasets:
- **IXI Dataset**: Brain MRI imaging consortium
- **FastMRI**: NYU Langone Health and Facebook AI Research
- **MR-ART**: Motion artifact benchmark dataset
- **KMAR**: Knee motion artifact repository

Special thanks to the organizers of the 2026 IEEE 5th World Conference on Applied Intelligence and Computing (AIC2026).

---

**Last Updated:** May 16, 2026
