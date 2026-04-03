<div align="center">

# AlloyVAE: A generative model for complex probabilistic field-to-field relationships in alloys


[![Paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2604.02281)

Ningyu Yan¹, Zhuocheng Xie², Kai Guo³, Yejun Gu³,⁴,\*, Huajian Gao⁵,\*, Yang Xiang¹,⁶,\*

<div align="center">
  <p>1. Department of Mathematics, The Hong Kong University of Science and Technology, Clear Water Bay, Kowloon, Hong Kong, China</p>
  <p>2. Department of Materials Science & Engineering, University of Toronto, Ontario, Canada M5S 3E4</p>
  <p>3. Institute of High Performance Computing, Agency for Science, Technology and Research, Singapore 138632</p>
  <p>4. Department of Mechanical Engineering, Whiting School of Engineering, Johns Hopkins University, Baltimore, MD, USA 21218</p>
  <p>5. Mechano-X Institute, Applied Mechanics Laboratory, Department of Engineering Mechanics, Tsinghua University, Beijing, China 100084</p>
  <p>6. HKUST Shenzhen–Hong Kong Collaborative Innovation Research Institute, Shenzhen, China</p>
</div>


\* *Corresponding authors*

</div>

---

## Abstract

<div align="justify" style="font-size: 0.95em; text-align: justify; line-height: 1.6;">

The inherent compositional heterogeneity of multi-principal element alloys (MPEAs) gives rise to complex, spatially varying mechanical fields that cannot be uniquely determined from coarse-grained composition descriptors. This non-uniqueness introduces intrinsically probabilistic structure–property relationships, posing a fundamental challenge to conventional deterministic modeling and machine learning approaches that collapse such mappings into average predictions. Here, we present AlloyVAE, a physics-informed generative framework that learns the full conditional distribution of mechanical fields from microstructural inputs. Built upon a conditional variational autoencoder architecture, the model incorporates learned smoothing operators to enhance functional regularity and a self-consistency mechanism to enforce physical plausibility. Trained on atomistic simulation data, AlloyVAE accurately predicts distributions of residual stress fields from composition and short-range order, and enables the generation of multiple physically consistent realizations under identical input conditions. Beyond forward prediction, the framework supports inverse design by optimizing composition fields to achieve targeted mechanical responses, and is extensible to coupled mappings involving eigenstrain. By capturing one-to-many structure–property relationships in heterogeneous materials, this work establishes a probabilistic paradigm for materials modeling and design, providing a scalable alternative to conventional simulations for navigating high-dimensional compositional spaces.

</div>

![Pipeline](assets/general_framework.jpg) <!-- 你可以把图片放在 assets 文件夹下 -->

# Overview

This repository contains a three-stage workflow for tungsten-carbide microstructure learning and design:

1. `dataset/`: convert raw MD simulation outputs into block-averaged tensors and normalized training arrays.
2. `model_training/`: define and train a conditional variational autoencoder (CVAE) that maps microstructure descriptors to stress fields.
3. `material_design/`: optimize concentration fields with a trained CVAE checkpoint for inverse material design.

## Recommended Conda Environment

Python `3.11` is recommended. Avoid `Python 3.13` for this project because it is more likely to cause package compatibility issues in Jupyter and PyTorch-based workflows.

```bash
conda create -n cvae-wc python=3.11 -y
conda activate cvae-wc
```

### Core packages

These are the minimum packages needed for the main GitHub workflow:

```bash
pip install numpy scikit-learn
pip install torch
```

If you want notebook-based inspection and visualization, also install:

```bash
pip install matplotlib seaborn umap-learn jupyter ipykernel
```

### GPU note

Training and optimization are much faster on GPU. Install a CUDA-enabled PyTorch build that matches your machine if GPU acceleration is needed. If not, the CPU build of PyTorch also works for basic testing.

## Folder Overview

### `dataset/`

This folder prepares the learning dataset from MD simulations.

- [dataprocess.py](/d:/plot/code_release/dataset/dataprocess.py) reads raw atomistic dump files in `MD_sim_examples/`, bins atoms into 3D blocks, and computes block-wise concentration, Warren-Cowley descriptors, and stress averages.
- [normalization.py](/d:/plot/code_release/dataset/normalization.py) merges block-averaged `.npy` files, downsamples from the original resolution to `4 x 4 x 4`, applies `MinMaxScaler`, shuffles samples, and produces the final training arrays:
  - `X_WC_3500.npy`
  - `Y_WC_3500.npy`

In short, `dataset/` converts raw simulation outputs into normalized tensors ready for model training.

### `model_training/`

This folder contains the CVAE model and training scripts.

- [CVAEWC.py](/d:/plot/code_release/model_training/CVAEWC.py) defines the main CVAE architecture:
  - 3D convolutional encoder for stress fields
  - smoothing/projection modules for concentration and SRO/WC conditions
  - latent reparameterization
  - decoder that reconstructs the stress field from latent variables and learned condition embeddings
- [dataset.py](/d:/plot/code_release/model_training/dataset.py) provides a lightweight PyTorch `Dataset` wrapper.
- [trainAlloyVAE.py](/d:/plot/code_release/model_training/trainAlloyVAE.py) loads `X_WC_3500.npy` and `Y_WC_3500.npy`, splits training data, trains the CVAE, stores loss curves, and saves checkpoints.
- [checkAlloyVAE.ipynb](/d:/plot/code_release/model_training/checkAlloyVAE.ipynb) is an analysis notebook for model evaluation, latent-space inspection, and prediction visualization.

In short, `model_training/` turns normalized dataset tensors into a trained generative surrogate model.

### `material_design/`

This folder performs inverse design using a trained checkpoint.

- [optimize.py](/d:/plot/code_release/material_design/optimize.py) loads a trained CVAE model, initializes candidate concentration fields and latent noise, then iteratively optimizes them to improve a target stress-related objective while keeping the generated structure self-consistent.

In short, `material_design/` uses the trained CVAE as a design engine rather than only a predictor.

## Suggested Workflow

1. Prepare or verify the raw block-averaged data in `dataset/block_averaged_data/`.
2. Run [normalization.py](/d:/plot/code_release/dataset/normalization.py) to generate `dataset/X_WC_3500.npy` and `dataset/Y_WC_3500.npy`.
3. Run [trainAlloyVAE.py](/d:/plot/code_release/model_training/trainAlloyVAE.py) to train the CVAE and save checkpoints.
4. Use [checkAlloyVAE.ipynb](/d:/plot/code_release/model_training/checkAlloyVAE.ipynb) for evaluation and visualization.
5. Run [optimize.py](/d:/plot/code_release/material_design/optimize.py) for inverse material design with a trained checkpoint.

## Important Notes

- Several non-GitHub scripts still contain machine-specific absolute paths such as `/home/...`. Update them before running on a new machine.
- The main public entry points are now `normalization.py`, `trainAlloyVAE.py`, and `optimize.py`.
- The notebook and optimization scripts assume that training checkpoints already exist.




