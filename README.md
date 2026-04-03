<div align="center">

# AlloyVAE: A generative model for complex probabilistic field-to-field relationships in alloys


[![Paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2604.02281)

Ningyu Yan¹, Zhuocheng Xie², Kai Guo³, Yejun Gu³,⁴,* Huajian Gao⁵,* Yang Xiang**¹,⁶,*


1. Department of Mathematics, The Hong Kong University of Science and Technology, Clear Water Bay, Kowloon, Hong Kong, China
2. Department of Materials Science & Engineering, University of Toronto, Ontario, Canada M5S 3E4
3. Institute of High Performance Computing, Agency for Science, Technology and Research, Singapore 138632
4. Department of Mechanical Engineering, Whiting School of Engineering, Johns Hopkins University, Baltimore, MD, USA 21218
5. Mechano-X Institute, Applied Mechanics Laboratory, Department of Engineering Mechanics, Tsinghua University, Beijing, China 100084
6. HKUST Shenzhen–Hong Kong Collaborative Innovation Research Institute, Shenzhen, China


\* *Corresponding authors*

</div>

---

## Abstract

The inherent compositional heterogeneity of multi-principal element alloys (MPEAs) gives rise to complex, spatially varying mechanical fields that cannot be uniquely determined from coarse-grained composition descriptors. This non-uniqueness introduces intrinsically probabilistic structure–property relationships, posing a fundamental challenge to conventional deterministic modeling and machine learning approaches that collapse such mappings into average predictions. Here, we present AlloyVAE, a physics-informed generative framework that learns the full conditional distribution of mechanical fields from microstructural inputs. Built upon a conditional variational autoencoder architecture, the model incorporates learned smoothing operators to enhance functional regularity and a self-consistency mechanism to enforce physical plausibility. Trained on atomistic simulation data, AlloyVAE accurately predicts distributions of residual stress fields from composition and short-range order, and enables the generation of multiple physically consistent realizations under identical input conditions. Beyond forward prediction, the framework supports inverse design by optimizing composition fields to achieve targeted mechanical responses, and is extensible to coupled mappings involving eigenstrain. By capturing one-to-many structure–property relationships in heterogeneous materials, this work establishes a probabilistic paradigm for materials modeling and design, providing a scalable alternative to conventional simulations for navigating high-dimensional compositional spaces.

![Pipeline](assets/Fig1_new.svg) <!-- 你可以把图片放在 assets 文件夹下 -->

---

The code will be released after the paper is accepted.

<!-- ## Features

- ✅ **Dense Matching**: Robust feature matching across frames.
- ✅ **Multi-View Foundation Models**: Leverage pre-trained models for better generalization.
- ✅ **Gaussian Splatting**: High-quality static scene reconstruction.
- ✅ **Real-Time Tracking**: Efficient pose estimation via bundle adjustment.

---

## Installation

```bash
git clone https://github.com/yourusername/M3.git
cd M3
pip install -r requirements.txt -->
