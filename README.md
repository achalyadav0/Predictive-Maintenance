# CNN-Based Fault Classification from Multi-Axis Vibration Spectrograms in Brownfield CNC Milling under Severe Class Imbalance

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end deep learning framework for predictive maintenance and real-time fault detection in industrial CNC milling machines operating under extreme class imbalance (96:4 normal-to-anomalous ratio).

---

## 📌 Abstract

Unplanned tool and process failures in Computer Numerical Control (CNC) milling cause costly scrap parts and downtime. This project implements a **multi-axis STFT spectrogram CNN pipeline** that converts tri-axial ($X, Y, Z$) accelerometer signals into Short-Time Fourier Transform spectrograms, fusing them as 3-channel tensors (similar to RGB images) for binary fault classification. 

Using Bayesian hyperparameter optimization targeting **Precision-Recall AUC (PR-AUC)** rather than standard accuracy or ROC-AUC, test performance improves from **0.483** (untuned baseline) to **0.937** (tuned SGD model) while maintaining a lightweight model footprint (<300K parameters).

---

## 🏗️ Architecture & Pipeline
Tri-Axial Sensor Signal (X, Y, Z @ 2 kHz)──►
Mean-Centering & Standardization
──►
Short-Time Fourier Transform (STFT)
(Hann window, NFFT=256, Overlap=128)
──►
Power Spectrogram in dB [-80, 0]
──►
3-Channel Tensor Stacking (129 x 306 x 3)
──►
CNN Architecture (Conv + BatchNorm + MaxPool + Dropout)
──►
Global Average Pooling ──► Dense Layers ──► Sigmoid Output

---

## 📊 Dataset Details

We evaluate on the brownfield CNC milling vibration benchmark dataset collected across 3 real-world industrial machines (M01–M03):

* **Total Samples:** 1,702 labeled recordings
* **Class Distribution:** 1,632 Normal (95.9%) vs. 70 Anomalous (4.1%)
* **Imbalance Ratio:** ~23:1 (Severe skew)
* **Sampling Rate:** 2 kHz tri-axial acceleration

---

## Experimental Results

Evaluated using a 65/35 stratified train/test split. All metrics are computed directly from predicted probabilities on held-out test data.

| Model Variant | Optimizer | Validation Strategy | Test PR-AUC 📈 | Test ROC-AUC 🎯 |
| :--- | :--- | :--- | :---: | :---: |
| Untuned Baseline | Adam | Fixed Architecture | 0.483 | 0.914 |
| Untuned Baseline | SGD + Momentum | Fixed Architecture | 0.513 | 0.923 |
| **Bayesian-Tuned** | Adam | Keras Tuner (PR-AUC) | 0.877 | 0.982 |
| **Bayesian-Tuned (Best)** | **SGD + Momentum** | **Keras Tuner (PR-AUC)** | **0.937** | **0.994** |

> **Key Insight:** Standard ROC-AUC produces overly optimistic metrics (~0.92) on untuned models despite poor minority-class detection. Targeting **PR-AUC** during Bayesian optimization doubled model precision-recall performance.



