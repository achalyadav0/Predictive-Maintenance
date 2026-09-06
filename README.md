# CNN-Based Fault Classification from Multi-Axis Vibration Spectrograms in Brownfield CNC Milling

This repository contains a research-focused implementation of a predictive maintenance pipeline for CNC milling using vibration signals. The project converts tri-axial accelerometer data into time-frequency spectrograms and classifies each recording as normal or anomalous using a convolutional neural network.

The project is designed for academic and experimental use, especially under severe class imbalance, and is intended to study robust fault detection in brownfield industrial settings.

## Overview

The implementation follows the methodology described in the included paper, which is stored as [CNC_ML_paper.pdf](CNC_ML_paper.pdf). The workflow is:

1. Download the CNC milling vibration dataset from Kaggle
2. Load X, Y, and Z acceleration channels
3. Remove DC bias and standardize each axis
4. Compute STFT spectrograms
5. Convert each axis spectrogram to a normalized 2D representation
6. Stack the three axes into a 3-channel tensor
7. Train a CNN for binary classification
8. Use class weighting and PR-AUC-based optimization under severe imbalance

## Research Objective

The goal is to detect rare anomalous machine behavior in CNC milling from vibration data where the normal class dominates the dataset. In this benchmark, the class ratio is approximately 96:4, which makes standard accuracy misleading and motivates PR-AUC as the primary metric.

## Dataset

The project uses the brownfield CNC milling benchmark introduced for real industrial monitoring. The dataset contains records from three machines (M01 to M03) and is labeled as:

- good / normal
- bad / anomalous

The accessible subset used in this repo contains:

- 1,702 labeled recordings
- 1,632 normal samples
- 70 anomalous samples
- imbalance ratio of about 23:1
- 2 kHz tri-axial accelerometer sampling

## Methodology

The CNN pipeline uses:

- mean-centering and standardization per axis
- STFT with Hann window, 256-sample segment length, 128 overlap
- power spectrogram conversion in dB scale
- fixed-size resizing to 129 x 306 spectrograms
- stacking of X/Y/Z spectrograms as channels
- CNN with batch normalization, pooling, and dropout
- binary cross-entropy with class weighting
- Bayesian hyperparameter tuning using Keras Tuner
- PR-AUC as the main validation objective

## Key Findings

The repo reports the following test results on the held-out split described in the paper:

| Model Variant | Optimizer | Test PR-AUC | Test ROC-AUC |
| --- | --- | ---: | ---: |
| Untuned baseline | Adam | 0.483 | 0.914 |
| Untuned baseline | SGD + momentum | 0.513 | 0.923 |
| Bayesian tuned | Adam | 0.877 | 0.982 |
| Bayesian tuned | SGD + momentum | 0.937 | 0.994 |

The main insight is that PR-AUC is much more informative than ROC-AUC for this task because the minority class is rare and standard ROC metrics can look overly optimistic.

## Important limitation

This repo is a research prototype, not a production-ready deployment model.

The paper explicitly notes a major limitation:

- the evaluation uses a random stratified split
- the dataset is designed to expose cross-machine and cross-time drift
- therefore, the reported performance may overestimate real-world generalization

For future work, a drift-aware evaluation such as leave-one-machine-out or time-based split is recommended.

## Repository Structure

- [predictive_maintinance.py](predictive_maintinance.py): main implementation containing data loading, preprocessing, CNN training, tuning, and evaluation
- [README.md](README.md): project summary and research overview
- [CNC_ML_paper.pdf](CNC_ML_paper.pdf): paper describing the methodology and results

## Setup

This project depends on:

- Python 3.8+
- TensorFlow / Keras
- Keras Tuner
- NumPy
- pandas
- matplotlib
- SciPy
- scikit-learn
- kagglehub

Install dependencies using:

```bash
pip install tensorflow matplotlib keras-tuner scikit-learn scipy pandas numpy kagglehub
```

Then run the main script:

```bash
python predictive_maintinance.py
```

> Note: Kaggle dataset access may require Kaggle credentials configured in your environment.

## Usage Notes

This repository is intended for:

- experimentation
- academic research
- model exploration and benchmarking
- learning about spectrogram-based fault detection in industrial settings

It is not presented here as a fully packaged production system or deployment pipeline.

## Summary

This project demonstrates a strong research approach for predictive maintenance in CNC milling using vibration spectrograms and CNNs. It is especially relevant when dealing with rare machine faults and severe class imbalance.

The strongest research takeaway is that PR-AUC-guided tuning substantially improves anomaly detection performance compared with untuned models, while also highlighting the need for drift-aware evaluation before claiming deployment readiness.



