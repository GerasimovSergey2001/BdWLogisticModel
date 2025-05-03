# Survival Analysis with Extended Beta-Discrete-Weibull-Logistic Model

## Overview

This project implements a custom survival analysis model based on the Beta-Discrete-Weibull (BdW) distribution and extends Beta-Logistic model, originally introduced by [Hubbard et al. (2021)](https://proceedings.mlr.press/v146/hubbard21a.html). Our extension incorporates covariate-dependent parameters through a hierarchical structure with duration dependence, enabling more expressive, personalized modeling of time-to-event data. The model supports both censored and uncensored data and is suitable for applications such as personalized risk modeling, user behavior forecasting, and event prediction.

## Key Features
Extension of Hubbard et al. (2021): Builds the BdW Logistic model, allowing for duration dependence modeling.


- **Custom Loss Function**  
  `BdWLoss` computes the negative log-likelihood loss for the Beta-Discrete-Weibull survival model, handling both censored and uncensored samples.

- **Flexible Model Architecture**  
  The `SurvivalModel` class supports different survival distributions:
  - Weibull
  - Beta-Logistic

- **Gamma Parameter Handling**  
  The model supports multiple strategies for handling the γ (gamma) shape parameter:
  - `constant`: A single gamma value for all samples.
  - `partitioned`: Gamma varies across defined column partitions.
  - `individual`: A unique gamma value for each sample.

- **Survival Function Calculation**  
  Includes methods for computing the survival function \( S(t) \) for any time point \( t \), based on the model's learned parameters.

---

## Requirements

- Python 3.x  
- [PyTorch](https://pytorch.org/)  
- NumPy  
- Matplotlib  
- scikit-learn  
- IPython *(for interactive plots in Jupyter Notebooks)*

---

## Model Description

The **Beta-Discrete-Weibull (BdW) Logistic** model integrates the flexibility of the Beta distribution with a discretized Weibull-like time-to-event distribution. It provides a likelihood-based approach to modeling survival data and supports censoring natively.

### Components

- **`BdWLoss`**
  - Computes the negative log-likelihood for both censored and uncensored events.
  - Supports batching and differentiable training via PyTorch.

- **`SurvivalModel`**
  - Supports multiple survival distributions (Weibull, Beta-Logistic).
  - Includes methods for:
    - Model training
    - Evaluation and loss tracking
    - Saving/loading model checkpoints
    - Computing the survival function
    - Plotting survival curves

---

## Example Use Cases

- **Medical Prognosis**: Predicting patient survival time post-diagnosis.
- **Churn Analysis**: Estimating customer lifetime for subscription services.
- **Industrial Maintenance**: Forecasting time to equipment failure.

---

## Theoretical Inference
- []
---
