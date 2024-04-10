# Price Process Modeling with Hawkes Processes

This repository contains code and documentation related to modeling asset price processes using mutually exciting Hawkes processes. The primary goal of this project is to investigate and model the microstructure noise observed in high-frequency financial data.

## Overview

The project consists of two main parts:

1. **Estimating Volatility at Short Timescales**:
   - Implementation of various volatility estimators, including naive high-frequency estimators, improved low-frequency estimators, and a combined approach.
   - Assessment of estimator performance using synthetic data generated from the Heston stochastic volatility model.
   - Application of estimators to tick-by-tick forex data to analyze volatility estimation at various sampling frequencies.

2. **Modeling the Price by a Hawkes Process**:
   - Introduction of mutually exciting Hawkes processes to model asset price movements.
   - Estimation of volatility and explanation of the signature plot using Hawkes process models.
   - Fitting the Hawkes process model to real tick data (e.g., EUR/USD forex pair) and comparison with theoretical predictions.

## Directory Structure

- `utils/`: Contains Python scripts implementing different volatility estimators.
- `notebooks/`: Includes Jupyter notebooks or scripts for modeling asset prices with Hawkes processes.

## Dependencies

- Python (>=3.6)
- NumPy, pandas, matplotlib, seaborn
- Jupyter Notebook (for running model analysis notebooks)
