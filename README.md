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

- `estimators/`: Contains Python scripts implementing different volatility estimators.
- `models/`: Includes Jupyter notebooks or scripts for modeling asset prices with Hawkes processes.
- `data/`: Sample datasets used for estimation and modeling purposes.
- `docs/`: Documentation files, including theoretical explanations and results.
- `tests/`: Unit tests for verifying the correctness of implemented estimators and models.

## Dependencies

- Python (>=3.6)
- NumPy, pandas, matplotlib, seaborn
- Jupyter Notebook (for running model analysis notebooks)
- pytest (for running tests)

## Usage

To run the volatility estimators or model analysis scripts, follow these steps:

1. Clone this repository to your local machine.
2. Navigate to the relevant directory (`estimators/` or `models/`).
3. Open and run the Python scripts or Jupyter notebooks using an appropriate Python environment.

For example:

```bash
git clone https://github.com/your-username/price-process-hawkes.git
cd price-process-hawkes/models
jupyter notebook Price_Model_Analysis.ipynb
