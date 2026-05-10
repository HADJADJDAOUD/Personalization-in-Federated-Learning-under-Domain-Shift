# Personalization in Federated Learning under Domain Shift

**Course:** Advanced Machine Learning — Spring 2026  
**Module:** Federated Learning & Transfer Learning

## Project Overview

This project addresses the domain shift problem in Federated Learning (FL), where hospitals exhibit biased and heterogeneous data distributions. We implement and compare:

- **FedAvg** (global baseline)
- **Local-only model** (purely local baseline)
- **pFedMe** (personalized FL with theoretical guarantees)
- Optionally: **Per-FedAvg**

We use the **FLamby Heart Disease** dataset (multi-center tabular medical data) to simulate a realistic cross-silo federated healthcare scenario.

## Repository Structure

```
project/
├── data/               # Raw or preprocessed datasets (or download scripts)
├── notebooks/          # Exploratory and experimental Jupyter notebooks
├── src/                # Python source code (models, trainers, utils)
├── reports/            # Weekly and final PDF reports
├── figures/            # Plots and diagrams
├── models/             # Saved checkpoints and configs
├── experiments/        # Logs, MLflow outputs, configs
├── README.md
└── requirements.txt
```

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd team-project
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

The FLamby Heart Disease dataset can be accessed at:  
https://fedbiomed.org/latest/user-guide/datasets/

Follow the FLamby installation instructions:
```bash
pip install flwr flwr-datasets
```

## Reproducing Results

> *(To be completed in Weeks 2–4 as experiments are finalized)*

1. **Baseline (FedAvg):** `python src/train_fedavg.py`
2. **Personalized (pFedMe):** `python src/train_pfedme.py`
3. **Evaluation:** `python src/evaluate.py`

## Key References

- Dinh, C. T., et al. *pFedMe: Personalized Federated Learning with Moreau Envelopes.* NeurIPS 2020.
- McMahan, B., et al. *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS 2017.
- FLamby: https://github.com/owkin/FLamby

## Team

| Role | Responsibility |
|------|---------------|
| ML Engineer | Model implementation & training |
| Data Engineer | Data preprocessing & pipelines |
| Research Lead | Literature review & SOTA framing |
| Project Manager | Planning & communication |
| Report Lead | Report structure, figures, slides |
