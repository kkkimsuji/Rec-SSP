# Rec-SSP: Balancing Long-Short-Term User Preferences via Multilevel Sequential Patterns for Review-Aware Recommendation
![Last Commit](https://img.shields.io/github/last-commit/kkkimsuji/Rec-SSP?style=flat-square)
[![Paper](https://img.shields.io/badge/MDPI_Electronics-Paper-blue)](https://doi.org/10.3390/electronics15040753)
[![DOI](https://img.shields.io/badge/DOI-10.3390/electronics15040753-red)](https://doi.org/10.3390/electronics15040753)


This repository contains the official implementation of the following paper:
> Jin, L., Li, X., **Kim, S.**, & Kim, J. (2026). Balancing Long–Short-Term User Preferences via Multilevel Sequential Patterns for Review-Aware Recommendation. Electronics, 15(4), 753.

## Overview
Rec-SSP (Recommender System emphasizing Short-term Preference using sequential patterns) is a novel review-aware recommendation framework designed to bridge the gap between a user’s stable historical interests and their rapidly evolving current intent. While traditional models often struggle to balance long-term data with fine-grained shifts, Rec-SSP explicitly models heterogeneous signals across multiple semantic levels to capture diverse cues such as emotional emphasis and category transitions. By employing a data-dependent Gated Fusion Mechanism, the model adaptively regulates the influence of historical stability versus recent dynamics based on the user's current context. Extensive evaluations on real-world datasets demonstrate that Rec-SSP significantly improves recommendation accuracy, consistently outperforming state-of-the-art baselines.

## Environment & Requirements

This project is implemented in **Python 3.8+**. To ensure reproducibility, please install the specific versions of the libraries listed below.

### 1. Key Dependencies
| Category | Library | Minimum Version | Description |
| :--- | :--- | :--- | :--- |
| **Deep Learning** | `TensorFlow` | `2.10.0` | Main model architecture & training |
| **NLP** | `Transformers` | `4.25.0` | BERT-based feature extraction |
| **NLP Backend** | `PyTorch` | `1.12.0` | Backend for HuggingFace Transformers |
| **Analysis** | `Pandas` | `1.5.0` | Data manipulation and storage |
| **Matrix** | `NumPy` | `1.23.0` | Efficient numerical operations |
| **ML Tools** | `scikit-learn` | `1.1.0` | Data splitting and metrics |

### 2. Utility Libraries
- `PyYAML` (>=6.0): For parsing the `config.yaml` file.
- `PyArrow` (>=10.0.0): For high-performance Parquet file handling.
- `tqdm` (>=4.64.0): For real-time progress bars during embedding extraction.

### 3. Installation
We recommend using a virtual environment. You can install all dependencies at once using the following command:

```bash
pip install -r requirements.txt
```
## Repository Structure

The repository is organized as follows to ensure a clear workflow from data preprocessing to model evaluation:

```text
Rec-SSP/
├── main.py               # Master controller for the full pipeline 
├── config.yaml           # Centralized hyperparameter configuration (K, Batch size, Learning rate) [cite: 374]
├── requirements.txt      # List of dependencies for environment setup
├── .gitignore            # Git ignore file (excludes .venv, large data, and caches)
├── README.md             # Project overview and documentation
│
├── model/
│   └── proposed.py       # Core Rec-SSP architecture (Long-term, Short-term, and Gated Fusion) [cite: 142]
│
├── src/
│   ├── bert.py           # Logic for extracting BERT [CLS] embeddings [cite: 175]
│   ├── data_processing.py# Data loading, 5-core filtering, and sequential feature generation [cite: 137]
│   └── train.py          # 7:1:2 Data splitting, training loop, and evaluation metrics [cite: 332]
│
└── data/                 # Dataset storage (Excluded from Git push) [cite: 328]
    ├── raw/              # Original Amazon/Yelp JSONL metadata and sample files [cite: 334]
    └── preprocessed/     # Generated pickle files for efficient model training
```
## Model Description

First, the model extracts long-term preferences by processing a user's aggregated historical review sets through a pretrained BERT model to capture stable, inherent tastes. Next, it models short-term preferences by identifying sequential patterns from recent interactions at both the review level (using BERT-based semantic flow) and the category level (using multi-hot attribute transitions). These multilevel sequences are processed through Gated Recurrent Units (GRU) to capture temporal dependencies and are further refined by a multi-head attention mechanism that emphasizes interactions most relevant to the target item characteristics. Finally, the model dynamically integrates these heterogeneous signals using a Gated Fusion Mechanism, which adaptively regulates the balance between historical stability and recent dynamics to predict the final preference rating via a multilayer perceptron (MLP) optimized by Mean Squared Error (MSE) loss.


<img width="3289" height="2036" alt="image" src="https://github.com/user-attachments/assets/25eb9f13-6d6b-4fcb-84a9-651278ab69ec" />


## Experimental Results

The table below summarizes the performance comparison between **Rec-SSP** and baseline models on datasets.

<table border="1">
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="2">Video Games</th>
      <th colspan="2">CDs and Vinyl</th>
      <th colspan="2">Musical Instruments</th>
      <th colspan="2">Yelp</th>
    </tr>
    <tr>
      <th>MAE</th>
      <th>RMSE</th>
      <th>MAE</th>
      <th>RMSE</th>
      <th>MAE</th>
      <th>RMSE</th>
      <th>MAE</th>
      <th>RMSE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>NCF</td>
      <td>0.957</td>
      <td>1.214</td>
      <td>0.747</td>
      <td>1.014</td>
      <td>0.838</td>
      <td>1.103</td>
      <td>0.866</td>
      <td>1.129</td>
    </tr>
    <tr>
      <td>D-attn</td>
      <td>0.864</td>
      <td>1.082</td>
      <td>0.733</td>
      <td>0.920</td>
      <td>0.790</td>
      <td>0.999</td>
      <td>0.829</td>
      <td>1.045</td>
    </tr>
    <tr>
      <td>DeepCONN</td>
      <td>0.801</td>
      <td>1.076</td>
      <td>0.616</td>
      <td>0.880</td>
      <td>0.717</td>
      <td>0.999</td>
      <td>0.803</td>
      <td>1.041</td>
    </tr>
    <tr>
      <td>NARRE</td>
      <td>0.798</td>
      <td>1.070</td>
      <td>0.594</td>
      <td>0.851</td>
      <td>0.705</td>
      <td>0.998</td>
      <td>0.802</td>
      <td>1.037</td>
    </tr>
    <tr>
      <td>RNS</td>
      <td>0.777</td>
      <td>1.065</td>
      <td>0.597</td>
      <td>0.863</td>
      <td>0.702</td>
      <td>0.989</td>
      <td>0.793</td>
      <td>1.034</td>
    </tr>
    <tr>
      <td>CCA</td>
      <td>0.711</td>
      <td>1.008</td>
      <td>0.572</td>
      <td>0.833</td>
      <td>0.674</td>
      <td>0.970</td>
      <td>0.767</td>
      <td>0.988</td>
    </tr>
    <tr style="background-color: #f2f2f2; font-weight: bold;">
      <td>Rec-SSP (Ours)</td>
      <td>0.679</td>
      <td>0.979</td>
      <td>0.537</td>
      <td>0.824</td>
      <td>0.621</td>
      <td>0.924</td>
      <td>0.753</td>
      <td>0.977</td>
    </tr>
  </tbody>
</table>
</table>

Rec-SSP achieved the best performance across all datasets in terms of both **MAE** and **RMSE**.

