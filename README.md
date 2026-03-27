# Customer Segmentation System

An end-to-end machine learning project for customer segmentation using clustering and dimensionality reduction, with a Streamlit dashboard for interactive analysis.

## Features

- Data preprocessing and scaling
- Clustering with KMeans, DBSCAN, GMM, and Hierarchical clustering
- Dimensionality reduction with PCA and Autoencoder
- Cluster quality evaluation metrics
- Interactive Streamlit dashboard with visualization, model comparison, and CSV export

## Tech Stack

- Python
- scikit-learn
- PyTorch
- Streamlit
- pandas, numpy, matplotlib, seaborn
- joblib

## Project Structure

customer-segmentation-system/
|- app/
|  |- app.py
|- data/
|  |- Mall_Customers.csv
|- models/
|  |- autoencoder.pth
|  |- dbscan_model.pkl
|  |- gmm_model.pkl
|  |- hierarchical_model.pkl
|  |- kmeans_autoencoder.pkl
|  |- kmeans_model.pkl
|  |- pca.pkl
|  |- scaler.pkl
|- notebooks/
|  |- 01_EDA.ipynb
|  |- 02_KMeans.ipynb
|  |- 03_hierarchical_clustering.ipynb
|  |- 04_cluster_evaluation.ipynb
|  |- 05_DBSCAN.ipynb
|  |- 06_gmm.ipynb
|  |- 07_pca.ipynb
|  |- 08_tsne_visualization.ipynb
|  |- 09_autoencoders.ipynb
|- src/
|  |- __init__.py
|  |- autoencoder.py
|  |- clustering.py
|  |- data_preprocessing.py
|  |- evaluation.py
|  |- pipeline.py
|- requirements.txt
|- README.md

## Dataset

- Source: data/Mall_Customers.csv from Kaggle
- Link:(https://www.kaggle.com/datasets/simtoor/mall-customers)
- Segmentation features:
  - Annual Income (k$)
  - Spending Score (1-100)

## Installation

1. Clone the repository.

	git clone https://github.com/viru4/customer_segmentation_system.git
	cd customer_segmentation_system

2. Create and activate a virtual environment.

	Windows PowerShell:

	python -m venv .venv
	.venv\Scripts\Activate.ps1

3. Install dependencies.

	pip install -r requirements.txt

4. If PyTorch install fails on your machine, install CPU wheels directly.

	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

## Run the App

From project root:

streamlit run app/app.py

Dashboard options:

- Dimensionality Reduction: None, PCA, Autoencoder
- Clustering Algorithm: KMeans, DBSCAN, GMM, Hierarchical

## Notebook Workflow

Recommended order:

01_EDA.ipynb -> 02_KMeans.ipynb -> 03_hierarchical_clustering.ipynb -> 04_cluster_evaluation.ipynb -> 05_DBSCAN.ipynb -> 06_gmm.ipynb -> 07_pca.ipynb -> 08_tsne_visualization.ipynb -> 09_autoencoders.ipynb

## Core Modules

- src/data_preprocessing.py: loads dataset, selects features, applies scaling.
- src/clustering.py: loads persisted clustering models and returns predictions.
- src/autoencoder.py: defines the PyTorch autoencoder architecture.
- src/pipeline.py: applies PCA and Autoencoder transformation pipeline.
- src/evaluation.py: computes clustering metrics for model quality comparison.

## Troubleshooting

1. Streamlit command not found

- Use the correct spelling: streamlit (not stramlit).
- Activate .venv first.

2. Import torch could not be resolved

- Select the project interpreter in VS Code.
- Install torch in the same environment used by VS Code and notebook kernel.

3. ModuleNotFoundError: No module named src

- Run from project root for app startup.
- In notebooks, add project root to sys.path before importing from src.

4. InconsistentVersionWarning for scikit-learn model files

- Existing .pkl model files may be saved with an older scikit-learn version.
- Retrain and resave models in your current environment to remove this warning and ensure reliable predictions.

## Roadmap

- Add dedicated training scripts for each model.
- Add automated tests for src modules.
- Add Docker support.
- Add richer business-segment insights and dashboards.

## Author

Virendra Kumar
