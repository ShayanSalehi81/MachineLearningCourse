# Machine Learning Course Assignments

This repository contains a collection of assignments and projects for the Machine Learning course at Sharif University of Technology. The repository includes various machine learning algorithms, deep learning models, and data processing techniques implemented in Python. Each directory corresponds to a specific topic or model, accompanied by code, datasets, and notebooks for practical experimentation and learning.

## Repository Structure

- **AdaBoost**
  - `heart_disease.csv`: Dataset for testing AdaBoost algorithms on tabular data, specifically heart disease prediction.
  - `ML_Models_for_Tabular_Datasets.ipynb`: Jupyter notebook implementing AdaBoost and other machine learning models on structured tabular data.

- **Autoencoder**
  - `Autoencoder.ipynb`: Implementation of autoencoder networks for dimensionality reduction, feature extraction, or unsupervised learning tasks.

- **CNN**
  - `CNN.ipynb`: Convolutional Neural Network (CNN) model implemented for image classification tasks, potentially with applications on datasets like CIFAR-10.

- **Decision Tree**
  - `Decision_Tree.ipynb`: Notebook detailing the implementation and training of decision tree models for classification tasks.

- **Linear Regression**
  - `Linear_Regression.ipynb`: Implementation of linear regression algorithms, focusing on regression problems and least squares optimization.

- **Neural Network**
  - `cifar10_downloader.bash`: Script to download the CIFAR-10 dataset, a commonly used image dataset for training neural networks.
  - `NeuralNet.ipynb`: Implementation of a custom neural network for tasks such as image classification, trained on the CIFAR-10 dataset.
  - `utils.py`: Utility functions to support neural network training, data preprocessing, and evaluation.

- **NN with PyTorch**
  - `data_utils.py`: Data processing utilities for handling datasets in PyTorch.
  - `Pytorch.ipynb`: Neural network implementation using the PyTorch framework, demonstrating how to train and evaluate models using this deep learning library.

- **SVM**
  - `SVM.ipynb`: Support Vector Machine (SVM) implementation for classification tasks, applied to suitable datasets for demonstrating SVM capabilities.

## Requirements

- Python 3.x
- Jupyter Notebook
- Libraries: `numpy`, `pandas`, `scikit-learn`, `tensorflow` (for CNNs), `torch` (for PyTorch models), and any additional libraries specific to each notebook.

## Usage

Each notebook can be run independently to explore the respective model or technique. Open any `.ipynb` file in Jupyter Notebook or JupyterLab to run the code cells step-by-step, allowing you to experiment with different parameters and observe the model's behavior and performance.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.