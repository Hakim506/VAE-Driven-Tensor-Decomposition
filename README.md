# VAE-Driven-Tensor-Decomposition

This repository contains the code and models related to the research paper "VAE-Driven Tensor Decomposition: A Novel Architecture for Efficient Deep Learning".

## Overview

The paper introduces several novel Variational Autoencoder (VAE) architectures that incorporate tensor decomposition techniques, such as Nonnegative Matrix Factorization (NMF), Nonnegative Tucker Decomposition (NTD), and Canonical Polyadic Decomposition - Nonnegative (CPD-NN). These methods aim to reduce model complexity while maintaining or improving reconstruction performance.

The key contributions are:

1. Integration of tensor decomposition into VAEs, creating VAE-NMF, VAE-NTD, and VAE-CPD-NN architectures.

2. Introduction of a symmetric tensor factorization architecture (sVAE-CPD-NN) and a matrix-based variant (mVAE-CPD-NN).

3. Evaluation of the proposed methods on MNIST, Fashion MNIST, and Geometric Shapes datasets, comparing reconstruction quality, classification performance, and model complexity.

## Repository Structure

1. **Models**: Contains the trained VAE and classifier models for each dataset.

2. **article.py**: Includes all the functions used in the research paper, such as model definitions, training, evaluation, and utility functions.

3. **article1.ipynb**, **article2.ipynb**, **article3.ipynb**: Jupyter Notebook files executing the experiments on the three datasets.

## Usage

1. Install the required dependencies by running `pip install -r requirements.txt`.

2. Open the Jupyter Notebook files (**article1.ipynb**, **article2.ipynb**, **article3.ipynb**) to reproduce the experiments and results.

3. The models can be loaded from the **Models** directory and used for further experimentation or deployment.

4. The **article.py** file contains the implementation of the various VAE architectures and utility functions, which can be imported and used in your own code.

Feel free to explore the code, experiment with the models, and let me know if you have any questions or feedback!
