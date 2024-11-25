import io
import os
import sys
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
import matplotlib.colors as clrs
import pandas as pd
from PIL import Image
from tqdm.notebook import tqdm
from IPython.display import display, Latex
import zipfile

from itertools import product
from scipy.linalg import khatri_rao
from sklearn.decomposition._nmf import _beta_divergence
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.utils import shuffle

tensorly_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../tensorly'))
sys.path.append(tensorly_path)
import tensorly as tl
from tensorly.decomposition import parafac, non_negative_parafac, tucker, non_negative_tucker_hals
from tensorly.random import random_cp, random_tucker

tensorflow_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../tensorflow'))
sys.path.append(tensorflow_path)
import tensorflow as tf

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset

# =============================    
# ===== GENERAL FUNCTIONS =====
# =============================  

##### SEED

def set_seed(seed=42):
    """
    Set the random seed for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
##### PARAMETER REDUCTION

def reduction(model1, model2):
    param1 = model1.count_parameters()
    param2 = model2.count_parameters()
    red = (param2 / param1 - 1) * 100
    print(f'Number of parameters: {param2} ({red:.2f}%)')

##### BINARY IMAGES TOOLS

def mask(img, threshold=0.2):
    msk = (img >= threshold) if threshold < 0.5 else (img < threshold)
    return msk.float()

def IoU_torch(img1, img2):
    assert img1.shape == img2.shape, "Input images must have the same dimensions"
    intersection = torch.logical_and(img1, img2).sum().float()
    union = torch.logical_or(img1, img2).sum().float()
    return intersection / union

##### MATRIX OPERATORS

def khatri_rao_tensor(A, B):
    assert A.shape[1] == B.shape[1], "Matrices must have the same number of columns"
    p, r = A.shape
    q, _ = B.shape
    return (A.unsqueeze(1) * B.unsqueeze(0)).reshape(p*q, r)

def inv_tensor(M):
    assert M.dim() == 2, "M must be a 2D matrix"
    return torch.linalg.inv(M)

def tr_pi_tensor(A, B):
    kr = khatri_rao_tensor(A, B)
    hp = torch.mm(A.T, A) * torch.mm(B.T, B)
    return torch.mm(kr, inv_tensor(hp).T)

##### VARIATIONAL AUTOENCODER

class VAE(nn.Module):
    def __init__(self, dataset='Fashion MNIST', input_channels=1, input_size=32, rank=64, name='VAE', 
                 train_losses=[], val_losses=[], train_IoU=[], val_IoU=[],
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(VAE, self).__init__()

        self.dataset = dataset
        self.input_channels = input_channels
        self.input_size = input_size
        self.rank = rank
        self.name = name
        self.train_losses = train_losses
        self.val_losses = val_losses
        self.train_IoU = train_IoU
        self.val_IoU = val_IoU
        self.device = device
        self.to(self.device)

    def encoder(self, x):
        raise NotImplementedError("To be redefined in children classes.")

    def decoder(self, z):
        raise NotImplementedError("To be redefined in children classes.")

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, coef=10**-3, bce=False):
        REC = F.binary_cross_entropy(recon_x, x, reduction='sum') if bce else F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return REC + KLD * coef

    def fit(self, trainloader, valloader, epochs=10, learning_rate=1e-3, threshold=0.2, best=True, coef=10**-3,print=True):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        best_metric = float('inf')
        best_model_state_dict = None

        for epoch in tqdm(range(epochs), desc=f'VAE-{self.name}', leave=False):
            # Training
            self.train()
            train_loss = 0
            train_iou = 0
            for imgs, _ in tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
                imgs = imgs.to(self.device)
                optimizer.zero_grad()
                recon_imgs, mu, logvar = self(imgs)
                loss = self.loss_function(recon_imgs, imgs, mu, logvar, coef)
                train_loss += loss.item()
                binarized_imgs = mask(imgs, threshold)
                binarized_outputs = mask(recon_imgs, threshold)
                train_iou += IoU_torch(binarized_imgs, binarized_outputs).item()
                loss.backward()
                optimizer.step()
            avg_train_loss = train_loss / len(trainloader.dataset)
            avg_train_iou = 100 * train_iou / len(trainloader)
            self.train_losses.append(avg_train_loss)
            self.train_IoU.append(avg_train_iou)

            # Validation
            self.eval()
            total_loss = 0
            total_iou = 0
            with torch.no_grad():
                for imgs, _ in valloader:
                    imgs = imgs.to(self.device)
                    recon_imgs, mu, logvar = self(imgs)
                    loss = self.loss_function(recon_imgs, imgs, mu, logvar, coef)
                    total_loss += loss.item()
                    binarized_imgs = mask(imgs, threshold)
                    binarized_outputs = mask(recon_imgs, threshold)
                    total_iou += IoU_torch(binarized_imgs, binarized_outputs).item() * imgs.size(0)
            avg_loss = total_loss / len(valloader.dataset)
            avg_iou = 100 * total_iou / len(valloader.dataset)
            self.val_losses.append(avg_loss)
            self.val_IoU.append(avg_iou)

            # Best model
            if avg_loss < best_metric:
                best_metric = avg_loss
                best_model_state_dict = self.state_dict()

        # Save best model state
        self.best_model_state_dict = best_model_state_dict

        # Print best model info
        if best:
            best_epoch = self.val_losses.index(best_metric) + 1
            display(Latex(f'Best {self.name} model at epoch {best_epoch} with error {best_metric:.5f}'))

        # Plot and show
        if print:
            self.plot()
            self.show(valloader)
            # t-SNE visualization
            self.tsne_visualization_both_sets(trainloader, valloader)

    def plot(self):
        plt.figure(figsize=(15, 5))
        idx = np.arange(len(self.train_losses)) + 1

        plt.subplot(1, 2, 1)
        plt.plot(idx[2:], self.train_losses[2:], label='Training Loss', linestyle='--', linewidth=2, color='k')
        plt.plot(idx[2:], self.val_losses[2:], label='Validation Loss', linewidth=3, color='darkcyan')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.grid()
        plt.legend()
        text = f'Final training error: {self.train_losses[-1]:.5f} and final validation error: {self.val_losses[-1]:.5f}'
        plt.text(0.5, -0.2, text, ha="center", fontsize=10, transform=plt.gca().transAxes, 
                 bbox=dict(facecolor='darkcyan', alpha=0.5, pad=5))

        plt.subplot(1, 2, 2)
        plt.plot(idx[2:], self.train_IoU[2:], label='Training IoU', linestyle='--', linewidth=2, color='k')
        plt.plot(idx[2:], self.val_IoU[2:], label='Validation IoU', linewidth=3, color='darkgreen')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.title('IoU Curve')
        plt.grid()
        plt.legend()
        text = f'Final training IoU: {self.train_IoU[-1]:.5f} and final validation IoU: {self.val_IoU[-1]:.5f}'
        plt.text(0.5, -0.2, text, ha="center", fontsize=10, transform=plt.gca().transAxes, 
                 bbox=dict(facecolor='darkgreen', alpha=0.5, pad=5))

        plt.tight_layout()
        plt.show()

    def show(self, dataloader, num_images=10, title='Reconstructed Images'):
        self.load_state_dict(self.best_model_state_dict)  # Load the best model state
        self.eval()
        images, _ = next(iter(dataloader))
        images = images.to(self.device)
        with torch.no_grad():
            reconstructed, _, _ = self(images)
        images = images.cpu()
        reconstructed = reconstructed.cpu()

        fig, axes = plt.subplots(2, num_images, figsize=(15, 3))
        for i in range(num_images):
            ax = axes[0, i]
            ax.imshow(images[i].reshape(self.input_size, self.input_size), cmap='gray')
            ax.axis('off')
            if i == 0:
                ax.set_title('Original Images')

            ax = axes[1, i]
            ax.imshow(reconstructed[i].reshape(self.input_size, self.input_size), cmap='gray')
            ax.axis('off')
            if i == 0:
                ax.set_title(title)
        plt.show()
        
    def test(self, testloader, threshold=0.2, num_images=10, coef=10**-3):
        self.load_state_dict(self.best_model_state_dict)  # Load the best model state
        self.eval()

        total_loss = 0
        total_iou = 0

        with torch.no_grad():
            for imgs, _ in testloader:
                imgs = imgs.to(self.device)
                recon_imgs, mu, logvar = self(imgs)
                loss = self.loss_function(recon_imgs, imgs, mu, logvar, coef)
                total_loss += loss.item()
                binarized_imgs = mask(imgs, threshold)
                binarized_outputs = mask(recon_imgs, threshold)
                total_iou += IoU_torch(binarized_imgs, binarized_outputs).item() * imgs.size(0)

        avg_loss = total_loss / len(testloader.dataset)
        avg_iou = total_iou / len(testloader.dataset)

        # Display metrics in an orange rectangle
        text = f'Test Error: {avg_loss:.5f}, IoU: {avg_iou:.5f}'
        plt.figure(figsize=(15, 1))
        plt.text(0.5, 0.5, text, ha="center", va="center", fontsize=10, bbox=dict(facecolor='lightseagreen', alpha=0.5, pad=5))
        plt.axis('off')
        plt.show()
        return avg_loss,avg_iou

    def tsne_visualization_both_sets(self, trainloader, valloader, num_images=1000, perplexity=20, n_iter=1500):
        self.load_state_dict(self.best_model_state_dict)
        self.eval()

        def get_latents_labels(loader, num_images):
            latents, logvars, labels = [], [], []
            with torch.no_grad():
                for imgs, lbls in loader:
                    imgs = imgs.to(self.device)
                    mu, logvar = self.encoder(imgs)  # Assume encoder returns mu and logvar
                    latents.append(mu.cpu().numpy())
                    logvars.append(logvar.cpu().numpy())
                    labels.append(lbls.numpy())
                    if len(latents) * imgs.size(0) >= num_images:
                        break
            return np.concatenate(latents)[:num_images], np.concatenate(logvars)[:num_images], np.concatenate(labels)[:num_images]

        train_latents, train_logvars, train_labels = get_latents_labels(trainloader, num_images)
        val_latents, val_logvars, val_labels = get_latents_labels(valloader, num_images)

        # Combine train and val data for a single t-SNE computation
        combined_latents = np.vstack((train_latents, val_latents))
        combined_logvars = np.vstack((train_logvars, val_logvars))  # Log-variances are now included
        combined_labels = np.hstack((train_labels, val_labels))

        # Perform t-SNE on combined data using the log-variances
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
        combined_latents_2d = tsne.fit_transform(combined_latents)

        # Optionally, use logvars for distribution visualization
        # Prepare the distribution of standard deviations for visualization
        train_stds = np.exp(train_logvars / 2)
        val_stds = np.exp(val_logvars / 2)

        # Split the results back into train and val
        train_latents_2d = combined_latents_2d[:len(train_latents)]
        val_latents_2d = combined_latents_2d[len(train_latents):]

        # Define class names based on the dataset
        class_names = {
            'MNIST': [f'{i}' for i in range(10)],
            'Fashion MNIST': ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
            'Geometric Shapes': ['Triangle', 'Square', 'Circle'],
            # Add other datasets as needed
        }.get(self.dataset, [f'Class {i}' for i in range(len(np.unique(combined_labels)))])

        # Determine the colormap based on the number of classes
        num_classes = len(class_names)
        if num_classes == 3:
            colors = ['brown', 'darkcyan', 'orange']
            cmap = clrs.ListedColormap(colors)
        else:
            cmap = 'tab10'

        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        for ax, data, labels, title in zip(axes, [train_latents_2d, val_latents_2d], 
                                               [train_labels, val_labels], 
                                               ['Train', 'Validation']):
            scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap=cmap, alpha=0.6, s=20)
            ax.set_title(f't-SNE visualization of {self.name} {title.lower()} latent space')
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, ticks=range(num_classes))
            cbar.set_label('Class')
            cbar.set_ticklabels(class_names)

        plt.tight_layout()
        plt.show()

        # Plot distribution of latent space
        self.plot_distribution(train_labels, train_latents, train_logvars)
    
    def plot_distribution(self, labels, mus, logvars):
        # Define class names based on the dataset
        class_names = {
            'MNIST': [f'{i}' for i in range(10)],
            'Fashion MNIST': ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
            'Geometric Shapes': ['Triangle', 'Square', 'Circle'],
            # Add other datasets as needed
        }.get(self.dataset, [f'Class {i}' for i in range(len(np.unique(labels)))])

        num_classes = len(class_names)
        fig, axes = plt.subplots(2, num_classes, figsize=(20, 5))

        for i in range(num_classes):
            idx = labels == i
            axes[0, i].hist(mus[idx].flatten(), bins=30, alpha=0.7, color='darkcyan')
            axes[0, i].set_title(f'{class_names[i]}')
            if i == 0:
                axes[0, i].set_ylabel('Means')
            axes[0, i].axvline(0, color='k', linestyle='--')
            axes[0, i].grid(True)

            axes[1, i].hist(np.exp(logvars[idx] / 2).flatten(), bins=30, alpha=0.7, color='darkgreen')
            if i == 0:
                axes[1, i].set_ylabel('Std Deviation')
            axes[1, i].axvline(1, color='k', linestyle='--')
            axes[1, i].grid(True)

        plt.tight_layout()
        plt.show()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def save_model(self, path):
        model_state = {
            'state_dict': self.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_IoU': self.train_IoU,
            'val_IoU': self.val_IoU,
            'best_model_state_dict': self.best_model_state_dict
        }
        torch.save(model_state, path)
        print(f'Model saved to {path}')

    def load_model(self, path):
        model_state = torch.load(path, map_location=self.device)
        self.load_state_dict(model_state['state_dict'])
        self.train_losses = model_state['train_losses']
        self.val_losses = model_state['val_losses']
        self.train_IoU = model_state['train_IoU']
        self.val_IoU = model_state['val_IoU']
        self.best_model_state_dict = model_state['best_model_state_dict']
        print(f'Model loaded from {path}')
        
    def load_and_plot(self, path, trainloader, valloader):
        # Load the model
        self.load_model(path)
        # Display the best metric and epoch
        best_metric = min(self.val_losses)
        best_epoch = self.val_losses.index(best_metric) + 1
        display(Latex(f'Best {self.name} model at epoch {best_epoch} with error {best_metric:.5f}'))
        # Plot and show
        self.plot()
        self.show(valloader)
        # t-SNE visualization
        self.tsne_visualization_both_sets(trainloader, valloader)
        
##### CLASSIFIER

class Classifier(nn.Module):
    def __init__(self, model, dataset='MNIST', input_dim=64, num_classes=10,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(Classifier, self).__init__()
        self.dataset = dataset
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.train_precisions = []
        self.train_recalls = []
        self.train_f1s = []
        self.val_precisions = []
        self.val_recalls = []
        self.val_f1s = []
        self.model = model
        self.name = model.name
        self.input_dim = input_dim
        self.device = device
        self.num_classes = num_classes
        self.best_model_state_dict = None
        self.best_val_loss = float('inf')
        self.to(self.device)
    
        if input_dim <= 96:
            self.cla = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, num_classes)
            ).to(self.device)
        elif 96 < input_dim <= 192:
            self.cla = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, 96),
                nn.ReLU(),
                nn.Linear(96, 48),
                nn.ReLU(),
                nn.Linear(48, 24),
                nn.ReLU(),
                nn.Linear(24, 12),
                nn.ReLU(),
                nn.Linear(12, num_classes)
            ).to(self.device)
        else:
            self.cla = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_dim, 192),
                nn.ReLU(),
                nn.Linear(192, 48),
                nn.ReLU(),
                nn.Linear(48, 12),
                nn.ReLU(),
                nn.Linear(12, num_classes)
            ).to(self.device)
    
    def forward(self, x):
        return self.cla(x)

    def extract_latents(self, dataloader, batch_size=32, shuffle=True):
        self.model.eval()
        latents = []
        labels = []
        with torch.no_grad():
            for imgs, lbls in dataloader:
                imgs = imgs.to(self.model.device)
                mu, logvar = self.model.encoder(imgs)
                encoded = self.model.reparameterize(mu, logvar)
                latents.append(encoded.cpu())
                labels.append(lbls)
        latents, labels = torch.cat(latents), torch.cat(labels)
        dataset = torch.utils.data.TensorDataset(latents, labels)
        dataloader_latents = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader_latents

    def fit(self, train_loader, val_loader, epochs=10, learning_rate=1e-3, batch_size=32):
        trainloader = self.extract_latents(train_loader, batch_size=batch_size) 
        valloader = self.extract_latents(val_loader, batch_size=batch_size, shuffle=False)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in tqdm(range(epochs), desc=f'Classifier {self.name}', leave=False):
            # Training
            self.train()
            train_loss = 0
            train_correct = 0
            total_samples = 0
            all_train_preds = []
            all_train_labels = []
            for latents, labels in tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
                latents, labels = latents.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                logits = self(latents)
                loss = criterion(logits, labels)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(logits, 1)
                train_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                all_train_preds.extend(predicted.cpu().numpy())
                all_train_labels.extend(labels.cpu().numpy())
            avg_train_loss = train_loss / len(trainloader)
            train_acc = train_correct / total_samples
            train_precision = precision_score(all_train_labels, all_train_preds, average='macro')
            train_recall = recall_score(all_train_labels, all_train_preds, average='macro')
            train_f1 = f1_score(all_train_labels, all_train_preds, average='macro')
            self.train_losses.append(avg_train_loss)
            self.train_accs.append(train_acc)
            self.train_precisions.append(train_precision)
            self.train_recalls.append(train_recall)
            self.train_f1s.append(train_f1)

            # Validation
            self.eval()
            val_loss = 0
            val_correct = 0
            total_samples = 0
            all_val_preds = []
            all_val_labels = []
            with torch.no_grad():
                for latents, labels in valloader:
                    latents, labels = latents.to(self.device), labels.to(self.device)
                    logits = self(latents)
                    loss = criterion(logits, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(logits, 1)
                    val_correct += (predicted == labels).sum().item()
                    total_samples += labels.size(0)
                    all_val_preds.extend(predicted.cpu().numpy())
                    all_val_labels.extend(labels.cpu().numpy())
            avg_val_loss = val_loss / len(valloader)
            val_acc = val_correct / total_samples
            val_precision = precision_score(all_val_labels, all_val_preds, average='macro')
            val_recall = recall_score(all_val_labels, all_val_preds, average='macro')
            val_f1 = f1_score(all_val_labels, all_val_preds, average='macro')
            self.val_losses.append(avg_val_loss)
            self.val_accs.append(val_acc)
            self.val_precisions.append(val_precision)
            self.val_recalls.append(val_recall)
            self.val_f1s.append(val_f1)

            # Save the best model
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.best_model_state_dict = self.state_dict()

        # Print best model info
        best_epoch = self.val_losses.index(self.best_val_loss) + 1
        print(f'Best classifier {self.name} at epoch {best_epoch} with CrossEntropy {self.best_val_loss:.5f}')

        self.plot()

    def plot(self):
        fig, axs = plt.subplots(1, 5, figsize=(25, 5))
        metrics = [
            ('Loss', self.train_losses, self.val_losses),
            ('Accuracy', self.train_accs, self.val_accs),
            ('Precision', self.train_precisions, self.val_precisions),
            ('Recall', self.train_recalls, self.val_recalls),
            ('F1 Score', self.train_f1s, self.val_f1s)
        ]
        
        for i, (metric, train_data, val_data) in enumerate(metrics):
            ax = axs[i]
            idx = np.arange(len(train_data)) + 1
            ax.plot(idx, train_data, label=f'Training {metric}', linestyle='--', linewidth=2, color='k')
            ax.plot(idx, val_data, label=f'Validation {metric}', linewidth=3, color='darkcyan')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Curve')
            ax.grid()
            ax.legend()
        
        plt.tight_layout()
        plt.show()

    def test(self, test_loader, batch_size=32):
        testloader = self.extract_latents(test_loader, batch_size=batch_size, shuffle=False)
        
        self.load_state_dict(self.best_model_state_dict)  # Load the best model state
        self.eval()
        test_correct = 0
        total_samples = 0
        all_test_preds = []
        all_test_labels = []
        with torch.no_grad():
            for latents, labels in testloader:
                latents, labels = latents.to(self.device), labels.to(self.device)
                logits = self(latents)
                _, predicted = torch.max(logits, 1)
                test_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                all_test_preds.extend(predicted.cpu().numpy())
                all_test_labels.extend(labels.cpu().numpy())
        test_acc = test_correct / total_samples
        test_precision = precision_score(all_test_labels, all_test_preds, average='macro')
        test_recall = recall_score(all_test_labels, all_test_preds, average='macro')
        test_f1 = f1_score(all_test_labels, all_test_preds, average='macro')
        
        # Define class names based on the dataset
        class_names = {
            'MNIST': [f'{i}' for i in range(10)],
            'Fashion MNIST': ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
            'Geometric Shapes': ['Triangle', 'Square', 'Circle'],
            # Add other datasets as needed
        }.get(self.dataset, [f'Class {i}' for i in range(len(np.unique(labels)))])
        
        # Confusion Matrix
        cm = confusion_matrix(all_test_labels, all_test_preds)
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')

        # ROC AUC
        if self.num_classes == 2:
            test_roc_auc = roc_auc_score(all_test_labels, all_test_preds)
            fpr, tpr, _ = roc_curve(all_test_labels, all_test_preds)
            roc_auc = auc(fpr, tpr)
            plt.subplot(1, 2, 2)
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.5f})')
            plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
        else:
            test_labels_binarized = label_binarize(all_test_labels, classes=range(self.num_classes))
            test_preds_binarized = label_binarize(all_test_preds, classes=range(self.num_classes))
            test_roc_auc = roc_auc_score(test_labels_binarized, test_preds_binarized, average='macro', multi_class='ovr')
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(self.num_classes):
                fpr[i], tpr[i], _ = roc_curve(test_labels_binarized[:, i], test_preds_binarized[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            plt.subplot(1, 2, 2)
            for i in range(self.num_classes):
                plt.plot(fpr[i], tpr[i], lw=2, label=f'{class_names[i]} (area = {roc_auc[i]:.5f})')
            plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
        plt.show()
        
        # Display metrics in an orange rectangle
        text = f'Test Accuracy: {test_acc:.5f}, Precision: {test_precision:.5f}, Recall: {test_recall:.5f}, F1 Score: {test_f1:.5f}, ROC AUC: {test_roc_auc:.5f}'
        plt.figure(figsize=(15, 1))
        plt.text(0.5, 0.5, text, ha="center", va="center", fontsize=10, bbox=dict(facecolor='darkcyan', alpha=0.5, pad=5))
        plt.axis('off')
        plt.show()

    def save_model(self, path):
        model_state = {
            'state_dict': self.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'train_precisions': self.train_precisions,
            'train_recalls': self.train_recalls,
            'train_f1s': self.train_f1s,
            'val_precisions': self.val_precisions,
            'val_recalls': self.val_recalls,
            'val_f1s': self.val_f1s,
            'best_model_state_dict': self.best_model_state_dict,
            'best_val_loss': self.best_val_loss,
        }
        torch.save(model_state, path)
        print(f'Model saved to {path}')
    
    def load_model(self, path):
        model_state = torch.load(path, map_location=self.device)
        self.load_state_dict(model_state['state_dict'])
        self.train_losses = model_state['train_losses']
        self.val_losses = model_state['val_losses']
        self.train_accs = model_state['train_accs']
        self.val_accs = model_state['val_accs']
        self.train_precisions = model_state['train_precisions']
        self.train_recalls = model_state['train_recalls']
        self.train_f1s = model_state['train_f1s']
        self.val_precisions = model_state['val_precisions']
        self.val_recalls = model_state['val_recalls']
        self.val_f1s = model_state['val_f1s']
        self.best_model_state_dict = model_state['best_model_state_dict']
        self.best_val_loss = model_state['best_val_loss']
        print(f'Model loaded from {path}')
        
    def load_and_plot(self, path, test_loader, batch_size=32):
        self.load_model(path)
        best_metric = min(self.val_losses)
        best_epoch = self.val_losses.index(best_metric) + 1
        display(Latex(f'Best classifier {self.name} at epoch {best_epoch} with CrossEntropy {best_metric:.5f}'))
        self.plot()
        self.test(test_loader, batch_size=batch_size)
    
# ==========================    
# ===== MNIST DATASETS =====
# ==========================    

##### DATA LOADING 

def load_dataset(name='Fashion MNIST', random_seed=42):
    datasets = {
        'MNIST': tf.keras.datasets.mnist,
        'Fashion MNIST': tf.keras.datasets.fashion_mnist
    }
    
    if name not in datasets:
        raise ValueError("Invalid name. Please choose from 'MNIST' or 'Fashion MNIST'.")
    
    (X_train_full, y_train_full), (X_test, y_test) = datasets[name].load_data()
    
    # Normalize images to range [0, 1]
    X_train_full = X_train_full.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, train_size=5/6, stratify=y_train_full, random_state=random_seed
    )
    
    # Resize images
    X_train, X_val, X_test = map(resize, (X_train, X_val, X_test))
    
    # Plot sample images
    plot_sample_images(X_train, y_train, name, 'Training images')
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def resize(images, size=32):
    return tf.image.resize(images[..., np.newaxis], [size, size], method='nearest').numpy()[..., 0]

def plot_sample_images(X, y, name='Fashion MNIST', title='Original images'):
    classes = {
        'MNIST': np.arange(10),
        'Fashion MNIST': ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    }[name]
    
    fig, axes = plt.subplots(3, 10, figsize=(15, 5))
    for label in range(10):
        class_indices = np.where(y == label)[0][:3]
        for i, idx in enumerate(class_indices):
            ax = axes[i, label]
            ax.imshow(X[idx], cmap='gray', interpolation='none')
            ax.axis('off')
            if i == 0:
                ax.set_title(f'{classes[label]}')
    plt.suptitle(title, size=20)
    plt.tight_layout()
    plt.show()

def dataloaders(Xtr, Ytr, Xva, Yva, Xte, Yte, batch_size=32):
    def process_data(X, Y):
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        Y = torch.tensor(Y, dtype=torch.long)
        return TensorDataset(X, Y)

    train_dataset = process_data(Xtr, Ytr)
    val_dataset = process_data(Xva, Yva)
    test_dataset = process_data(Xte, Yte)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size),
        DataLoader(test_dataset, batch_size=batch_size)
    )
    
##### VARIATIONAL AUTOENCODERS

class Classic_VAE(VAE):
    def __init__(self, dataset='Fashion MNIST', input_channels=1, input_size=32, rank=64, name='classic', 
                 train_losses=[], val_losses=[], train_IoU=[], val_IoU=[],  
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(Classic_VAE, self).__init__(dataset, input_channels, input_size, rank, name, 
                                          train_losses, val_losses, train_IoU, val_IoU, device)

        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(input_channels, 4, kernel_size=5, stride=2, padding=2),  # Output: (4, 16, 16)
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),  # Output: (8, 8, 8)
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # Output: (16, 4, 4)
            nn.ReLU(),
            nn.Flatten()
        ).to(self.device)

        # Mean and Log Variance
        self.mean = nn.Linear(16 * 4 * 4, rank)
        self.std = nn.Linear(16 * 4 * 4, rank)

        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(rank, 16 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (16, 4, 4)),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (8, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (4, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(4, input_channels, kernel_size=4, stride=2, padding=1),  # Output: (input_channels, 32, 32)
            nn.Sigmoid()
        ).to(self.device)

    def encoder(self, x):
        x = self.enc(x)
        mu, logvar = self.mean(x), self.std(x)
        return mu, logvar

    def decoder(self, z):
        return self.dec(z)
    
class VAE_NMF(VAE):
    def __init__(self, dataset='Fashion MNIST', input_channels=1, input_size=32, rank=64, name='NMF', 
                 train_losses=[], val_losses=[], train_IoU=[], val_IoU=[],  
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(VAE_NMF, self).__init__(dataset, input_channels, input_size, rank, name,
                                      train_losses, val_losses, train_IoU, val_IoU, device)

        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(input_channels, 4, kernel_size=5, stride=2, padding=2),  # Output: (4, 16, 16)
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),  # Output: (8, 8, 8)
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # Output: (16, 4, 4)
            nn.ReLU(),
            nn.Flatten()
        ).to(self.device)

        # Mean and Std
        self.mean = nn.Linear(16 * 4 * 4, input_size * rank)
        self.std = nn.Linear(16 * 4 * 4, input_size * rank)

        self.V = nn.Parameter(torch.empty(rank, input_size, device=self.device), requires_grad=True)
        nn.init.kaiming_uniform_(self.V, mode='fan_out', nonlinearity='relu')

        self.unf = nn.Unflatten(1, (1, input_size, rank))

    def encoder(self, x):
        x = self.enc(x)
        mu, logvar = self.mean(x), self.std(x)
        return mu, logvar

    def decoder(self, z):
        z = self.unf(z)
        return torch.sigmoid(torch.matmul(z, self.V.to(self.device)))
    
class VAE_NTD(VAE):
    def __init__(self, dataset='Fashion MNIST', input_channels=1, input_size=32, rank=[8, 8, 64], name='NTD', 
                 train_losses=[], val_losses=[], train_IoU=[], val_IoU=[],  
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(VAE_NTD, self).__init__(dataset, input_channels, input_size, rank, name,
                                      train_losses, val_losses, train_IoU, val_IoU, device)

        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(input_channels, 4, kernel_size=5, stride=2, padding=2),  # Output: (4, 16, 16)
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),  # Output: (8, 8, 8)
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # Output: (16, 4, 4)
            nn.ReLU(),
            nn.Flatten()
        ).to(self.device)

        # Mean and Std
        self.mean = nn.Linear(16 * 4 * 4, rank[2])
        self.std = nn.Linear(16 * 4 * 4, rank[2])
        
        # Parameters (Weights)
        self.G3 = nn.Parameter(torch.empty(rank[2], rank[0] * rank[1], device=self.device), requires_grad=True)
        nn.init.kaiming_uniform_(self.G3, mode='fan_out', nonlinearity='relu')
        self.W = nn.Parameter(torch.empty(input_size, rank[0], device=self.device), requires_grad=True)
        nn.init.kaiming_uniform_(self.W, mode='fan_out', nonlinearity='relu')
        self.H = nn.Parameter(torch.empty(input_size, rank[1], device=self.device), requires_grad=True)
        nn.init.kaiming_uniform_(self.H, mode='fan_out', nonlinearity='relu')

        # Reconstruction
        self.rec_q = nn.Unflatten(1, (1, rank[2]))
        self.rec = nn.Unflatten(2, (input_size, input_size))

    def encoder(self, x):
        x = self.enc(x)
        mu, logvar = self.mean(x), self.std(x)
        return mu, logvar

    def decoder(self, z):
        qg = F.relu(torch.matmul(self.rec_q(z), self.G3.to(self.device)))
        wh = torch.kron(self.W.T, self.H.T)
        x_hat = F.sigmoid(torch.matmul(qg, wh))
        return self.rec(x_hat)

class VAE_CPD_NN(VAE):
    def __init__(self, dataset='Fashion MNIST', input_channels=1, input_size=32, rank=64, name='CPD-NN', 
                 train_losses=[], val_losses=[], train_IoU=[], val_IoU=[],  
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), encod='Neuronal'):
        super(VAE_CPD_NN, self).__init__(dataset, input_channels, input_size, rank, name,
                                         train_losses, val_losses, train_IoU, val_IoU, device)

        # Encoder type
        self.encod = encod
        # Encoder
        self.fl = nn.Flatten()
        if encod == 'Neuronal':
            self.enc = nn.Sequential(
                nn.Conv2d(input_channels, 4, kernel_size=5, stride=2, padding=2),  # Output: (4, 16, 16)
                nn.ReLU(),
                nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),  # Output: (8, 8, 8)
                nn.ReLU(),
                nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # Output: (16, 4, 4)
                nn.ReLU(),
                nn.Flatten()
            ).to(self.device)

            # Mean and Std
            self.mean = nn.Linear(16 * 4 * 4, rank)
            self.std = nn.Linear(16 * 4 * 4, rank)

        else:
            # Mean and Std
            self.mean = nn.Linear(rank, rank)
            self.std = nn.Linear(rank, rank)

        self.rec_c = nn.Unflatten(1, (1, rank))
        self.rec = nn.Unflatten(2, (input_size, input_size))

        # Parameters (Weights)
        self.A = nn.Parameter(torch.empty(input_size, rank, device=self.device), requires_grad=True)
        nn.init.kaiming_uniform_(self.A, mode='fan_out', nonlinearity='relu')
        self.B = nn.Parameter(torch.empty(input_size, rank, device=self.device), requires_grad=True)
        nn.init.kaiming_uniform_(self.B, mode='fan_out', nonlinearity='relu')

        if encod == 'Matrices':
            self.Y = nn.Parameter(torch.empty(input_size, rank, device=self.device), requires_grad=True)
            nn.init.kaiming_uniform_(self.Y, mode='fan_out', nonlinearity='relu')
            self.Z = nn.Parameter(torch.empty(input_size, rank, device=self.device), requires_grad=True)
            nn.init.kaiming_uniform_(self.Z, mode='fan_out', nonlinearity='relu')

        # Activation function
        self.sig = nn.ReLU()
    
    def encoder(self, x):
        if self.encod == 'Neuronal':
            c = self.enc(x)
        else:
            if self.encod == 'Matrices':
                term = khatri_rao_tensor(self.Y.to(self.device), self.Z.to(self.device))
            elif self.encod == 'Symmetric':
                term = tr_pi_tensor(self.A.to(self.device), self.B.to(self.device))
            else:
                raise ValueError("Invalid encoder. Please choose from 'Neuronal', 'Matrices', or 'Symmetric'.")
            
            prod = torch.matmul(self.fl(x), term)
            c = self.sig(prod)
            c = self.fl(c)
        
        mu, logvar = self.mean(c), self.std(c)
        return mu, logvar
        
    def decoder(self, z):
        term = khatri_rao_tensor(self.sig(self.A), self.sig(self.B)).T
        x_hat = torch.matmul(self.rec_c(z), term.to(self.device))
        return torch.sigmoid(self.rec(x_hat))
    
# ====================================    
# ===== GEOMETRIC SHAPES DATASET =====
# ====================================     
    
##### DATA LOADING    
    
def load_geometric_shapes(zip_path='../../data/GeometricShapes.zip', zip_folder='GeometricShapes/', random_seed=42, batch_size=1000, resize_to=96):
    label_mapping = {
       'Triangle': 0,
       'Square': 1,
       'Circle': 2
    }

    all_images = []
    all_labels = []

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for shape, label in label_mapping.items():
            shape_folder = os.path.join(zip_folder, shape)
            image_names = [name for name in zip_ref.namelist() if name.startswith(shape_folder) and name.lower().endswith(('.png', '.jpg', '.jpeg'))]

            for i in range(0, len(image_names), batch_size):
                batch_names = image_names[i:i+batch_size]
                batch_images = []
                batch_labels = []

                for img_name in batch_names:
                    with zip_ref.open(img_name) as image_file:
                        image = Image.open(image_file).convert('L')
                        image_array = np.array(image, dtype=np.float32)

                        if np.min(image_array) == np.max(image_array):
                            continue

                        image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))

                        border_pixels = np.concatenate([
                           image_array[0, :], image_array[-1, :], image_array[:, 0], image_array[:, -1]
                        ])

                        if np.max(border_pixels) == np.max(image_array):
                            image_array = np.max(image_array) - image_array

                        batch_images.append(image_array)
                        batch_labels.append(label)
                        
                # Resize the batch of images
                batch_images = resize(np.array(batch_images), size=resize_to)

                all_images.extend(batch_images)
                all_labels.extend(batch_labels)

                #print(f"Processed {len(all_images)} images so far...")

    all_images = np.array(all_images)
    all_labels = np.array(all_labels)
    
    ## Print some details
    #print(f"Total images loaded: {len(all_images)}")
    #print(f"Shape of images array: {all_images.shape}")
    #print(f"Shape of labels array: {all_labels.shape}")

    all_images = np.expand_dims(all_images, axis=1)  # Add channel dimension

    X_train, X_temp, y_train, y_temp = train_test_split(
       all_images, all_labels, stratify=all_labels, test_size=0.3, random_state=random_seed)

    X_val, X_test, y_val, y_test = train_test_split(
       X_temp, y_temp, stratify=y_temp, test_size=0.5, random_state=random_seed)

    return X_train, y_train, X_val, y_val, X_test, y_test

def show_geometric_shapes(dataset, class_labels=[0, 1, 2], num_images=10):
    indices = {label: [] for label in class_labels}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label = label.item()
        if len(indices[label]) < num_images:
            indices[label].append(idx)
        if all(len(indices[label]) >= num_images for label in class_labels):
            break

    titles = {0: "Triangle", 1: "Square", 2: "Circle"}

    for label, idxs in indices.items():
        print(f"Images of {titles[label]}:")
        plt.figure(figsize=(20, 4))
        for i, idx in enumerate(idxs):
            image, _ = dataset[idx]
            plt.subplot(1, num_images, i + 1)
            plt.imshow(image[0], cmap='gray')
            plt.axis('off')
        plt.show()

def geometric_shapes_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32, plot=False):
    X_train, X_val, X_test = torch.tensor(X_train), torch.tensor(X_val), torch.tensor(X_test)
    y_train, y_val, y_test = torch.tensor(y_train), torch.tensor(y_val), torch.tensor(y_test)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if plot:
        show_geometric_shapes(train_dataset)

    return train_loader, val_loader, test_loader

##### VARIATIONAL AUTOENCODERS

class Classic_Geometric_VAE(VAE):
    def __init__(self, dataset='Geometric Shapes', input_channels=1, input_size=96, rank=64, name='classic', 
                 train_losses=None, val_losses=None, train_IoU=None, val_IoU=None,  
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        if train_losses is None:
            train_losses = []
        if val_losses is None:
            val_losses = []
        if train_IoU is None:
            train_IoU = []
        if val_IoU is None:
            val_IoU = []

        super(Classic_Geometric_VAE, self).__init__(dataset, input_channels, input_size, rank, name, 
                                                    train_losses, val_losses, train_IoU, val_IoU, device)

        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=5, stride=2, padding=2),  # Output: (8, 48, 48)
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # Output: (16, 24, 24)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Output: (32, 12, 12)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: (64, 6, 6)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: (128, 3, 3)
            nn.ReLU(),
            nn.Flatten()
        ).to(self.device)

        # Mean and Log Variance
        self.mean = nn.Linear(128 * 3 * 3, rank)
        self.std = nn.Linear(128 * 3 * 3, rank)

        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(rank, 128 * 3 * 3),
            nn.ReLU(),
            nn.Unflatten(1, (128, 3, 3)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (64, 6, 6)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (32, 12, 12)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (16, 24, 24)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (8, 48, 48)
            nn.ReLU(),
            nn.ConvTranspose2d(8, input_channels, kernel_size=4, stride=2, padding=1),  # Output: (input_channels, 96, 96)
            nn.Sigmoid()
        ).to(self.device)

    def encoder(self, x):
        x = self.enc(x)
        mu, logvar = self.mean(x), self.std(x)
        return mu, logvar

    def decoder(self, z):
        return self.dec(z)
    
class Geometric_VAE_NTD(VAE):
    def __init__(self, dataset='Geometric Shapes', input_channels=1, input_size=96, rank=[8, 8, 64], name='NTD', 
                 train_losses=None, val_losses=None, train_IoU=None, val_IoU=None,  
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        if train_losses is None:
            train_losses = []
        if val_losses is None:
            val_losses = []
        if train_IoU is None:
            train_IoU = []
        if val_IoU is None:
            val_IoU = []

        super(Geometric_VAE_NTD, self).__init__(dataset, input_channels, input_size, rank, name,
                                                train_losses, val_losses, train_IoU, val_IoU, device)

        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=5, stride=2, padding=2),  # Output: (8, 48, 48)
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # Output: (16, 24, 24)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Output: (32, 12, 12)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: (64, 6, 6)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: (128, 3, 3)
            nn.ReLU(),
            nn.Flatten()
        ).to(self.device)

        # Mean and Log Variance
        self.mean = nn.Linear(128 * 3 * 3, rank[2])
        self.std = nn.Linear(128 * 3 * 3, rank[2])

        # Parameters (Weights)
        self.G3 = nn.Parameter(torch.empty(rank[2], rank[0] * rank[1], device=self.device), requires_grad=True)
        nn.init.kaiming_uniform_(self.G3, mode='fan_out', nonlinearity='relu')
        self.W = nn.Parameter(torch.empty(input_size, rank[0], device=self.device), requires_grad=True)
        nn.init.kaiming_uniform_(self.W, mode='fan_out', nonlinearity='relu')
        self.H = nn.Parameter(torch.empty(input_size, rank[1], device=self.device), requires_grad=True)
        nn.init.kaiming_uniform_(self.H, mode='fan_out', nonlinearity='relu')

        # Reconstruction
        self.rec_q = nn.Unflatten(1, (1, rank[2]))
        self.rec = nn.Unflatten(2, (input_size, input_size))

    def encoder(self, x):
        x = self.enc(x)
        mu, logvar = self.mean(x), self.std(x)
        return mu, logvar

    def decoder(self, z):
        qg = F.relu(torch.matmul(self.rec_q(z), self.G3.to(self.device)))
        wh = torch.kron(self.W.T, self.H.T)
        x_hat = F.sigmoid(torch.matmul(qg, wh))
        return self.rec(x_hat)
    
class Geometric_VAE_CPD_NN(VAE):
    def __init__(self, dataset='Geometric Shapes', input_channels=1, input_size=96, rank=64, name='CPD-NN', 
                 train_losses=None, val_losses=None, train_IoU=None, val_IoU=None,  
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), encod='Neuronal'):
        if train_losses is None:
            train_losses = []
        if val_losses is None:
            val_losses = []
        if train_IoU is None:
            train_IoU = []
        if val_IoU is None:
            val_IoU = []

        super(Geometric_VAE_CPD_NN, self).__init__(dataset, input_channels, input_size, rank, name,
                                                   train_losses, val_losses, train_IoU, val_IoU, device)

        # Encoder type
        self.encod = encod
        # Encoder
        self.fl = nn.Flatten()
        if encod == 'Neuronal':
            self.enc = nn.Sequential(
                nn.Conv2d(input_channels, 8, kernel_size=5, stride=2, padding=2),  # Output: (8, 48, 48)
                nn.ReLU(),
                nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # Output: (16, 24, 24)
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Output: (32, 12, 12)
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: (64, 6, 6)
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: (128, 3, 3)
                nn.ReLU(),
                nn.Flatten()
            ).to(self.device)

            # Mean and Log Variance
            self.mean = nn.Linear(128 * 3 * 3, rank)
            self.std = nn.Linear(128 * 3 * 3, rank)
            
        else:
            # Mean and Std
            self.mean = nn.Linear(rank, rank)
            self.std = nn.Linear(rank, rank)
            
        self.rec_c = nn.Unflatten(1, (1, rank))
        self.rec = nn.Unflatten(2, (input_size, input_size))

        # Parameters (Weights)
        self.A = nn.Parameter(torch.empty(input_size, rank, device=self.device), requires_grad=True)
        nn.init.kaiming_uniform_(self.A, mode='fan_out', nonlinearity='relu')
        self.B = nn.Parameter(torch.empty(input_size, rank, device=self.device), requires_grad=True)
        nn.init.kaiming_uniform_(self.B, mode='fan_out', nonlinearity='relu')

        if encod == 'Matrices':
            self.Y = nn.Parameter(torch.empty(input_size, rank, device=self.device), requires_grad=True)
            nn.init.kaiming_uniform_(self.Y, mode='fan_out', nonlinearity='relu')
            self.Z = nn.Parameter(torch.empty(input_size, rank, device=self.device), requires_grad=True)
            nn.init.kaiming_uniform_(self.Z, mode='fan_out', nonlinearity='relu')

        # Activation function
        self.sig = nn.ReLU()

    def encoder(self, x):
        if self.encod == 'Neuronal':
            c = self.enc(x)
        else:
            if self.encod == 'Matrices':
                term = khatri_rao_tensor(self.Y.to(self.device), self.Z.to(self.device))
            elif self.encod == 'Symmetric':
                term = tr_pi_tensor(self.A.to(self.device), self.B.to(self.device))
            else:
                raise ValueError("Invalid encoder. Please choose from 'Neuronal', 'Matrices', or 'Symmetric'.")
            
            prod = torch.matmul(self.fl(x), term)
            c = self.sig(prod)
            c = self.fl(c)
        
        mu, logvar = self.mean(c), self.std(c)
        return mu, logvar
        
    def decoder(self, z):
        term = khatri_rao_tensor(self.sig(self.A), self.sig(self.B)).T
        x_hat = torch.matmul(self.rec_c(z), term.to(self.device))
        return torch.sigmoid(self.rec(x_hat))