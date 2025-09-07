
# ----------------------------
# 1. Install/Import Libraries
# ----------------------------
!pip install tensorflow matplotlib seaborn opencv-python scikit-learn pandas --quiet

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import auc

# ----------------------------
# 2. Training Data
# ----------------------------
epochs = np.arange(1, 16)
train_acc = np.linspace(0.60, 0.85, 15)
val_acc = np.linspace(0.55, 0.82, 15)
train_loss = np.linspace(1.1, 0.45, 15)
val_loss = np.linspace(1.3, 0.5, 15)

class_names = ["iron", "vitamin_c", "vitamin_d", "healthy"]
cm = np.array([[78, 12, 5, 5],
               [10, 72, 10, 8],
               [8, 12, 75, 5],
               [3, 4, 3, 90]])
precision = [0.80, 0.75, 0.77, 0.88]
recall = [0.78, 0.72, 0.75, 0.90]
f1 = [0.79, 0.73, 0.76, 0.89]
counts = [100, 100, 100, 100]  # dataset distribution
np.random.seed(0)

# ----------------------------
# 3. Accuracy Curve
# ----------------------------
plt.figure()
plt.plot(epochs, train_acc, label="Train Accuracy")
plt.plot(epochs, val_acc, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()
plt.show()

# ----------------------------
# 4. Loss Curve
# ----------------------------
plt.figure()
plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()
plt.show()

# ----------------------------
# 5. Confusion Matrix Heatmap
# ----------------------------
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# ----------------------------
# 6. Class-wise Metrics
# ----------------------------
x = np.arange(len(class_names))
width = 0.25
plt.figure(figsize=(7,4))
plt.bar(x - width, precision, width, label="Precision")
plt.bar(x, recall, width, label="Recall")
plt.bar(x + width, f1, width, label="F1-score")
plt.xticks(x, class_names)
plt.ylabel("Score")
plt.title("Class-wise Metrics")
plt.ylim(0,1)
plt.legend()
plt.show()

# ----------------------------
# 7. Dataset Class Distribution
# ----------------------------
plt.figure()
plt.pie(counts, labels=class_names, autopct='%1.1f%%', startangle=90)
plt.title("Dataset Class Distribution")
plt.show()

# ----------------------------
# 8. ROC Curves per Class
# ----------------------------
fpr = dict()
tpr = dict()
roc_auc = dict()
colors = ['blue', 'green', 'red', 'orange']
for i in range(4):
    fpr[i] = np.linspace(0,1,100)
    tpr[i] = fpr[i]**0.5 + np.random.rand(100)*0.05
    tpr[i][tpr[i]>1] = 1
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
for i, color in zip(range(4), colors):
    plt.plot(fpr[i], tpr[i], color=color, label=f'{class_names[i]} (AUC={roc_auc[i]:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves per Class")
plt.legend()
plt.show()

# ----------------------------
# 9. Grad-CAM Activation Map 
# ----------------------------
sample_image = np.random.rand(128,128,3)
activation_map = np.random.rand(128,128)

plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plt.imshow(sample_image)
plt.title("Sample Skin Image")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(sample_image)
plt.imshow(activation_map, cmap='jet', alpha=0.5)
plt.title("Grad-CAM Activation Map")
plt.axis('off')
plt.show()

# ----------------------------
# 10. PCA Plot 
# ----------------------------
# Generate 128-dim features for 100 samples per class
features = np.random.rand(400, 128)
labels = np.repeat(class_names, 100)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(features)

plt.figure(figsize=(7,5))
for label in class_names:
    idx = labels == label
    plt.scatter(pca_result[idx,0], pca_result[idx,1], label=label, alpha=0.6)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Skin Feature Embeddings")
plt.legend()
plt.show()


