# guarda modelos serializados
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.dl_models import MLPModel 
from sklearn.metrics import f1_score, balanced_accuracy_score
from models.utils import train_one_epoch, evaluate, kfold

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

import wandb
# init wandb
epochs = 10
batch_size = 128

wandb.init(project="Final-Project-TimeSeries-UTEC", name="MLP-temp-spec", config={
    "epochs": epochs,
    "batch_size": batch_size,
    "model": "MLP",
})


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

project_root = os.path.dirname(os.getcwd())
data_path = os.path.join(project_root,'finalprojectTS', 'data')
processed_path = data_path + '/processed'
save_models_path = os.path.join(project_root, 'finalprojectTS', 'models', 'save_models')

features_temporal = pd.read_csv(processed_path + '/features_temporales_labelNum_overlap50.csv')
features_espectrales = pd.read_csv(processed_path + '/features_espectrales_labelNum_overlap50.csv')

# los labels estan en features_temporal en la ultima columna
X_temp = features_temporal.iloc[:, 1:-1]  # Todas las columnas excepto la primera y la última
y_temp = features_temporal.iloc[:, -1]   # Solo la última columna

X_spec = features_espectrales.iloc[:, 1:-1]  # Todas las columnas excepto la última
y_spec = features_espectrales.iloc[:, -1]   # Solo la última columna

# Dividir el dataset en entrenamiento y prueba
X_temp_train, X_temp_test, y_temp_train, y_temp_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
X_spec_train, X_spec_test, y_spec_train, y_spec_test = train_test_split(X_spec, y_spec, test_size=0.2, random_state=42)

## --- K-Fold cross-validation SOLO en train ---
print("K-Fold en train MLP temporales:")
kfold(
    X_temp_train.values, y_temp_train.values,
    model_class=MLPModel,
    model_kwargs={'input_size': X_temp_train.shape[1], 'num_classes': 6},
    epochs=5,
    batch_size=128,
    n_splits=5,
    device=device
)

print("K-Fold en train MLP espectrales:")
kfold(
    X_spec_train.values, y_spec_train.values,
    model_class=MLPModel,
    model_kwargs={'input_size': X_spec_train.shape[1], 'num_classes': 6},
    epochs=5,
    batch_size=128,
    n_splits=5,
    device=device
)

# --- Entrenamiento final y evaluación en test (temporales)---
print("\nEntrenamiento final y evaluación en test MLP (temporales):")
model = MLPModel(input_size=X_temp_train.shape[1], num_classes=6).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_temp_train.values), torch.LongTensor(y_temp_train.values)), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_temp_test.values), torch.LongTensor(y_temp_test.values)), batch_size=batch_size*2)

for epoch in range(epochs):
    train_one_epoch(model, train_loader, criterion, optimizer, device, epoch=epoch)

preds, labels = evaluate(model, test_loader, device)
f1 = f1_score(labels, preds, average='weighted')
bal_acc = balanced_accuracy_score(labels, preds)
wandb.log({"test_f1": f1, "test_balanced_accuracy": bal_acc})

# --- Entrenamiento final y evaluación en test (espectrales) ---
print("\nEntrenamiento final y evaluación en test MLP(espectrales):")
model = MLPModel(input_size=X_spec_train.shape[1], num_classes=6).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_spec_train.values), torch.LongTensor(y_spec_train.values)), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_spec_test.values), torch.LongTensor(y_spec_test.values)), batch_size=batch_size*2)

for epoch in range(epochs):
    train_one_epoch(model, train_loader, criterion, optimizer, device, epoch=epoch)

preds, labels = evaluate(model, test_loader, device)
f1 = f1_score(labels, preds, average='weighted')
bal_acc = balanced_accuracy_score(labels, preds)
wandb.log({"test_f1": f1, "test_balanced_accuracy": bal_acc})

wandb.finish()