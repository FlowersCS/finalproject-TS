import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.dl_models import CNNModel
from sklearn.metrics import f1_score, balanced_accuracy_score
from models.utils import train_one_epoch, evaluate, kfold

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

import wandb
# init wandb
epochs = 10
batch_size = 128

wandb.init(project="Final-Project-TimeSeries-UTEC", name="CNN-temp-spec", config={
    "epochs": epochs,
    "batch_size": batch_size,
    "model": "CNN",
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

# Para CNN: (batch, channels=1, length)
X_temp_train_np = X_temp_train.to_numpy().reshape(-1, 1, X_temp_train.shape[1])
X_temp_test_np  = X_temp_test.to_numpy().reshape(-1, 1, X_temp_test.shape[1])
y_temp_train_np = y_temp_train.to_numpy()
y_temp_test_np  = y_temp_test.to_numpy()

X_spec_train_np = X_spec_train.to_numpy().reshape(-1, 1, X_spec_train.shape[1])
X_spec_test_np  = X_spec_test.to_numpy().reshape(-1, 1, X_spec_test.shape[1])
y_spec_train_np = y_spec_train.to_numpy()
y_spec_test_np  = y_spec_test.to_numpy()

print("KFold en train CNN temporales")
kfold(
    X_temp_train_np, y_temp_train_np,
    model_class=CNNModel,
    model_kwargs={
        'input_channels': 1,
        'num_classes': len(np.unique(y_temp_train)),
        'input_features': X_temp_train.shape[1] 
    },
    epochs=epochs,
    batch_size=batch_size,
    n_splits=5,
    device=device
)

print("KFold en train CNN espectrales")
kfold(
    X_spec_train_np, y_spec_train_np,
    model_class=CNNModel,
    model_kwargs={
        'input_channels': 1,
        'num_classes': len(np.unique(y_spec_train)),
        'input_features': X_spec_train.shape[1] 
    },
    epochs=epochs,
    batch_size=batch_size,
    n_splits=5,
    device=device
)


# --- Entrenamiento final y evaluación en test (temporales)---
print("Entrenamiento final y evaluación en test (temporales)")
train_ds = TensorDataset(torch.FloatTensor(X_temp_train_np), torch.LongTensor(y_temp_train_np))
test_ds  = TensorDataset(torch.FloatTensor(X_temp_test_np),  torch.LongTensor(y_temp_test_np))
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=batch_size*2)

model = CNNModel(input_channels=1, num_classes=len(np.unique(y_temp_train)), input_features=X_temp_train.shape[1]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    train_one_epoch(model, train_loader, criterion, optimizer, device, epoch=epoch)

preds, labels = evaluate(model, test_loader, device)
f1 = f1_score(labels, preds, average='weighted')
bal_acc = balanced_accuracy_score(labels, preds)
print("Test F1-score:", f1)
print("Test Balanced accuracy:", bal_acc)
wandb.log({"test_f1": f1, "test_bal_acc": bal_acc})


# --- Entrenamiento final y evaluación en test (espectrales)---
print("Entrenamiento final y evaluación en test (espectrales)")
train_ds = TensorDataset(torch.FloatTensor(X_spec_train_np), torch.LongTensor(y_spec_train_np))
test_ds  = TensorDataset(torch.FloatTensor(X_spec_test_np),  torch.LongTensor(y_spec_test_np))
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=batch_size*2)

model = CNNModel(input_channels=1, num_classes=len(np.unique(y_spec_train)), input_features=X_spec_train.shape[1]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    train_one_epoch(model, train_loader, criterion, optimizer, device, epoch=epoch)

preds, labels = evaluate(model, test_loader, device)
f1 = f1_score(labels, preds, average='weighted')
bal_acc = balanced_accuracy_score(labels, preds)
print("Test F1-score:", f1)
print("Test Balanced accuracy:", bal_acc)
wandb.log({"test_f1": f1, "test_bal_acc": bal_acc})

wandb.finish()