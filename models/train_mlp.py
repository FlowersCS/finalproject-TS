# guarda modelos serializados
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.dl_models import MLPModel 
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
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
save_results_path = os.path.join(project_root, 'finalprojectTS', 'models', 'results')
os.makedirs(save_models_path, exist_ok=True)
os.makedirs(save_results_path, exist_ok=True)

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
f1_scores_temp, bal_acc_temp = kfold(
    X_temp_train.values, y_temp_train.values,
    model_class=MLPModel,
    model_kwargs={'input_size': X_temp_train.shape[1], 'num_classes': 6},
    epochs=5,
    batch_size=128,
    n_splits=7,
    device=device
)

print("K-Fold en train MLP espectrales:")
f1_scores_spec, bal_acc_spec = kfold(
    X_spec_train.values, y_spec_train.values,
    model_class=MLPModel,
    model_kwargs={'input_size': X_spec_train.shape[1], 'num_classes': 6},
    epochs=5,
    batch_size=128,
    n_splits=7,
    device=device
)

# Guardar los resultados de K-Fold en csv (temporales)
csv_path_cvTemp = os.path.join(save_results_path,'resultados_cv_temporales.csv')
df_cvTemp = pd.read_csv(csv_path_cvTemp)

df_temp = pd.concat([df_cvTemp, pd.DataFrame([{
    "Modelo": "MLP",
    "F1-score": np.mean(f1_scores_temp),
    "Balanced accuracy": np.mean(bal_acc_temp),
}])], ignore_index=True)

df_temp.to_csv(csv_path_cvTemp, index=False)

# Guardar los resultados de K-Fold en csv (espectrales)
csv_path_cvSpec = os.path.join(save_results_path,'resultados_cv_espectrales.csv')
df_cvSpec = pd.read_csv(csv_path_cvSpec)

df_spec = pd.concat([df_cvSpec, pd.DataFrame([{
    "Modelo": "MLP",
    "F1-score": np.mean(f1_scores_spec),
    "Balanced accuracy": np.mean(bal_acc_spec),
}])], ignore_index=True)

df_spec.to_csv(csv_path_cvSpec, index=False)


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
acc = accuracy_score(labels, preds)
print("Test F1-score:", f1)
print("Test Balanced accuracy:", bal_acc)
print("Test Accuracy:", acc)
wandb.log({"test_f1": f1, "test_balanced_accuracy": bal_acc, "test_accuracy": acc})

# Guardar el modelo entrenado
model_save_path = os.path.join(save_models_path, "mlp_features_temporales.pkl")
torch.save(model.state_dict(), model_save_path)

csv_test_temp = os.path.join(save_results_path, "resultados_test_temporales.csv")
df_test_temp = pd.read_csv(csv_test_temp)
df_test_temp = pd.concat([df_test_temp, pd.DataFrame([{
    "Modelo": "MLP",
    "F1-score": f1,
    "Balanced accuracy": bal_acc,
    "Accuracy": acc,
    "Archivo": "mlp_features_temporales.pkl"
}])], ignore_index=True)
df_test_temp.to_csv(csv_test_temp, index=False)

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
acc = accuracy_score(labels, preds)
print("Test F1-score:", f1)
print("Test Balanced accuracy:", bal_acc)
print("Test Accuracy:", acc)
wandb.log({"test_f1": f1, "test_balanced_accuracy": bal_acc, "test_accuracy": acc})

# Guardar el modelo entrenado
model_save_path = os.path.join(save_models_path, "mlp_features_espectrales.pkl")
torch.save(model.state_dict(), model_save_path)

csv_test_spec = os.path.join(save_results_path, "resultados_test_espectrales.csv")
df_test_spec = pd.read_csv(csv_test_spec)
df_test_spec = pd.concat([df_test_spec, pd.DataFrame([{
    "Modelo": "MLP",
    "F1-score": f1,
    "Balanced accuracy": bal_acc,
    "Accuracy": acc,
    "Archivo": "mlp_features_espectrales.pkl"
}])], ignore_index=True)
df_test_spec.to_csv(csv_test_spec, index=False)

wandb.finish()